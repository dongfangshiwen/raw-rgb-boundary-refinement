"""Run the group-held-out MUSES selector experiment and replay its raw counts."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.recording_gate import gate
from experiments.recording_gate.gate import OUT, PROJECT, ARMS, MODES, SCALES, digest, read, write


def register(rows):
    src = PROJECT / "benchmark_data/muses-v70"
    from boundary_refinement.evaluation.controls.audit_dual_output import AUTHOR_HASH
    sources = {Path(m.__file__).name: digest(m.__file__) for m in
               (gate, gate.ev, gate.ev.seg, gate.ev.setup.peer, __import__("boundary_refinement.models.operators.range_kernel", fromlist=["prepare"]))}
    spec = {
        "design": "Source-only training on unadapted released LiDSeg scores. Preserve each official score map; add "
                  "the parameter-free four-range correction. At top-class disagreements choose official or corrected label. "
                  "No MUSES-fitted decoder, detail adapter, restored RGB, or target logits participate in fitting.",
        "sample": "96 human-labeled fog images, nine source recordings. Leave each entire recording out; also fit all nine. "
                  "Other MUSES splits (24 calibration and 58 official validation) are disjoint from training by recording. "
                  "They were inspected in historical research. These are replication/development data, not new geography.",
        "targets": "Freeze all-nine weights before target inference. Evaluate both Mask2Former and SegFormer-B1 on the same "
                   "82 frames from nine other recordings as a pooled analysis, preserving the two split strata, and list split scores.",
        "primary": "At 512, native six-pixel reference-band 19-class mIoU. Independent benefit of aligned RGB in the selector "
                   "requires positive paired lower 95% bounds against always-apply fixed correction, fitted global, and "
                   "equal-capacity shifted input on BOTH transfer backbones. Report semantic-only control, 1024 and full image "
                   "as secondary. Do not select models after viewing targets. Report rescues, damage and incorrect-to-incorrect changes.",
        "proposal": "Frozen range strengths 6*sigmoid(-0.5), spatial power 1, radii .02/.05/.10/.20; RGB quantized at working size. "
                    "Candidates are the native released label and argmax of released log scores plus the native fixed colour correction.",
        "selector": "t=m0/(m0+m1), clipped to 1e-5,1-1e-5, where m0 and m1 are the opposing top-label margins of each candidate. "
                    "Accept the proposed label when linear_score > logit(t). Features: intercept, signed input scale, bilinear "
                    "coarse donor vote for new minus old, nearest matching donor's relative RGB MSE for old vs new. Set the RGB "
                    "cue to zero when either label has no contributing donor. The negative control rolls only the selector's "
                    "guide by half of the working grid on both axes; the fixed proposal is identical across arms.",
        "fit": "Only disputed pixels on which either candidate is correct. Up to 1536 sampled without replacement per frame and "
               "scale using SHA256(id/side). Equal recording weight and equal frame-scale weight within recording. "
               "CUDA float64 convex logistic objective with fixed logit(t) offset, L2=0.002 on every coefficient, Newton with "
               "line search, gradient convergence, no tuning or early stopping from evaluation labels.",
        "statistics": "Pool integer confusion counts, fixed 19 classes with zero for zero union. 5000 paired group percentile "
                      "bootstrap samples, RNG 11222. Stratify the pooled MUSES transfer draws by prior split (4 + 5 drives). "
                      "Pointwise intervals conditional on the fitted rule. Treat source LOO as descriptive.",
        "sources": sources, "source_manifests": {s: digest(src / f"{s}.json") for s in rows},
        "frames": {s: [{k: r[k] for k in ("group","id")} for r in rr] for s, rr in rows.items()},
        "checkpoints": {"lidseg": AUTHOR_HASH, "mask2former": gate.ev.masknet.SHA, "segformer": gate.ev.seg.HASHES["pytorch_model.bin"]},
        "targets_input_long_edges": list(SCALES), "arms": ARMS, "scoring_code": digest(__file__),
    }
    file = OUT / "protocol.json"
    if file.exists():
        assert read(file) == spec
    else:
        write(file, spec)
    return digest(file)


def make_bank(rows, kernel, source, lock):
    file = OUT / "source-samples.npz"
    if file.is_file():
        with np.load(file, allow_pickle=False) as f:
            assert f["lock"].item() == lock
            return f["features"], f["target"], f["keys"], f["records"], f["lengths"]
    features, labels, keys, records, lengths = [], [], [], [], []
    for i, row in enumerate(rows):
        rgb, y = gate.open_muses(row)
        for side in SCALES:
            key = row["id"] + "/" + str(side)
            x, target = gate.packet(source, kernel, rgb, y, None, side, sample_key=key)
            features.append(x)
            labels.append(target)
            keys.append(key)
            records.append(row["group"])
            lengths.append(len(target))
        if (i+1) % 16 == 0:
            print("source images for fitting", i+1, len(rows), flush=True)
    packed = np.concatenate(features), np.concatenate(labels), np.array(keys), np.array(records), np.array(lengths)
    gate.save_array(file, features=packed[0], target=packed[1], keys=packed[2], records=packed[3], lengths=packed[4], lock=lock)
    return packed


def optimize(x, offset, y, weights):
    reg = .002
    theta = x.new_zeros(x.shape[1])
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    def objective(t):
        z = x @ t - offset
        return (weights*(torch.nn.functional.softplus(z)-z*y)).sum() + .5*reg*t.square().sum()
    for step in range(40):
        p = (x @ theta - offset).sigmoid()
        grad = x.T @ (weights*(p-y)) + reg*theta
        err = grad.abs().max().item()
        if err < 1e-8:
            break
        hessian = x.T @ ((weights*p*(1-p))[:,None]*x) + reg*eye
        direction = torch.linalg.solve(hessian, grad)
        value = objective(theta)
        rate = 1.
        while objective(theta - rate*direction) > value - .0001*rate*torch.dot(grad, direction):
            rate *= .5
            if rate < 1e-9:
                raise ArithmeticError("Line search failed")
        theta -= rate * direction
    if err > 1e-7:
        raise ArithmeticError(f"Logistic fit did not converge: {err}")
    return {"weights": theta.tolist(), "objective": objective(theta).item(), "gradient_max": err, "updates": step}


def fit(rows, kernel, source, lock):
    target = OUT / "weights.json"
    if target.is_file():
        card = read(target)
        assert card["protocol_sha256"] == lock and card["samples_sha256"] == digest(OUT / "source-samples.npz")
        return card["fits"]
    features, labels, keys, records, lengths = make_bank(rows, kernel, source, lock)
    assert len(keys) == len(set(keys.tolist())) == 192 and len(labels) == lengths.sum()
    group_names = np.unique(records)
    assert len(group_names) == 9
    valid_per_group = {g: np.count_nonzero((records == g) & (lengths > 0)) for g in group_names}
    assert min(valid_per_group.values()) > 0
    weight = np.repeat([1/max(1, n)/valid_per_group[g] for n, g in zip(lengths, records)], lengths)
    x = torch.from_numpy(features).cuda().double()
    y = torch.from_numpy(labels).cuda().double()
    all_weights = torch.from_numpy(weight).cuda().double()
    group_ids = np.repeat(records, lengths)
    fits = {}
    for held in [*group_names, "all"]:
        use = torch.from_numpy(group_ids != held).cuda()
        xx, yy, w = x[use], y[use], all_weights[use]
        w = w / w.sum()
        fits[held] = {}
        for mode, cols in MODES.items():
            design = torch.cat((torch.ones_like(xx[:, :1]), xx[:, cols]), dim=1)
            fits[held][mode] = optimize(design, xx[:, 0], yy, w)
    write(target, {"protocol_sha256":lock, "samples_sha256":digest(OUT/"source-samples.npz"), "fits":fits,
                   "informative_sampled_pixels":len(features), "fit_count":sum(map(len,fits.values())),
                   "source_only":True, "maximum_gradient":max(v["gradient_max"] for f in fits.values() for v in f.values())})
    print("fitted", len(fits)*len(MODES), "models; source pixel sample", len(features), flush=True)
    return fits


def evaluate_panel(name, model, kernel, items, fits, lock, loader=gate.open_muses):
    file = OUT / f"counts-{name}.npz"
    ids = [r["id"] for _,r in items]
    assert len(ids) == len(set(ids))
    shape = (len(items), len(SCALES), 2, len(ARMS))
    counts = np.zeros((*shape,19,19), np.int32)
    edits = np.zeros((*shape,3), np.int64)
    start = 0
    weights_hash = digest(OUT / "weights.json")
    if file.is_file():
        with np.load(file, allow_pickle=False) as f:
            assert f["lock"].item() == lock and f["weights"].item() == weights_hash and f["ids"].tolist() == ids
            start = f["finished"].item()
            counts[:start], edits[:start] = f["counts"][:start], f["edits"][:start]
    for i in range(start,len(items)):
        split, r = items[i]
        rgb, y = loader(r)
        edge = gate.boundary_band(y)
        fold = r["group"] if name == "lidseg-loo" else "all"
        for k, side in enumerate(SCALES):
            base, pack, f = gate.packet(model,kernel,rgb,y,edge,side)
            counts[i,k], edits[i,k] = gate.measure(base,pack,f,fits[fold])
        if i == 0 or (i+1)%16 == 0 or i+1 == len(items):
            gate.save_array(file, counts=counts, edits=edits, ids=ids, groups=[r["group"] for _,r in items],
                            splits=[s for s,_ in items], finished=i+1, lock=lock, arms=ARMS, scales=SCALES, weights=weights_hash)
            print(name, i+1, "/", len(items), flush=True)


def summary_for(raw, edits, groups, splits, classes, descriptive=False):
    names = np.unique(groups)
    bank = np.stack([raw[groups == g].sum(0, dtype=np.int64) for g in names])
    edit_counts = edits.sum(0, dtype=np.int64)
    strata = np.array([np.unique(splits[groups == g]).item() for g in names])
    hit = bank.diagonal(axis1=-2,axis2=-1)
    union = bank.sum(-1)+bank.sum(-2)-hit
    stat = np.stack((hit[...,classes], union[...,classes]),-1)
    mat = stat.reshape(len(names), -1).astype(float)
    def score(w):
        a = (w@mat).reshape(len(w),*stat.shape[1:])
        return 100*np.divide(a[...,0],a[...,1],out=np.zeros_like(a[...,0]),where=a[...,1]>0).mean(-1)
    absval = score(np.ones((1,len(names))))[0]
    boot = None
    if not descriptive and len(names)>=6:
        rng=np.random.default_rng(11222)
        draws=np.zeros((5000,len(names)), np.int64)
        for s in sorted(set(strata)):
            idx=np.flatnonzero(strata==s)
            draws[:,idx]=rng.multinomial(len(idx), np.full(len(idx),1/len(idx)), len(draws))
        boot=np.concatenate([score(w) for w in np.array_split(draws,50)])
    report={}
    for k,side in enumerate(SCALES):
        report[str(side)]={}
        for j,reg in enumerate(("full","boundary")):
            outcomes={}
            for m,arm in enumerate(ARMS):
                r,h,o=edit_counts[k,j,m].tolist()
                outcomes[arm]={"miou":float(absval[k,j,m]),"rescued":r,"harmed":h,"both_wrong":o}
            contrasts={}
            for new in ("fixed","aligned"):
                m=ARMS.index(new)
                for old in (["official"] if new=="fixed" else [a for a in ARMS if a!=new]):
                    n=ARMS.index(old)
                    ci=None if boot is None else np.quantile(boot[:,k,j,m]-boot[:,k,j,n],[.025,.975]).tolist()
                    contrasts[f"{new} minus {old}"]={"pp":float(absval[k,j,m]-absval[k,j,n]),"ci95":ci}
            report[str(side)][reg]={"arms":outcomes,"contrasts":contrasts}
    return {"frames":len(raw),"groups":names.tolist(),"statistics":report}


def rescore(panels, outfile, classes=tuple(range(19))):
    report={}
    for name in panels:
        fpath=OUT/f"counts-{name}.npz"
        with np.load(fpath, allow_pickle=False) as f:
            assert f["finished"].item()==len(f["ids"]) and f["weights"].item()==digest(OUT/"weights.json")
            assert f["arms"].tolist()==ARMS and f["scales"].tolist()==list(SCALES)
            r,e,g,s=f["counts"],f["edits"],f["groups"],f["splits"]
        assert np.all(r.sum(-1)==r[:,:,:,:1].sum(-1))
        changes=r.diagonal(axis1=-1,axis2=-2).sum(-1)
        np.testing.assert_array_equal(changes-changes[:,:,:,:1],e[...,0]-e[...,1])
        res=summary_for(r,e,g,s,classes,name=="lidseg-loo")
        res["counts_sha256"]=digest(fpath)
        if len(set(s))>1:
            res["by_split"]={v:summary_for(r[s==v],e[s==v],g[s==v],s[s==v],classes,True) for v in np.unique(s)}
        report[name]=res
        for side in SCALES:
            print(name,side,res["statistics"][str(side)]["boundary"]["contrasts"]["aligned minus fixed"],flush=True)
    write(OUT/outfile,{"weights_sha256":digest(OUT/"weights.json"), "panels":report})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rescore",action="store_true")
    args=ap.parse_args()
    names=["lidseg-loo","mask2former","segformer"]
    if not args.rescore:
        gate.ev.setup.configure_cuda()
        rows=gate.source_rows()
        lock=register(rows)
        write(OUT/"environment.json",{"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                                     "cudnn":torch.backends.cudnn.version(),"protocol":lock})
        kernel=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
        src=gate.Source()
        fits=fit(rows["fit"],kernel,src,lock)
        evaluate_panel("lidseg-loo",src,kernel,[("fit",r) for r in rows["fit"]],fits,lock)
        del src
        torch.cuda.empty_cache()
        items=[(s,r) for s in ("calibrate","held") for r in rows[s]]
        for name in names[1:]:
            model=gate.ev.Target(name)
            evaluate_panel(name,model,kernel,items,fits,lock)
            del model
            torch.cuda.empty_cache()
    rescore(names,"muses-results.json")

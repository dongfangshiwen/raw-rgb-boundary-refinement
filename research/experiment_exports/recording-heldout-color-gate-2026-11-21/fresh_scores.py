"""Frozen BDD100K score-placement, colour and acceptance comparisons."""
import argparse
import numpy as np
from PIL import Image
import torch

import gate
from fresh_clips import HERE, ROOT, read, put
from scripts.score_path_v104 import metrics

ARMS=["released","paper_fit","released_fit","released_fixed","shifted_fit","aligned_gate","semantic_gate","fixed_gaussian"]
SEEDS=[101,211,307]
TARGETS=["mask2former","segformer"]
PAIRS={"released_fit":["released","paper_fit","released_fixed","shifted_fit","fixed_gaussian"],
       "released_fixed":["released","fixed_gaussian"],"aligned_gate":["released_fixed","semantic_gate"]}


def load_case(row):
    files=[ROOT/p for p in row["paths"]]
    assert [gate.digest(f) for f in files]==row["sha256"]
    with Image.open(files[0]) as im: rgb=np.asarray(im.convert("RGB")).copy()
    with Image.open(files[1]) as im: gt=np.array(im)
    assert sorted(rgb.shape[:2])==[720,1280] and gt.shape==rgb.shape[:2] and set(np.unique(gt))<=set(range(19))|{255}
    return rgb,torch.from_numpy(gt).cuda().long()


@torch.inference_mode()
def evaluate(model, target, kernel, gaussian, learned, gates, rgb, y, edge, metric, side):
    im=gate.image_tensor(rgb,side)
    z,official=model(im,y.shape)
    ev=gate.ev
    coarse=ev.resize(z,y.shape)
    if target=="segformer":
        assert torch.equal(coarse,official)
    guide=im.clamp(0,1).mul(255).round().byte()
    st=ev.prepare(z,guide)
    fixed=ev.resize(kernel.correction(z,st),y.shape)
    score=lambda x:np.stack(metric.score(x.argmax(1)[0]))
    counts=np.empty((2,len(ARMS),len(SEEDS),19,19),np.int32)
    for arm,scores in (("released",official),("released_fixed",official+fixed)):
        counts[:,ARMS.index(arm)]=score(scores)[:,None]
    del fixed
    counts[:,ARMS.index("fixed_gaussian")]=score(official+ev.resize(gaussian.correction(z,st),y.shape))[:,None]
    del st
    for shifted in (False,True):
        colour=torch.roll(guide,(guide.shape[-2]//2,guide.shape[-1]//2),(-2,-1)) if shifted else guide
        stencil=ev.legacy.guide_stencil(colour[[0,0]],z.argmax(1)[0])
        for j,seed in enumerate(SEEDS):
            d=ev.resize(ev.legacy.prediction("dual_free",learned["dual_core",seed],z,stencil),y.shape)
            if not shifted:
                counts[:,ARMS.index("paper_fit"),j]=score(coarse+d)
            counts[:,ARMS.index("shifted_fit" if shifted else "released_fit"),j]=score(official+d)
    if target=="segformer":
        np.testing.assert_array_equal(counts[:,1],counts[:,2])
    del d,official,coarse
    base,pack,features=gate.packet(model,kernel,rgb,y,edge,side)
    tested,edits=gate.measure(base,pack,features,gates)
    for a,b in (("released","official"),("released_fixed","fixed")):
        np.testing.assert_array_equal(counts[:,ARMS.index(a),0],tested[:,gate.ARMS.index(b)])
    for a,b in (("aligned_gate","aligned"),("semantic_gate","semantic")):
        counts[:,ARMS.index(a)]=tested[:,gate.ARMS.index(b)][:,None]
    assert np.all(counts>=0) and np.all(counts[1]<=counts[0]) and np.all(counts.sum(-1)==counts[:,:1,:1].sum(-1))
    return counts,edits


def protocol():
    signatures={p:gate.digest(HERE/p) for p in ("bdd-protocol.json","bdd-evaluation.json","weights.json","gate.py","fresh_scores.py")}
    card={"signatures":signatures,"targets":TARGETS,"arms":ARMS,"seeds":SEEDS,"scales":list(gate.SCALES),"comparisons":PAIRS,
          "gaussian_control":"Prospectively added before weather screening or target evaluation. Same four bilinear donor supports, active coarse cells and released-score placement. Replace the four saturated RGB ranges with a single squared-distance Gaussian with fixed RGB bandwidth 0.05, the established range-decoder initialization. A matched analytical comparison, not an author JBU reproduction."}
    put(HERE/"bdd-scoring-lock.json",card,True)
    return gate.digest(HERE/"bdd-scoring-lock.json")


def run():
    gate.ev.setup.configure_cuda()
    lock=protocol(); rows=read(HERE/"bdd-evaluation.json")["rows"]
    assert len({r["group"] for r in rows})==len(rows)
    learned=gate.ev.original.load()
    accept=read(HERE/"weights.json")["fits"]["all"]
    fixed=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    gaussian=gate.ev.RangeDecoder("bilinear_gaussian").cuda().eval().requires_grad_(False)
    for target in TARGETS:
        model=gate.ev.Target(target)
        dest=HERE/f"bdd-counts-{target}.npz"
        frames=len(rows)
        counts=np.zeros((frames,2,2,len(ARMS),3,19,19),np.int32)
        edits=np.zeros((frames,2,2,len(gate.ARMS),3),np.int64)
        start=0
        if dest.exists():
            with np.load(dest,allow_pickle=False) as saved:
                assert saved["lock"].item()==lock and saved["ids"].tolist()==[r["id"] for r in rows]
                start=saved["finished"].item()
                counts[:start],edits[:start]=saved["counts"][:start],saved["edits"][:start]
        for i in range(start,frames):
            rgb,y=load_case(rows[i])
            edge=gate.boundary_band(y)
            metric=gate.ev.ev.Counts(y)
            assert metric.edge_count==int((edge & (y<19)).sum())
            for j,s in enumerate(gate.SCALES):
                counts[i,j],edits[i,j]=evaluate(model,target,fixed,gaussian,learned,accept,rgb,y,edge,metric,s)
            if i==start or (i+1)%12==0 or i+1==frames:
                gate.save_array(dest,counts=counts,edits=edits,scales=gate.SCALES,arms=ARMS,gate_arms=gate.ARMS,
                                ids=[r["id"] for r in rows],groups=[r["group"] for r in rows],
                                panels=[r["panel"] for r in rows],finished=i+1,lock=lock)
                print(target,i+1,"/",frames,flush=True)
        del model
        torch.cuda.empty_cache()


def score():
    lock=protocol()
    res={}
    for target in TARGETS:
        dest=HERE/f"bdd-counts-{target}.npz"
        with np.load(dest,allow_pickle=False) as f:
            assert f["finished"].item()==len(f["ids"]) and f["lock"].item()==lock
            counts,panels,groups=f["counts"],f["panels"],f["groups"]
            assert len(np.unique(groups))==len(groups) and f["arms"].tolist()==ARMS
        target_report={}
        for panel in ("visible_fog","random"):
            use=panels==panel
            if not use.any():
                target_report[panel]={"groups":0};continue
            data=counts[use].astype(np.int64)
            truth=data.sum(-1)
            assert (truth==truth[:,:,:,:1,:1]).all()
            hit=data.diagonal(axis1=-1,axis2=-2);union=truth+data.sum(-2)-hit
            stats=np.stack((hit,union,truth),-1)
            per_scale={}
            for s,side in enumerate(gate.SCALES):
                reg={}
                for k,name in enumerate(("full","boundary")):
                    out=metrics(stats[:,s,k][:,None],groups[use].tolist(),ARMS,[target],PAIRS,11223)
                    reg[name]=dict(out["targets"][target],classes_with_reference=sum(n>0 for n in out["class_support_pixels"]))
                per_scale[str(side)]=reg
                print(target,panel,side,reg["boundary"]["contrasts"]["released_fit - released"]["pp"],flush=True)
            target_report[panel]={"groups":int(use.sum()),"scales":per_scale}
        res[target]={"counts_sha256":gate.digest(dest),"panels":target_report}
    put(HERE/"bdd-results.json",dict(results=res,scoring_lock=lock,bootstrap=5000))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rescore",action="store_true")
    if not p.parse_args().rescore: run()
    score()

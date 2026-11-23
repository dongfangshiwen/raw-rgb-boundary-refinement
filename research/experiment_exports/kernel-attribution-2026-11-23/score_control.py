"""Evaluate matched source-only refits on previously studied IDD drives."""
from pathlib import Path
from time import monotonic

import numpy as np
import torch

import audit
import summarize as summary
from fit_control import TRAINABLE, STEPS
from scripts import eval_range_v101 as ev

HERE = Path(__file__).resolve().parent
ARMS = ("fixed", *TRAINABLE)


def load_fits(protocol):
    states = {}
    hashes = {}
    for seed in audit.SEEDS:
        path = HERE / f"source-fit-{seed}.json"
        record = ev.read(path)
        assert record["protocol"] == protocol and record["updates_per_arm"] == STEPS
        hashes[str(seed)] = ev.digest(path)
        for arm, info in record["arms"].items():
            model = ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
            model.load_state_dict({k:torch.as_tensor(v) for k,v in info["weights"].items()}, strict=True)
            states[arm, seed] = model
    states["fixed", 0] = ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    return states, hashes


@torch.inference_mode()
def infer(target, rows, models, path, protocol, weights):
    base_model = ev.Target(target)
    with np.load(HERE / f"counts-{target}.npz", allow_pickle=False) as prior:
        fixed_counts = prior["counts"][:,:,2,0]
        ids, groups = prior["ids"], prior["groups"]
    assert ids.tolist() == [r["id"] for r in rows] and groups.tolist() == [r["group"] for r in rows]
    shape = (len(rows),2,len(ARMS),len(audit.SEEDS),19,19)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            assert stored["protocol"].item() == protocol and stored["weights"].tolist() == weights
            assert stored["ids"].tolist() == ids.tolist() and stored["groups"].tolist() == groups.tolist()
            counts, start = stored["counts"], int(stored["finished"])
    else:
        counts, start = np.zeros(shape, dtype=np.int32), 0
    clock = monotonic()
    for i in range(start,len(rows)):
        rgb, truth = ev.scene(rows[i], "idd")
        score = audit.ScoreFrame(truth)
        size = truth.shape
        assert size == (1536,2048)
        image = torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        image = ev.resize(image,(384,512),antialias=True).contiguous()
        logits, released = base_model(image,size)
        del released
        stencil = ev.prepare(logits,image.clamp(0,1).mul(255).round().byte())
        canonical = ev.resize(logits,size)
        def classify(net):
            delta = ev.resize(net.correction(logits,stencil),size)
            return score.confusion(score.labels(canonical + delta))
        fixed = classify(models["fixed",0])
        np.testing.assert_array_equal(fixed, fixed_counts[i], err_msg=f"{target} {ids[i]}")
        counts[i,:,0] = fixed[:,None]
        for k, arm in enumerate(ARMS[1:],start=1):
            for j, seed in enumerate(audit.SEEDS):
                counts[i,:,k,j] = classify(models[arm,seed])
        assert np.all(counts[i].sum(-1) == fixed[:,None,None].sum(-1))
        if (i+1)%32 == 0 or i+1 == len(rows):
            temp = path.with_suffix(".tmp")
            with temp.open("wb") as stream:
                np.savez_compressed(stream,counts=counts,ids=ids,groups=groups,seeds=audit.SEEDS,
                                    arms=ARMS,weights=weights,finished=i+1,protocol=protocol)
            temp.replace(path)
            print(f"{target} {i+1}/{len(rows)} refit comparisons, {monotonic()-clock:.1f}s",flush=True)
    return ids, groups, counts


def stats():
    protocol = ev.digest(HERE / "refit-protocol.json")
    draws = np.random.default_rng(11221).multinomial(24, np.full(24,1/24),size=5000)
    result = {"role":"Exploratory source refitting and repeated evaluation; target drives previously used.",
              "seeds":list(audit.SEEDS), "protocol":protocol, "input_long_edge":512,
              "source_weights_sha256":{str(s):ev.digest(HERE/f"source-fit-{s}.json") for s in audit.SEEDS},
              "evaluation_script_sha256":ev.digest(__file__), "targets":{}, "source_fit_count":9,
              "source_updates":9*STEPS, "source_images":96, "source_recordings":9, "target_images":146,
              "target_recordings":24,"bootstrap":"5000 paired recording resamples; pointwise 95%, mean over three fixed source seeds"}
    target_scores = {}
    for target in audit.TARGETS:
        path = HERE / f"refit-counts-{target}.npz"
        with np.load(path, allow_pickle=False) as pack:
            assert int(pack["finished"]) == 146 and pack["protocol"].item() == protocol
            assert pack["arms"].tolist() == list(ARMS)
            assert pack["weights"].tolist() == [result["source_weights_sha256"][str(s)] for s in audit.SEEDS]
            groups = pack["groups"]
            order = np.unique(groups)
            assert len(order)==24
            grouped = summary.extended(summary.group_sum(pack["counts"],groups,order))
        point, reps = summary.bootstrap(grouped, audit.CLASSES, draws)
        with np.load(HERE / f"counts-{target}.npz",allow_pickle=False) as old:
            assert old["groups"].tolist() == groups.tolist()
            old_grouped = summary.extended(summary.group_sum(old["counts"][:,:,[2,3,4,5]],groups,order))
        prior, prior_reps = summary.bootstrap(old_grouped,audit.CLASSES,draws)
        np.testing.assert_allclose(point[:,0],prior[:,0],rtol=0,atol=1e-11)
        np.testing.assert_allclose(reps[:,:,0],prior_reps[:,:,0],rtol=0,atol=1e-11)
        target_scores[target] = (point, reps)
        regions = {}
        for i,region in enumerate(summary.REGIONS):
            contrasts = {ARMS[j]:summary.ci(point[i,j]-point[i,0],reps[:,i,j]-reps[:,i,0]) for j in (1,2,3)}
            refit_vs_swap = {ARMS[j]:summary.ci(point[i,j]-prior[i,j],reps[:,i,j]-prior_reps[:,i,j]) for j in (1,2,3)}
            regions[region]={"absolute_miou":dict(zip(ARMS,point[i].tolist())),
                             "compared_to_fixed_pp":contrasts,"compared_to_source_parameter_substitution_pp":refit_vs_swap}
        result["targets"][target]={"counts_sha256":ev.digest(path),"regions":regions}
    pm,rm = target_scores["mask2former"]
    ps,rs = target_scores["segformer"]
    result["segformer_minus_mask2former_boundary_effect"] = {
        ARMS[j]:summary.ci((ps[1,j]-ps[1,0])-(pm[1,j]-pm[1,0]),
              (rs[:,1,j]-rs[:,1,0])-(rm[:,1,j]-rm[:,1,0])) for j in (1,2,3)}
    dest = HERE / "refit-results.json"
    if dest.exists():
        assert ev.read(dest) == result
    else:
        audit.write_json(dest,result)
    for name,info in result["targets"].items():
        print(name,info["regions"]["boundary"]["compared_to_fixed_pp"],flush=True)


def main():
    ev.setup.configure_cuda()
    rows, _ = audit.protocol()
    protocol = ev.digest(HERE / "refit-protocol.json")
    models, hashes = load_fits(protocol)
    weights = [hashes[str(s)] for s in audit.SEEDS]
    for target in audit.TARGETS:
        infer(target,rows,models,HERE/f"refit-counts-{target}.npz",protocol,weights)
        torch.cuda.empty_cache()
    stats()


if __name__ == "__main__":
    main()

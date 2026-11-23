"""Freeze the source decoder and ablate fitted colour and spatial parameters on prior cohorts."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image
import torch

from scripts import eval_anchor_v105 as ext
from scripts import range_decoder_v101 as appendix
from scripts import score_path_v104 as score


OUT = ext.WORK / "experiment_exports/fixed-range-audit-v106"
SEEDS = tuple(ext.op.SEEDS)
ARMS = ("released", "fixed4", "fit_range", "fit_spatial", "both_v101", "paper")
CONTRASTS = {"fixed4": ("released",), "fit_range": ("fixed4",), "fit_spatial": ("fixed4",),
             "both_v101": ("fixed4", "fit_spatial", "fit_range"), "paper": ("released", "fixed4", "both_v101")}
PANELS = ("bdd", "idd", "wp")
ROAD = ("mask2former", "segformer")
PLAN = {
    "status": "Secondary analysis using previously examined cohorts, frozen checkpoints, no target fitting or model selection.",
    "question": "Does MUSES training of four colour amplitudes and one spatial exponent add boundary mIoU over the Appendix A.1 fixed multiscale kernel?",
    "primary": "Released manuscript decoder minus fixed four-range kernel, on the pre-existing native 6-pixel GT boundary band. Report full-image mIoU as well.",
    "factorial": "Apply Appendix A.1 RangeDecoder(bilinear_bounded4) in all four cells with the same active lattice, guide and projection. Change only colour weights and/or spatial exponent.",
    "controls": "Raw RGB at all four scales 0.02/0.05/0.10/0.20. Fixed amplitude 6*sigmoid(-0.5) per scale and fixed spatial exponent 1. Fitted values copied by seed, with identical raw guides merged. Report original paper predictions separately.",
    "cohorts": "All 128 latest unrestricted BDD keyframes (19 classes), all 62 prior IDD fog images (17 common classes); all 67 prior WeatherProof fog/mixed images scored with SegFormer-B0 ADE, five direct label matches. Retain the pre-existing audited exclusions and camera merge for reported WeatherProof pools. Report pure fog separately.",
    "uncertainty": "Paired percentile cluster bootstrap, 5000 draws, seed 10517, mean over source seeds 101/211/307 in each draw. The two factorial conditional effects and weather subgroup analyses are descriptive.",
    "verification": "Require exact per-frame native confusion matrices for original released and manuscript outputs versus V105 for both regions and all seeds. Report native prediction differences for the alternate implementation with both parameters fitted.",
}


def rows_for(panel):
    if panel == "idd":
        return ext.prior.rows_for("idd")
    path = ext.OUT / (panel + ".json")
    return score.read(path)["rows"], path


def decoders(checkpoints):
    fixed = appendix.RangeDecoder("bilinear_bounded4").cuda().requires_grad_(False).eval()
    nets = {("fixed4", 0): fixed}
    settings = {}
    for seed in SEEDS:
        checkpoint = checkpoints["dual_core", seed]
        amplitude = (3 * checkpoint.range_weights.sigmoid()).sum(0)
        spatial = checkpoint.temperature.reshape(())
        settings[str(seed)] = {"colour": amplitude.tolist(), "exponent": (.25 + 1.5 * spatial.sigmoid()).item()}
        for arm in ("fit_range", "fit_spatial", "both_v101"):
            net = appendix.RangeDecoder("bilinear_bounded4").cuda().requires_grad_(False).eval()
            if arm in ("fit_range", "both_v101"):
                net.mix.copy_(torch.logit(amplitude / 6))
            if arm in ("fit_spatial", "both_v101"):
                net.spatial.copy_(spatial)
            nets[arm, seed] = net
    return nets, settings


def manuscript_masks(z, small, native, old, anchors, road):
    mask = lambda x: x.argmax(1)[0].byte().cpu().numpy()
    if road:
        stencil = ext.prior.paper.paper_stencil(z, small)
        return [mask(ext.prior.paper.paper_scores(z, stencil, old["dual_core", s], native)) for s in SEEDS]
    st = ext.op.Stencil(z, small)
    ids = st.active.nonzero().flatten()
    if not len(ids):
        return [mask(native)] * len(SEEDS)
    feat = tuple(torch.cat(items) for items in zip(*(st.features(i) for i in ids.split(6144))))
    predictions = []
    for s in SEEDS:
        delta = z.new_zeros(16 * st.h * st.w, st.c)
        delta[ids] = ((anchors[s](feat).softmax(-1) - feat[0])[:, None] @ feat[-1]).squeeze(1)
        grid = delta.reshape(4*st.h, 4*st.w, st.c).permute(2, 0, 1).contiguous()[None]
        predictions.append(mask(native + ext.prior.base.resize(grid, native.shape[-2:])))
    return predictions


def predict(z, released, small, old, anchors, nets, road):
    resize = ext.prior.base.resize
    shape = released.shape[-2:]
    native = resize(z, shape)
    masks = {"released": [released.argmax(1)[0].byte().cpu().numpy()],
             "paper": manuscript_masks(z, small, native, old, anchors, road)}
    grid_shape = tuple(4*n for n in z.shape[-2:])
    guide = small if small.shape[-2:] == grid_shape else resize(small, grid_shape, antialias=True)
    stencil = appendix.prepare(z, guide.clamp(0, 1).mul(255).round().byte())
    for (arm, _), net in nets.items():
        out = native + resize(net.correction(z, stencil), shape)
        masks.setdefault(arm, []).append(out.argmax(1)[0].byte().cpu().numpy())
        del out
    changed = [np.count_nonzero(a != b) for a, b in zip(masks["paper"], masks["both_v101"])]
    return masks, np.array([*changed, np.prod(shape)], dtype=np.int64)


@torch.inference_mode()
def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "protocol.json"
    if protocol.exists():
        assert score.read(protocol) == PLAN
    else:
        score.write(protocol, PLAN)
    ext.prior.base.configure_cuda()
    old, anchors, _ = ext.fit.load()
    nets, settings = decoders(old)
    models = {name: ext.prior.base.prior.Target(name) for name in ROAD}
    models["ade5"] = ext.ADE()
    audit = ext.OUT / "view-audit-decisions.json"
    card = {"settings": settings, "arms": list(ARMS), "seeds": list(SEEDS), "panels": {},
            "hardware": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda}
    start = monotonic()
    for panel in PANELS:
        rows, manifest = rows_for(panel)
        suites = ("ade5",) if panel == "wp" else ROAD
        prior_card = ext.OUT / ("road-recheck-evaluation.json" if panel == "idd" else panel+"-evaluation.json")
        meta = score.read(prior_card)
        files = [protocol, manifest, prior_card, audit, ext.OUT/"training.json", Path(__file__),
                 Path(ext.__file__), Path(appendix.__file__), Path(ext.prior.paper.__file__)]
        lock = sha256("".join(ext.prior.base.digest(f) for f in files).encode()).hexdigest()
        dest = OUT / "counts" / panel
        dest.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            key = row["id"].replace("/", "_") + ".npz"
            file = dest / key
            if file.exists():
                with np.load(file, allow_pickle=False) as f:
                    assert f["lock"].item() == lock and f["id"].item() == row["id"]
                continue
            rgb, ys = ext.prior.old.open_frame(row)
            padded, (h, w) = ext.prior.base.peer_input(rgb)
            small = padded[..., :h, :w].contiguous()
            if panel == "wp":
                raw = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
                small = ext.prior.base.resize(raw, tuple(max(32, min(512, 32*round(d/32))) for d in (h,w)), antialias=True)
                with Image.open(ext.WORK / row["paths"][-1]) as im:
                    y = np.array(im)
                if y.ndim == 3:
                    y = y[:, :, 0]
            source_counts = ext.OUT / ("road-recheck" if panel == "idd" else "counts") / panel / key
            out = {}
            with np.load(source_counts, allow_pickle=False) as previous:
                assert previous["id"].item() == row["id"] and previous["group"].item() == row["group"]
                oldlock = meta["panels"][panel]["lock"] if panel == "idd" else meta["lock"]
                assert previous["lock"].item() == oldlock
                for name in suites:
                    z, released = models[name](small, rgb.shape[:2])
                    masks, differences = predict(z, released, small, old, anchors, nets, panel != "wp")
                    truth, lookup = ext.maps(y, models[name].ids[:5]) if panel == "wp" else (ys[0], None)
                    cm = ext.count(masks, truth, lookup, ARMS)
                    if panel == "idd":
                        history = previous["counts"][meta["targets"].index(name)]
                        oldarms = meta["arms"]
                    else:
                        history, oldarms = previous[name], meta["suites"][name]
                    for arm in ("released", "paper"):
                        np.testing.assert_array_equal(cm[:, ARMS.index(arm)], history[:, oldarms.index(arm)])
                    out[name], out[name+"_diff"] = cm, differences
                    del z, released, masks
            tmp = file.with_suffix(".part")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **out, lock=lock, id=row["id"], group=row["group"], weather=row.get("weather",""))
            tmp.replace(file)
            if i == 0 or (i+1) % 32 == 0 or i+1 == len(rows):
                print(panel, i+1, "/", len(rows), round(monotonic()-start, 1), "s", flush=True)
        card["panels"][panel] = {"lock":lock, "suites":list(suites), "frames":len(rows), "manifest":str(manifest)}
    card["seconds"] = monotonic() - start
    score.write(OUT / "evaluation.json", card)


def summarize():
    card = score.read(OUT / "evaluation.json")
    audit = score.read(ext.OUT / "view-audit-decisions.json")
    result = {"panels": {}, "checks": {}, "bootstrap_draws": 5000, "bootstrap_seed": 10517}
    for panel, metadata in card["panels"].items():
        rows, _ = rows_for(panel)
        for suite in metadata["suites"]:
            by_subset = defaultdict(dict)
            counts = np.zeros(4, np.int64)
            maximum = 0.
            frame_counts = defaultdict(int)
            for row in rows:
                file = OUT / "counts" / panel / (row["id"].replace("/", "_") + ".npz")
                with np.load(file, allow_pickle=False) as f:
                    assert f["lock"].item() == metadata["lock"] and f["group"].item() == row["group"]
                    assert f["id"].item() == row["id"] and f["weather"].item() == row.get("weather", "")
                    cm = f[suite]; diff = f[suite+"_diff"]
                assert cm.shape == (2,len(ARMS),len(SEEDS),19,19) and cm.dtype == np.int32
                counts += diff
                maximum = max(maximum, max(diff[:3]) / diff[3])
                gt = cm.sum(-1); hits = cm.diagonal(axis1=-2, axis2=-1)
                assert (cm >= 0).all() and (cm[1] <= cm[0]).all() and (gt == gt[:,:1,:1]).all()
                stats = np.stack([hits, gt + cm.sum(-2) - hits, gt], -1)
                group = row["group"]
                subsets = [panel]
                if panel == "wp":
                    if group in audit["exclude_new_groups"]:
                        continue
                    group = audit["merge_into"].get(group, group)
                    subsets = (["wp_fog","wp_fog_mixed"] if row["weather"] == "fog" else ["wp_fog_mixed"])
                for tag in subsets:
                    frame_counts[tag] += 1
                    by_subset[tag][group] = by_subset[tag].get(group, 0) + stats
            result["checks"][panel+"_"+suite] = dict(frames=len(rows), changed_by_seed=counts[:3].tolist(),
                    native_pixels_per_seed=int(counts[3]), max_frame_seed_disagreement_fraction=maximum)
            for tag, bank in by_subset.items():
                names = sorted(bank); classes = [c for c in range(19) if c not in (9,16)] if panel == "idd" else list(range(5 if panel == "wp" else 19))
                stats = np.take(np.stack([bank[g] for g in names]), classes, axis=-2)
                res = {name: score.metrics(stats[:,i,None], names, ARMS, [suite], CONTRASTS, 10517)
                       for i,name in enumerate(("full","boundary"))}
                res.update(classes=classes, images=frame_counts[tag], groups=len(bank))
                result["panels"].setdefault(tag, {})[suite] = res
    score.write(OUT / "results.json", result)
    print("Audit scores saved to", OUT / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.score_only:
        evaluate()
    summarize()

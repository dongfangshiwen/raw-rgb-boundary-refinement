"""Frozen native segmentation predictions, paired confusion matrices and direct donor measurements."""
import argparse
from collections import Counter, defaultdict
import gzip
import json

import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.evaluation.semantic import evaluate_transport as prior_eval
from boundary_refinement.models.segmentation import matched_recall as roc
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.training import prototype_risk as method
from boundary_refinement.models.operators import sapa as sapa
from boundary_refinement.models.operators import donor_colors as weighted
from boundary_refinement.data.staging import stage_vkitti_fog as virtual
from boundary_refinement.models.operators import subpixel_transport as base
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band, compress_scores
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT, SEEDS = method.ROOT, method.SEEDS
MODES = (*method.FIXED, *method.RUNS)
REFERENCES = ("bilinear", "trained_mean", "dysample", "sapa_context")
NAMES = (*MODES, *REFERENCES)
PAIRS = (("fixed", "mean"), ("fixed", "rolled"), ("soft", "fixed"),
         ("fixed", "trained_mean"), ("fixed", "dysample"), ("fixed", "sapa_context"),
         ("local_proto", "fixed"), ("local_proto", "constant_proto"), ("local_proto", "local_ce"),
         ("local_ce", "fixed"), ("constant_proto", "fixed"))
STATS = ("owner_percent", "raw_rgb_rmse", "restored_rgb_rmse", "effective_pixels")


def truth_cells(data, truth, grid):
    lo = F.interpolate(truth[None, None].float(), size=tuple(4*s for s in grid), mode="nearest-exact")[0, 0].long()
    owns, count, valid, proto = method.membership(data, lo, *grid)
    labels = base.pack(lo[None], *grid)[..., 0]
    mixed = valid & (labels.amax(1) != labels.amin(1))
    included = mixed & (count > 0)
    subsets = torch.stack((included, included & (count <= 4), included & (count > 4) & (count < 12),
                           included & (count >= 12), valid & (count == 16))).float()
    counts = torch.stack((valid.sum(), mixed.sum(), (mixed & (count == 0)).sum(), included.sum())).tolist()
    return owns, proto, subsets, counts


def measure(data, weights, cells):
    owns, target, subsets, _ = cells
    donor = (data["child"] * weights[..., None]).sum(1)
    error = 255. * (donor - target).reshape(-1, 2, 3).square().mean(-1).sqrt()
    quantities = torch.cat((torch.ones_like(donor[:, :1]), 100. * (weights*owns).sum(1, keepdim=True),
                            error, weights.square().sum(1, keepdim=True).reciprocal()), -1)
    return (subsets.double() @ quantities.double()).tolist()


def draws(n):
    counts = np.ones((1, n), dtype=np.int64)
    if n > 1:
        rng = np.random.default_rng(97037)
        counts = np.vstack((counts, rng.multinomial(n, np.full(n, 1./n), size=2000)))
    return counts


def interval(values):
    return dict(diff=float(values[0]),
                ci95=np.quantile(values[1:], [.025, .975]).tolist() if len(values) > 1 else None)


def scores(rows, classes, key):
    datasets = defaultdict(dict)
    for r in rows:
        if key not in r:
            continue
        name = r["ds"]
        g = roc.capture_id(name, r["group"])
        for ds in ((name, "foggy_driving_101") if name.startswith("foggy_driving_") and key == "counts" else (name,)):
            if g not in datasets[ds]:
                datasets[ds][g] = np.zeros((len(NAMES), 3, 19, 19), dtype=np.int64)
            datasets[ds][g][NAMES.index(r["method"]), SEEDS.index(r["seed"])] += r[key]
    out = {}
    for ds, bank in datasets.items():
        n = len(bank)
        sampled, counted = draws(n), np.stack(list(bank.values()))
        values, lights = {}, {}
        for i, arm in enumerate(NAMES):
            chunks, light_counts = [], None
            for batch in np.array_split(sampled, max(1, len(sampled) // 64)):
                matrix = (batch @ counted[:, i].reshape(n, -1)).reshape(-1, 3, 19, 19)
                diag = matrix.diagonal(axis1=-2, axis2=-1)
                total = matrix.sum(-2) + matrix.sum(-1) - diag
                iou = np.divide(100.*diag, total, out=np.zeros_like(diag, dtype=float), where=total > 0)
                chunks.append(iou[..., list(classes)].mean(-1))
                if light_counts is None:
                    light_counts = iou[0, :, 6].mean()
            values[arm] = np.concatenate(chunks)
            lights[arm] = float(light_counts)
        out[ds] = dict(groups=n, scores={a: v[0].mean() for a, v in values.items()},
                       seed_scores={a: v[0].tolist() for a, v in values.items()}, traffic_light_IoU=lights,
                       effects={a+"-"+b: interval((values[a]-values[b]).mean(1)) for a, b in PAIRS})
    return out


def diagnostics(rows):
    bank = defaultdict(dict)
    coverage = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for r in rows:
        if "donor" not in r:
            continue
        ds, g = r["ds"], roc.capture_id(r["ds"], r["group"])
        if g not in bank[ds]:
            bank[ds][g] = np.zeros((len(MODES), 5, 5))
        bank[ds][g][MODES.index(r["method"])] += r["donor"]
        if r["method"] == "fixed":
            coverage[ds] += r["coverage"]
    result = {}
    for ds, groups in bank.items():
        measured = np.stack(list(groups.values()))
        sample = draws(len(groups))
        measured = (sample @ measured.reshape(len(groups), -1)).reshape(len(sample), len(MODES), 5, 5)
        result[ds] = dict(groups=len(groups), coverage_across_three_predictions=coverage[ds].tolist(), subsets={})
        for k, kind in enumerate(("mixed_owner", "low", "middle", "high", "pure_owner")):
            per = measured[:, :, k, 1:] / np.maximum(measured[:, :, k, :1], 1.)
            result[ds]["subsets"][kind] = dict(
                scores={arm: dict(zip(STATS, per[0, i].tolist())) for i, arm in enumerate(MODES)},
                effects={a+"-"+b: {name: interval(per[:, MODES.index(a), j]-per[:, MODES.index(b), j])
                                     for j, name in enumerate(STATS)}
                         for a, b in PAIRS if a in MODES and b in MODES})
    return result


def cohort(part):
    if part == "new":
        counts = {"new_muses_" + k: v["selected"] for k, v in read(held.SCENES)["cohorts"].items() if v["selected"]}
        return held.scene_cases(), counts, held.SCENES, range(19)
    if part == "virtual":
        return virtual.cases(), read(virtual.HELD)["counts"], virtual.HELD, virtual.CLASSES
    it, expected, source = prior_eval.cases(part)
    return it, expected, source, range(19)


def summarize(part, rows, classes, source):
    path = ROOT / f"{part}-individual.json.gz"
    atomic_write(ROOT / f"{part}-test.json", dict(cohort=digest(source), evaluator=digest(__file__),
                 trained=digest(ROOT / "completed.json"), ledger=digest(path),
                 counts=dict(Counter(r["ds"] for r in rows if r["method"] == "fixed" and r["seed"] == SEEDS[0])),
                 full=scores(rows, classes, "counts"), edge=scores(rows, classes, "edge"),
                 diagnostics=diagnostics(rows), utc=now()))


@torch.inference_mode()
def evaluate(part):
    configure_cuda()
    iterator, expected, source, classes = cohort(part)
    engine, anchors = base.DualScaleFeatures(), base.anchors()
    decoders, gates, older, published = weighted.load(), method.fitted(), base.load(), sapa.nets()
    rows, seen = [], Counter()
    for ds, key, group, _, rgb, truth in iterator:
        views, restored, _ = engine(rgb)
        guides = base.render_guides(rgb, restored, views.fine.shape[-2:])
        edge = None if ds == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        for seed in SEEDS:
            z = raw_anchor(anchors["raw_detail", seed], views)
            grid = z.shape[-2:]
            original = resize(z, truth.shape)
            canonical = F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False)
            data = method.cell_data(z, guides)
            cells = truth_cells(data, truth, grid) if edge is not None else None
            template = base.guide_stencil(guides[:0], z.argmax(1)[0])
            for arm in NAMES:
                row = dict(ds=ds, id=key, group=group, method=arm, seed=seed)
                if arm in MODES or arm == "trained_mean":
                    mode = "mean" if arm == "trained_mean" else arm
                    w = method.weights(mode, data, gates.get((mode, seed)))
                    stencil = weighted.prepare(guides, z, "dual_core", template, w)
                    transport = older["dual_free", seed] if arm == "trained_mean" else decoders["dual_core", seed]
                    delta = base.prediction("dual_free", transport, z, stencil)
                    if cells is not None and arm in MODES:
                        row.update(donor=measure(data, w, cells), coverage=cells[-1])
                elif arm == "sapa_context":
                    delta = sapa.run(arm, published[arm, seed], z, guides) - canonical
                elif arm == "dysample":
                    delta = older[arm, seed](z) - canonical
                else:
                    delta = None
                native = original if delta is None else original + resize(delta, truth.shape)
                prediction = native.argmax(1)[0]
                row["counts"] = confusion_matrix(prediction, truth, 19).tolist()
                if edge is not None:
                    row["edge"] = confusion_matrix(prediction, edge, 19).tolist()
                rows.append(row)
        seen[ds] += 1
        if sum(seen.values()) % 25 == 0:
            print(part, dict(seen), flush=True)
    if seen != expected or len(rows) != sum(seen.values()) * len(NAMES) * len(SEEDS):
        raise ValueError(f"Missing semantic prediction {seen} {expected}")
    path = ROOT / f"{part}-individual.json.gz"
    compress_scores(path, rows)
    summarize(part, rows, classes, source)
    print("completed", part, dict(seen), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parts", nargs="+", choices=("new", "external", "virtual", "rain", "snow"))
    for part in cli.parse_args().parts:
        evaluate(part)

"""Use restoration features to learn which confident neighboring pixels can guide fog segmentation."""
from boundary_refinement.data.source_integrity import matches_registered
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from collections import Counter, defaultdict
import gzip
import json

import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.anchored_cue import GuidedResidual, anchor_fit, anchored_records
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import DRIVING, ZURICH, resize_image, verified_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import CROP, DATA, INDEX, SEEDS, SIZE, Frame, aggregate, all_cases, cases, patch, paths
from boundary_refinement.evaluation.semantic.restoration_cue import ROOT as ANCHOR
from boundary_refinement.training.road_heads import ROOT as RGB
from boundary_refinement.evaluation.semantic.spatial_restoration import RestorationFeatures
from boundary_refinement.data.staging.stage_muses_affinity import INDEX as HELD_INDEX, MANIFEST as HELD
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/restoration-affinity-v75"
STEPS = 400
OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-3, 0), (3, 0), (0, -3), (0, 3))
ARMS = {
    "raw_residual": "raw_refined", "rgb_residual": "rgb_fusion", "head_residual": "correction_heads",
    "raw_graph": "raw_refined", "rgb_graph": "rgb_fusion", "color_graph": "color_match",
    "head_graph": "correction_heads", "shift_graph": "shift_heads", "head_graph_ce": "correction_heads",
}
PRIMARY = "head_graph"
FIXED = ("author_raw", "raw", "uniform_graph")
BAND_RADIUS = 6


def neighbors(x, offsets=OFFSETS, fill=0):
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    h, w = x.shape[-2:]
    padded = F.pad(x, (radius,) * 4, value=fill)
    return torch.stack([padded[..., radius + dy:radius + dy + h, radius + dx:radius + dx + w]
                        for dy, dx in offsets], dim=1)


class NeighborEvidence:
    def __init__(self, logits):
        # Work with centered class evidence so arbitrary per-pixel logit shifts do not bias the links.
        base = logits - logits.mean(dim=1, keepdim=True)
        leading = (base / 1.7).softmax(dim=1).topk(2, dim=1).values
        margin = leading[:, :1] - leading[:, 1:2]
        donor = neighbors(margin)
        self.weight = 4. * (donor - margin[:, None]).clamp_min(0.)
        self.value = neighbors(base) - base[:, None]

    def update(self, affinity):
        weight = self.weight * affinity
        return (weight * self.value).sum(1) / (1. + weight.sum(1))


def affinity_logits(descriptor):
    difference = neighbors(descriptor) - descriptor[:, None]
    return 2.5 - 6. * difference.square().mean(dim=2, keepdim=True)


def edge_loss(link, labels):
    other = neighbors(labels[:, None], fill=255)
    current = labels[:, None, None]
    valid = (other != 255) & (current != 255)
    positive = valid & (other == current)
    negative = valid & ~positive
    # Normalize separately so common within-class pairs do not overwhelm class boundaries.
    return .5 * ((F.softplus(-link) * positive).sum() / positive.sum().clamp_min(1) +
                 (F.softplus(link) * negative).sum() / negative.sum().clamp_min(1))


def displacement(cues):
    context, feature = cues["correction_heads"].chunk(2, dim=1)
    dims = tuple(n // 2 for n in feature.shape[-2:])
    return torch.cat((context, feature.roll(dims, dims=(-2, -1))), dim=1)


def segmentation_loss(logits, update, label, weight):
    final = logits + update
    ce = F.cross_entropy(final, label, weight=weight, ignore_index=255)
    correct = (logits.argmax(1) == label)
    old_nll = F.cross_entropy(logits, label, ignore_index=255, reduction="none")
    new_nll = F.cross_entropy(final, label, ignore_index=255, reduction="none")
    harm = ((new_nll - old_nll).clamp_min(0.) * correct).sum() / correct.sum().clamp_min(1)
    return ce + .65 * harm + .010 * update.square().mean()


def protocol():
    parents = [DATA / "fit.json", DATA / "calibrate.json", DATA / "held.json", INDEX, ANCHOR / "completed.json",
               ANCHOR / "calibration.json", RGB / "rgb-completed.json", SOURCE_ROOT / "boundary_refinement/evaluation/semantic/spatial_restoration.py",
               SOURCE_ROOT / "boundary_refinement/models/restoration/anchored_cue.py"]
    return dict(source_SHA256=digest(__file__), parents={str(p): digest(p) for p in parents},
                methods=ARMS, seeds=list(SEEDS), steps=STEPS, crop=CROP, offsets=[list(offset) for offset in OFFSETS],
                parameters=26208, boundary_radius_native=BAND_RADIUS, strength=1.,
                plan="Freeze all RGB weights, LiDSeg, and the V71 raw-only adaptation, which never used the five "
                     "official validation recordings. Use 96 V70 fit frames only, with equal 400-step runs and "
                     "26,208 trainable parameters. Train ordinary per-pixel residuals or embedding affinities under "
                     "the same segmentation loss. The graph offers logits only from better-supported nearby raw "
                     "pixels. Test raw, RGB, color-only, aligned and spatially shifted correction cues. Include a graph "
                     "without same-class edge supervision and a fixed uniform-affinity graph.",
                primary="Unit update strength everywhere. Use all newly staged official fog train frames of the "
                        "five withheld recordings after prior-image deduplication. Assess legacy 19-class pooled mIoU "
                        "and a fixed six-native-pixel boundary band as a mechanism diagnostic. Require the aligned graph "
                        "to exceed the same-capacity raw, RGB, own-feature residual and shifted comparisons. Old MUSES "
                        "validation, Zurich and Driving Fine are named repeated checks. BDD is excluded due to two clips.")


@torch.no_grad()
def feature_pool(engine, smoke):
    groups = defaultdict(list)
    histogram = torch.zeros(19, dtype=torch.float64)
    for n, (_, group, _, image, truth, _) in enumerate(paths("fit"), 1):
        _, small, cues, _, _ = engine(image)
        cues["shift_heads"] = displacement(cues)
        label = F.interpolate(truth[None, None].float(), small.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        valid = (label != 255)
        if not bool(valid.any()):
            raise ValueError("A recorded training frame has no usable labels")
        histogram += torch.bincount(label[valid].long(), minlength=19)
        names = {"raw", *ARMS.values()}
        groups[group].append((small.half().cpu(), {name: cues[name].half().cpu() for name in names}, label))
        if n % 24 == 0:
            print(dict(encoded_fit_frames=n), flush=True)
        if smoke and n == 4:
            break
    if sum(len(values) for values in groups.values()) != (4 if smoke else SIZE["fit"]):
        raise ValueError("The recorded fit population changed")
    class_weights = ((histogram.sum() / 19. + 1.) / (histogram + 1.)).pow(.20).clamp(.65, 2.).cuda().float()
    return tuple(groups.values()), class_weights


def graph_mode(method):
    return method.endswith("_graph") or method == "head_graph_ce"


def module_path(mode, seed):
    return ROOT / f"{mode}-{seed}.pth"


def fit_one(mode, seed, records, weight, smoke):
    path = module_path(mode, seed)
    log, lock = path.with_suffix(".json"), ROOT / "protocol.json"
    if not smoke and log.is_file():
        record = read(log)
        if record["weights_SHA256"] != digest(path) or record["protocol_SHA256"] != digest(lock):
            raise ValueError(f"Stored fit differs from the protocol: {mode}/{seed}")
        return record
    set_seed(seed)
    module = GuidedResidual().cuda()
    # An exactly zero embedding would have zero affinity gradients at initialization.
    if graph_mode(mode):
        nn.init.normal_(module.output.weight, std=.014)
    optimiser = torch.optim.AdamW(module.parameters(), lr=.0005, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, STEPS, eta_min=.00006)
    rng = torch.Generator().manual_seed(seed + 75003)
    iterations = 3 if smoke else STEPS
    for _ in range(iterations):
        optimiser.zero_grad(set_to_none=True)
        cropped = [patch(Frame(logits, cues[ARMS[mode]], cues[ARMS[mode]][:, :0], labels), rng)
                   for logits, cues, labels in cases(records, rng)]
        signal = torch.cat([r.features for r in cropped])
        label = torch.stack([r.labels for r in cropped])
        logits = torch.cat([r.logits for r in cropped])
        result = module(signal)
        if graph_mode(mode):
            link = affinity_logits(result)
            update = NeighborEvidence(logits).update(link.sigmoid())
            aux = .30 * edge_loss(link, label) if mode != "head_graph_ce" else 0.
        else:
            update, aux = result, 0.
        loss = segmentation_loss(logits, update, label, weight) + aux
        loss.backward()
        nn.utils.clip_grad_norm_(module.parameters(), 2.5, error_if_nonfinite=True)
        optimiser.step()
        schedule.step()
    record = dict(method=mode, seed=seed, steps=iterations, loss=float(loss.detach()),
                  parameters=sum(p.numel() for p in module.parameters()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in module.state_dict().items()})
        record.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
        atomic_write(log, record)
    print(record, flush=True)
    return record


def fit(smoke):
    configure_cuda()
    if not smoke:
        lock, plan = ROOT / "protocol.json", protocol()
        if lock.is_file() and read(lock) != plan:
            raise ValueError("Cannot reuse a fit from a different protocol")
        atomic_write(lock, plan)
    engine = RestorationFeatures()
    frozen, strength = anchor_fit()
    values, weight = feature_pool(engine, smoke)
    recorded = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        points = anchored_records(values, frozen["raw", seed], strength)
        recorded.extend(fit_one(mode, seed, points, weight, smoke) for mode in ARMS)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=recorded, steps=sum(r["steps"] for r in recorded),
                     protocol_SHA256=digest(lock), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                     finished_utc=now()))


def fitted():
    card, lock = read(ROOT / "completed.json"), ROOT / "protocol.json"
    if card["protocol_SHA256"] != digest(lock) or not matches_registered(read(lock), protocol()):
        raise ValueError("The fitted protocol or its inputs changed")
    bank = {}
    for info in card["runs"]:
        key = info["method"], info["seed"]
        path = module_path(*key)
        if digest(path) != info["weights_SHA256"] or info["parameters"] != 26208 or info["steps"] != STEPS:
            raise ValueError(f"Mismatched affinity checkpoint: {key}")
        model = GuidedResidual().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[key] = model
    if set(bank) != {(mode, seed) for mode in ARMS for seed in SEEDS}:
        raise ValueError("Some model arms are missing")
    return bank


def registered_cases(stage):
    if stage == "external":
        yield from all_cases(True)
        return
    if stage in ("development", "calibrate"):
        scope = "held" if stage == "development" else "calibrate"
        yield from (("muses_" + stage, *v) for v in paths(scope))
        return
    card = read(HELD)
    if card["registration_SHA256"] != digest(HELD_INDEX) or any(
            digest(DATA / f"{s}.json") != h for s, h in card["v70_manifests"].items()):
        raise ValueError("The first-use MUSES masks no longer match the registered recordings")
    used_groups = {row["group"] for scope in ("fit", "calibrate") for row in read(DATA / f"{scope}.json")["pairs"]}
    if used_groups.intersection(card["groups"]):
        raise ValueError("A new test recording enters the fitted adapter or anchor")
    for case in card["pairs"]:
        a, b = case["frame_camera"], case["gt_semantic"]
        rgb = verified_image(a["file"], a["SHA256"], False)
        mask = verified_image(b["file"], b["SHA256"], True)
        image, _ = resize_image(rgb)
        yield "muses_first_use", case["id"], case["group"], case["daytime"], image, torch.from_numpy(mask.copy()).cuda().long(), rgb


def boundary_band(truth):
    current = truth[None, None]
    other = neighbors(current, offsets=OFFSETS[:4], fill=255)
    valid_edge = ((current[:, None] != 255) & (other != 255) & (other != current[:, None])).any(1)
    return F.max_pool2d(valid_edge.float(), 2 * BAND_RADIUS + 1, 1, BAND_RADIUS)[0, 0].bool()


def predictions(dense, small, cues, target_shape, bank, reference, strength):
    orig = resize(dense, target_shape)
    result = {("author_raw", 0): orig.argmax(1)[0]}
    cues["shift_heads"] = displacement(cues)
    for seed in SEEDS:
        correction = strength * reference["raw", seed](cues["raw"])
        anchor = orig + resize(correction, target_shape)
        spatial = NeighborEvidence(small + correction)
        result["raw", seed] = anchor.argmax(1)[0]
        result["uniform_graph", seed] = (anchor + resize(spatial.update(.75), target_shape)).argmax(1)[0]
        for mode, signal in ARMS.items():
            update = bank[mode, seed](cues[signal])
            if graph_mode(mode):
                update = spatial.update(affinity_logits(update).sigmoid())
            result[mode, seed] = (anchor + resize(update, target_shape)).argmax(1)[0]
    return result


def compress_scores(path, rows):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), compresslevel=5, mtime=0))
    temporary.replace(path)


@torch.inference_mode()
def score(stage):
    configure_cuda()
    engine, models = RestorationFeatures(), fitted()
    reference, strength = anchor_fit()
    rows, shown, counts = [], {}, Counter()
    for dataset, key, group, lighting, image, truth, rgb in registered_cases(stage):
        dense, small, cues, restored, _ = engine(image)
        inferred = predictions(dense, small, cues, truth.shape, models, reference, strength)
        band = boundary_band(truth)
        regions = (truth, truth.masked_fill(~band, 255), truth.masked_fill(band, 255))
        for (method, seed), value in inferred.items():
            complete, edge, interior = [confusion_matrix(value, target, 19).tolist() for target in regions]
            rows.append(dict(dataset=dataset, id=key, group=group, lighting=lighting, method=method, seed=seed,
                             counts=complete, boundary=edge, interior=interior))
        maximum = 5 if stage == "held" else 2
        group_key = dataset, group
        if group_key not in shown and sum(k[0] == dataset for k in shown) < maximum:
            viewed = [blend(rgb, inferred[m, SEEDS[0]].byte().cpu().numpy())
                      for m in ("raw_residual", "head_residual", "raw_graph", PRIMARY, "shift_graph")]
            shown[group_key] = (f"{key} / {lighting or dataset}", [rgb, to_rgb(restored), *viewed, paint(truth.byte().cpu().numpy())],
                                ("Fog", "Restored RGB", "Raw residual", "Feature residual", "Raw graph",
                                 "Restoration graph", "Shifted graph", "Reference"))
        counts[dataset] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(stage=stage, scored=sum(counts.values())), flush=True)
    expected = dict(muses_first_use=len(read(HELD)["pairs"])) if stage == "held" else {
        "external": dict(foggy_zurich_testv2=40, foggy_driving_fine=33),
        "development": dict(muses_development=SIZE["held"]), "calibrate": dict(muses_calibrate=SIZE["calibrate"]),
    }[stage]
    if counts != expected:
        raise ValueError(f"Misaligned evaluation population: {counts}, expected {expected}")
    controls = [(PRIMARY, method) for method in (*FIXED, *ARMS) if method != PRIMARY]
    intervals = {r: paired_iou_intervals(rows, controls, "method", r) for r in ("counts", "boundary", "interior")}
    metrics = {r: aggregate([{**row, "counts": row[r]} for row in rows]) for r in intervals}
    individual = ROOT / f"{stage}-individual.json.gz"
    preview = ROOT / f"{stage}-predictions.png"
    compress_scores(individual, rows)
    draw(preview, list(shown.values()))
    manifests = {"held": [HELD], "development": [DATA / "held.json"], "calibrate": [DATA / "calibrate.json"],
                 "external": [ZURICH, DRIVING]}
    atomic_write(ROOT / f"{stage}-test.json", dict(counts=dict(counts), groups=sorted({r["group"] for r in rows}),
                 metrics=metrics, intervals=intervals, native_boundary_radius=BAND_RADIUS,
                 data_SHA256={str(p): digest(p) for p in manifests[stage]}, scores_SHA256=digest(individual),
                 weights_SHA256=digest(ROOT / "completed.json"), picture_SHA256=digest(preview), finished_utc=now()))
    print({key: round(v["miou"], 3) for key, v in metrics["counts"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "held", "external", "development", "calibrate"))
    stage = parser.parse_args().stage
    if stage in ("smoke", "fit"):
        fit(stage == "smoke")
    else:
        score(stage)

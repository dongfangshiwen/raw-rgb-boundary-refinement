"""Refit internal-feature controls on a second, recording-disjoint real-fog split."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from collections import Counter, defaultdict
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.anchored_cue import GuidedResidual, anchored_records, objective
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import resize_image, verified_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import CROP, DATA as PRIOR, SEEDS, STRENGTHS, Frame, aggregate, cases, patch
from boundary_refinement.evaluation.semantic.restoration_cue import Adapter
from boundary_refinement.training.road_heads import ROOT as RGB, geometry, render_images
from boundary_refinement.evaluation.semantic.spatial_restoration import RestorationFeatures, bdd_cases, common_signed, first_activations, normalize_spatial
from boundary_refinement.data.staging.stage_bdd_fog import MANIFEST as BDD
from boundary_refinement.data.staging.stage_muses_real_fog import thumbnail
from boundary_refinement.data.staging.stage_muses_recording import INDEX, ROOT as DATA
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/recording-restoration-v74"
MODES = ("raw_refined", "rgb_fusion", "color_match", "decoder_output", "shared_update", "correction_heads",
         "correction_average", "correction_shared", "correction_shifted")
PRIMARY = "correction_shared"
STEPS = 400


class CorrectionViews(RestorationFeatures):
    @torch.no_grad()
    def __call__(self, image):
        dense, small, masks, rgb, matched = super().__call__(image)
        bank = self.base.bank
        layers = dict(local=bank["utility", SEEDS[0]][-2], tone=bank["utility_tone", SEEDS[0]][-2])
        with first_activations(layers) as counterpart:
            render_images(geometry(matched, self.base.rgb), bank, SEEDS[0])
        b = normalize_spatial(torch.cat([resize(counterpart[n][1], small.shape[-2:]) for n in layers], dim=1))
        context, reduced = masks["correction_heads"].chunk(2, dim=1)
        # The isometry in V73 is frozen. Its transpose recovers the original 48-channel coordinates.
        a = F.conv2d(reduced, self.shape_map.T[..., None, None])
        shared = common_signed(a, b)
        cues = dict(correction_average=(a + b) * .5, correction_shared=shared,
                    correction_shifted=shared.roll(tuple(n // 2 for n in shared.shape[-2:]), dims=(-2, -1)))
        masks.update({m: torch.cat((context, self.unfold(x)), dim=1) for m, x in cues.items()})
        return dense, small, {m: masks[m] for m in ("raw", *MODES)}, rgb, matched


def manifests(scope):
    path = DATA / f"{scope}.json"
    signed = read(path)
    if signed["registration_SHA256"] != digest(INDEX):
        raise ValueError(f"The new recording scope changed: {scope}")
    rows = signed["pairs"]
    if scope == "fit":
        rows = [*rows, *read(PRIOR / "held.json")["pairs"]]
    return rows


def pairs(scope):
    for case in manifests(scope):
        r, y = case["frame_camera"], case["gt_semantic"]
        image = verified_image(r["file"], r["SHA256"], False)
        truth = verified_image(y["file"], y["SHA256"], True)
        x, _ = resize_image(image)
        yield case["id"], case["group"], x, torch.from_numpy(truth.copy()).cuda().long(), image


def register():
    return dict(code_SHA256=digest(__file__), new_split_SHA256=digest(INDEX), train_SHA256=digest(DATA / "fit.json"),
                expanded_train_SHA256=digest(PRIOR / "held.json"), calibration_SHA256=digest(DATA / "calibrate.json"),
                rgb_backbone_SHA256=digest(RGB / "rgb-completed.json"),
                features_SHA256=digest(SOURCE_ROOT / "boundary_refinement/evaluation/semantic/spatial_restoration.py"), modes=list(MODES),
                fit_images=len(manifests("fit")), seeds=list(SEEDS), steps=STEPS, strengths=list(STRENGTHS), crop=CROP,
                design="Train the raw LiDSeg adapter from random initialization using 46 fit images and the 58 older "
                       "MUSES validation frames. Freeze it and use exactly the same matched correction layer and loss "
                       "as V73. Fit all arms from scratch. Add a control for illumination changes inside the two "
                       "trained RGB correction heads. Measure ordinary, averaged, same-sign and displaced features "
                       "from those heads. Prior MUSES validation is fit-only for these models. Calibrate on nine frames, "
                       "test on 44 previously unused frames and keep all their recordings out of both other stages. "
                       "Earlier project versions saw other photographs of these routes. The two BDD frames are a repeat probe.")


@torch.no_grad()
def cache(engine, smoke):
    pool, histogram = defaultdict(list), torch.zeros(19, dtype=torch.float64)
    for i, (_, group, x, truth, _) in enumerate(pairs("fit"), start=1):
        _, small, cues, _, _ = engine(x)
        mask = F.interpolate(truth[None, None].float(), small.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        count = mask != 255
        if not bool(count.any()):
            raise ValueError("A training frame has no valid semantic pixels")
        histogram += torch.bincount(mask[count].long(), minlength=19)
        pool[group].append((small.half().cpu(), {m: z.half().cpu() for m, z in cues.items()}, mask))
        if i % 20 == 0:
            print(dict(encoded_real_frames=i), flush=True)
        if smoke and i == 4:
            break
    if sum(map(len, pool.values())) != (4 if smoke else len(manifests("fit"))):
        raise ValueError("Not all registered fit frames were encoded")
    weights = ((histogram.sum() / 19. + 1.) / (histogram + 1.)).pow(.20).clamp(.65, 2.).cuda().float()
    return tuple(pool.values()), weights


def train(mode, seed, groups, scales, smoke):
    path = ROOT / f"{mode}-{seed}.pth"
    log, plan = path.with_suffix(".json"), ROOT / "protocol.json"
    set_seed(seed)
    module = (Adapter if mode == "raw" else GuidedResidual)().cuda()
    if not smoke and log.is_file():
        data = read(log)
        if digest(path) != data["weights_SHA256"] or digest(plan) != data["protocol_SHA256"]:
            raise ValueError(f"Recorded weight changed: {path.name}")
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return module.eval().requires_grad_(False), data
    rand = torch.Generator().manual_seed(seed + 74003)
    optim = torch.optim.AdamW(module.parameters(), lr=.0005, weight_decay=.01)
    adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optim, STEPS, eta_min=.00006)
    for i in range(3 if smoke else STEPS):
        optim.zero_grad(set_to_none=True)
        mini = []
        for logits, z, labels in cases(groups, rand):
            h = z[mode]
            packed = patch(Frame(logits, h, h[:, :0], labels), rand)
            if mode == "raw":
                offset = module(packed.features)
                loss = F.cross_entropy(packed.logits + offset, packed.labels[None], weight=scales, ignore_index=255)
                mini.append(loss + .010 * offset.square().mean())
            else:
                mini.append(objective(module, packed, scales, "ordinary"))
        error = torch.stack(mini).mean()
        error.backward()
        nn.utils.clip_grad_norm_(module.parameters(), 2.5, error_if_nonfinite=True)
        optim.step()
        adjust.step()
    data = dict(method=mode, seed=seed, steps=i + 1, loss=float(error.detach()), parameters=sum(t.numel() for t in module.parameters()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in module.state_dict().items()})
        data.update(weights_SHA256=digest(path), protocol_SHA256=digest(plan))
        atomic_write(log, data)
    print(data, flush=True)
    return module.eval().requires_grad_(False), data


def fit(smoke):
    configure_cuda()
    note, path = register(), ROOT / "protocol.json"
    if not smoke:
        if path.exists() and read(path) != note:
            raise ValueError("The recording study training evidence changed")
        atomic_write(path, note)
    bank, weights = cache(CorrectionViews(), smoke)
    results = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        raw, row = train("raw", seed, bank, weights, smoke)
        results.append(row)
        common = anchored_records(bank, raw, 1.)
        for mode in MODES:
            _, row = train(mode, seed, common, weights, smoke)
            results.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=results, steps=sum(r["steps"] for r in results),
                     protocol_SHA256=digest(path), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def model_bank():
    card = read(ROOT / "completed.json")
    lock = ROOT / "protocol.json"
    if digest(lock) != card["protocol_SHA256"] or read(lock) != register():
        raise ValueError("The held-recording experiment changed")
    bank = {}
    for r in card["runs"]:
        name, seed = r["method"], r["seed"]
        path = ROOT / f"{name}-{seed}.pth"
        if digest(path) != r["weights_SHA256"]:
            raise ValueError(f"Unexpected held-recording model hash: {path.name}")
        net = (Adapter if name == "raw" else GuidedResidual)().cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[name, seed] = net
    if set(bank) != {(mode, seed) for mode in ("raw", *MODES) for seed in SEEDS}:
        raise ValueError("Missing independently fitted controls")
    return bank


def check_data():
    card = read(DATA / "audit.json")
    if not card["recordings_disjoint"] or any(digest(DATA / f"{s}.json") != h for s, h in card["manifests"].items()):
        raise ValueError("The real-fog split audit changed")
    groups = {kind: {row["group"] for row in manifests(kind)} for kind in ("fit", "calibrate", "held")}
    if sum(map(len, groups.values())) != len(set().union(*groups.values())):
        raise ValueError("A recording crosses the expanded split")
    old_val = read(PRIOR / "held.json")["pairs"]
    new_hold = manifests("held")
    ref = [(row, thumbnail(row)) for row in old_val]
    for a in new_hold:
        x, p = thumbnail(a)
        for b, (y, q) in ref:
            if a["frame_camera"]["SHA256"] == b["frame_camera"]["SHA256"] or ((p != q).mean() < .035 and np.abs(x - y).mean() < .035):
                raise ValueError(f"Possible repeated RGB from expanded training: {a['id']} {b['id']}")
    return groups


def scene(engine, bank, x, shape):
    logits, _, hints, restored, color = engine(x)
    orig = resize(logits, shape)
    anchor = {s: orig + resize(bank["raw", s](hints["raw"]), shape) for s in SEEDS}
    return orig, anchor, hints, restored, color


@torch.inference_mode()
def calibrate(engine, bank):
    sums = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64))
    for _, _, x, gt, _ in pairs("calibrate"):
        _, anchor, f, _, _ = scene(engine, bank, x, gt.shape)
        for mode in MODES:
            for seed in SEEDS:
                z = resize(bank[mode, seed](f[mode]), gt.shape)
                for s in STRENGTHS:
                    p = (anchor[seed] + s * z).argmax(1)[0]
                    sums[mode, seed, s] += confusion_matrix(p, gt, 19)
    scores = {m: {str(w): 100. * mean(segmentation_metrics(sums[m, s, w])["miou"] for s in SEEDS) for w in STRENGTHS} for m in MODES}
    selected = {k: float(max(v, key=lambda w: (v[w], -float(w)))) for k, v in scores.items()}
    card = dict(chosen=selected, scores=scores, calibration_SHA256=digest(DATA / "calibrate.json"),
                weight_manifest_SHA256=digest(ROOT / "completed.json"), finished_utc=now())
    atomic_write(ROOT / "calibration.json", card)
    print(card, flush=True)
    return card


@torch.inference_mode()
def score(external):
    configure_cuda()
    groups = check_data()
    engine, bank = CorrectionViews(), model_bank()
    path = ROOT / "calibration.json"
    cal = read(path) if path.exists() else calibrate(engine, bank)
    if cal["calibration_SHA256"] != digest(DATA / "calibrate.json") or cal["weight_manifest_SHA256"] != digest(ROOT / "completed.json"):
        raise ValueError("Experiment calibration changed")
    queue = (("bdd_fog", key, group, x, y, pic) for _, key, group, _, x, y, pic in bdd_cases()) if external else (
        ("muses_new_frames", key, group, x, y, pic) for key, group, x, y, pic in pairs("held"))
    rows, panels, seen, independently_scored = [], [], Counter(), set()
    for split, key, group, x, gt, image in queue:
        orig, raw, f, restored, color = scene(engine, bank, x, gt.shape)
        variants = {("author_raw", 0): orig.argmax(1)[0]}
        variants.update({("raw", s): d.argmax(1)[0] for s, d in raw.items()})
        for m in MODES:
            for s in SEEDS:
                delta = resize(bank[m, s](f[m]), gt.shape)
                variants[m, s] = (raw[s] + cal["chosen"][m] * delta).argmax(1)[0]
                if m == PRIMARY:
                    variants[PRIMARY + "_unshrunk", s] = (raw[s] + delta).argmax(1)[0]
        for (m, seed), p in variants.items():
            rows.append(dict(dataset=split, id=key, group=group, method=m, seed=seed, counts=confusion_matrix(p, gt, 19).tolist()))
        if group not in independently_scored:
            outlook = [blend(image, variants[m, SEEDS[0]].byte().cpu().numpy()) for m in ("raw_refined", "rgb_fusion", PRIMARY)]
            panels.append((f"{split} / {group} / {key}", [image, to_rgb(restored), to_rgb(color), *outlook, paint(gt.byte().cpu().numpy())],
                           ("Fog", "Restored RGB", "Color matched", "Raw refinement", "RGB fusion", "Correction features", "Reference")))
        seen[split] += 1
        independently_scored.add(group)
    count = len(read(BDD)["pairs"]) if external else len(manifests("held"))
    if sum(seen.values()) != count or (not external and independently_scored != groups["held"]):
        raise ValueError("The registered held recording set changed")
    result = aggregate(rows)
    controls = [(PRIMARY, m) for m in ("raw", "author_raw", *(m for m in MODES if m != PRIMARY))]
    contrasts = paired_iou_intervals(rows, controls, "method", "counts")
    tag = "bdd" if external else "muses"
    table, preview = ROOT / f"{tag}-individual.json", ROOT / f"{tag}-predictions.png"
    atomic_write(table, rows)
    draw(preview, panels)
    atomic_write(ROOT / f"{tag}-test.json", dict(counts=dict(seen), groups={k: sorted(v) for k, v in groups.items()}, metrics=result,
                 intervals=contrasts, scores=len(rows), scores_SHA256=digest(table), picture_SHA256=digest(preview),
                 data_SHA256=digest(BDD if external else DATA / "audit.json"), calibration_SHA256=digest(path), finished_utc=now()))
    print({k: round(v["miou"], 4) for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "test", "bdd"))
    step = parser.parse_args().stage
    if step in ("smoke", "fit"):
        fit(step == "smoke")
    else:
        score(external=step == "bdd")

"""Compare spatial restoration decoder features with output-RGB and color controls."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.anchored_cue import GuidedResidual, anchor_fit, anchored_records, objective
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_author_mct import OFFICIAL
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import resize_image, verified_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import (
    CROP, DATA, INDEX, SEEDS, SIZE, STRENGTHS, Frame, aggregate, all_cases, cases, patch, paths
)
from boundary_refinement.evaluation.semantic.restoration_cue import ROOT as RAW, SceneViews, evidence
from boundary_refinement.data.staging.stage_bdd_fog import INDEX as BDD_INDEX, MANIFEST as BDD_MANIFEST
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/spatial-restoration-v73"
ARMS = ("raw_refined", "rgb_fusion", "color_match", "decoder_output", "decoder_update", "averaged_update",
        "shared_update", "shifted_update", "correction_heads")
PRIMARY = "shared_update"
STEPS = 400


@contextmanager
def first_activations(layers):
    saved = {}

    def hook(name):
        def observe(_layer, args, out):
            if name not in saved:
                saved[name] = (args[0].detach(), out.detach())
        return observe

    handles = [layer.register_forward_hook(hook(name)) for name, layer in layers.items()]
    try:
        yield saved
    finally:
        for handle in handles:
            handle.remove()


def normalize_spatial(feature):
    space = feature.float() - feature.float().mean(dim=(-2, -1), keepdim=True)
    spread = space.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
    return (space / spread).clamp(-4., 4.)


def common_signed(raw, matched):
    return torch.where(raw * matched > 0., raw.sign() * torch.minimum(raw.abs(), matched.abs()), 0.)


class RestorationFeatures(SceneViews):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(73007)
        # A fixed isometry places 48 channels in the same 51-channel input width as the LiDSeg cues.
        self.shape_map = torch.linalg.qr(torch.randn(51, 48, generator=generator))[0].cuda()

    def unfold(self, x):
        if x.shape[1] != 48:
            raise ValueError(f"Unexpected restoration feature width: {x.shape}")
        return F.conv2d(x, self.shape_map[..., None, None])

    @torch.no_grad()
    def __call__(self, image):
        model = self.base.rgb
        weights = self.base.bank
        objects = dict(decoder=model.basenet.layer4, local=weights["utility", SEEDS[0]][-2],
                       tone=weights["utility_tone", SEEDS[0]][-2])
        # The first decoder call is the unflipped primary MCT pass of the actual dual-output graph.
        with first_activations(objects) as features:
            dense, small, semantic, restored, matched = super().__call__(image)
        with first_activations({"decoder": model.basenet.layer4}) as null:
            square = F.interpolate(2. * matched - 1., (model.ts, model.ts), mode="area")
            model.basenet(square)
        orig_before, orig_after = features["decoder"]
        color_before, color_after = null["decoder"]
        size = small.shape[-2:]
        a = normalize_spatial(resize(orig_after - orig_before, size))
        b = normalize_spatial(resize(color_after - color_before, size))
        shared = common_signed(a, b)
        learned = [resize(features[name][1], size) for name in ("local", "tone")]
        internal = dict(decoder_output=normalize_spatial(resize(orig_after, size)), decoder_update=a,
                        averaged_update=(a + b) * .5, shared_update=shared,
                        shifted_update=shared.roll(tuple(n // 2 for n in size), dims=(-2, -1)),
                        correction_heads=normalize_spatial(torch.cat(learned, dim=1)))
        fresh = {name: torch.cat((semantic[:1], self.unfold(z)), dim=1) for name, z in internal.items()}
        fresh.update(raw=evidence(semantic, "raw"), raw_refined=torch.cat((semantic[:1], semantic[:1]), dim=1),
                     rgb_fusion=evidence(semantic, "ordinary"), color_match=evidence(semantic, "color_only"))
        return dense, small, fresh, restored, matched


def registration():
    return dict(implementation_SHA256=digest(__file__), train_manifest_SHA256=digest(DATA / "fit.json"),
                recording_review_SHA256=digest(DATA / "audit.json"), baseline_SHA256=digest(RAW / "completed.json"),
                baseline_calibration_SHA256=digest(RAW / "calibration.json"),
                author_MCT_SHA256=digest(OFFICIAL / "reference_registration.json"),
                data_registration_SHA256=digest(INDEX), modes=list(ARMS), seed=list(SEEDS), steps=STEPS,
                crop=CROP, strengths=list(STRENGTHS),
                design="Reuse the fixed raw LiDSeg adapter per seed. Use the same 26,208-parameter layer and harm-aware "
                       "loss for all trained arms. Tap layer4 before and after its feature update. Normalize spatially and "
                       "keep the sign-consistent minimum across original and globally color-matched input. Compare the "
                       "averaged update, color matching, ordinary restored-RGB guidance, same-capacity raw refinement, "
                       "unrestricted decoder output, decoder update alone, a spatially shifted cue, and feature maps "
                       "directly from the two trained RGB correction heads. Preserve the original dual-output graph.",
                decision="Use MUSES only to train and calibrate. MUSES held and the earlier two real fog datasets are "
                         "development rechecks. New BDD fog clips are external and unused for any amplitude or model "
                         "choice. Require the held effects to exceed the ordinary RGB, color and internal controls.")


@torch.no_grad()
def packets(engine, smoke):
    groups = defaultdict(list)
    classes = torch.zeros(19, dtype=torch.float64)
    for n, (_, group, _, raw, gt, _) in enumerate(paths("fit"), start=1):
        _, tiny, cues, _, _ = engine(raw)
        mask = F.interpolate(gt[None, None].float(), tiny.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        valid = mask != 255
        if not bool(valid.any()):
            raise ValueError("An annotated MUSES frame has no valid fit pixels")
        classes += torch.bincount(mask[valid].long(), minlength=19)
        groups[group].append((tiny.half().cpu(), {name: z.half().cpu() for name, z in cues.items()}, mask))
        if n % 24 == 0:
            print(dict(restoration_features_encoded=n), flush=True)
        if smoke and n == 4:
            break
    if sum(map(len, groups.values())) != (4 if smoke else SIZE["fit"]):
        raise ValueError("The fit signal pool is incomplete")
    return tuple(groups.values()), ((classes.sum() / 19. + 1.) / (classes + 1.)).pow(.20).clamp(.65, 2.).cuda().float()


def fit(smoke):
    configure_cuda()
    lock = ROOT / "protocol.json"
    recipe = registration()
    if not smoke:
        if lock.exists() and read(lock) != recipe:
            raise ValueError("The registered feature experiment changed")
        atomic_write(lock, recipe)
    source, (baseline, scale) = RestorationFeatures(), anchor_fit()
    pool, weights = packets(source, smoke)
    recorded = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        anchored = anchored_records(pool, baseline["raw", seed], scale)
        for mode in ARMS:
            path = ROOT / f"{mode}-{seed}.pth"
            log = path.with_suffix(".json")
            if not smoke and log.exists():
                note = read(log)
                if digest(path) != note["weights_SHA256"] or digest(lock) != note["protocol_SHA256"]:
                    raise ValueError(f"The existing matched train run changed: {path.name}")
                recorded.append(note)
                continue
            set_seed(seed)
            head = GuidedResidual().cuda()
            generator = torch.Generator().manual_seed(seed + 72007)
            opt = torch.optim.AdamW(head.parameters(), lr=.0005, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00006)
            for step in range(3 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                loss = []
                for logits, signal, labels in cases(anchored, generator):
                    frame = signal[mode]
                    picked = patch(Frame(logits, frame, frame[:, :0], labels), generator)
                    loss.append(objective(head, picked, weights, "ordinary"))
                value = torch.stack(loss).mean()
                value.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 2.5, error_if_nonfinite=True)
                opt.step()
                schedule.step()
            note = dict(method=mode, seed=seed, steps=step + 1, loss=float(value.detach()), parameters=sum(p.numel() for p in head.parameters()))
            if not smoke:
                torch_save(path, {k: t.detach().cpu() for k, t in head.state_dict().items()})
                note.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
                atomic_write(log, note)
                recorded.append(note)
            print(note, flush=True)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=recorded, steps=sum(x["steps"] for x in recorded),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def fitted():
    manifest = read(ROOT / "completed.json")
    protocol = ROOT / "protocol.json"
    if digest(protocol) != manifest["protocol_SHA256"] or not matches_registered(read(protocol), registration()):
        raise ValueError("The original training inputs or code changed")
    bank = {}
    for row in manifest["runs"]:
        mode, seed = row["method"], row["seed"]
        path = ROOT / f"{mode}-{seed}.pth"
        if digest(path) != row["weights_SHA256"]:
            raise ValueError(f"Checkpoint hash differs for {path.name}")
        model = GuidedResidual().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[mode, seed] = model
    if set(bank) != {(mode, s) for mode in ARMS for s in SEEDS}:
        raise ValueError("Some of the matched controls were not fitted")
    return bank


def forward_scene(engine, anchor, strength, frame, shape):
    dense, _, cues, rgb, color = engine(frame)
    logits = resize(dense, shape)
    baseline = {s: logits + strength * resize(anchor["raw", s](cues["raw"]), shape) for s in SEEDS}
    return logits, baseline, cues, rgb, color


@torch.inference_mode()
def calibrate(engine, heads, raw, amplitude):
    ms = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64))
    for _, _, _, frame, gt, _ in paths("calibrate"):
        _, logits, cue, _, _ = forward_scene(engine, raw, amplitude, frame, gt.shape)
        for (mode, seed), head in heads.items():
            change = resize(head(cue[mode]), gt.shape)
            for strength in STRENGTHS:
                label = (logits[seed] + strength * change).argmax(1)[0]
                ms[mode, seed, strength] += confusion_matrix(label, gt, 19)
    scores = {m: {str(w): 100. * mean(segmentation_metrics(ms[m, s, w])["miou"] for s in SEEDS) for w in STRENGTHS} for m in ARMS}
    keep = {m: float(max(q, key=lambda k: (q[k], -float(k)))) for m, q in scores.items()}
    result = dict(chosen=keep, scores=scores, calibration_SHA256=digest(DATA / "calibrate.json"),
                  trained_SHA256=digest(ROOT / "completed.json"), finished_utc=now())
    atomic_write(ROOT / "calibration.json", result)
    print(result, flush=True)
    return result


def bdd_cases():
    selected = read(BDD_MANIFEST)
    if digest(BDD_INDEX) != selected["registration_SHA256"]:
        raise ValueError("The external BDD test set no longer matches its registration")
    for row in selected["pairs"]:
        image = verified_image(row["image"], row["image_SHA256"], False)
        mask = verified_image(row["label"], row["label_SHA256"], True)
        raw, _ = resize_image(image)
        yield "bdd_fog", row["id"], row["group"], "", raw, torch.from_numpy(mask.copy()).cuda().long(), image


@torch.inference_mode()
def test(kind):
    configure_cuda()
    source, bank = RestorationFeatures(), fitted()
    raw, strength = anchor_fit()
    check = read(DATA / "audit.json")
    if not check["recordings_disjoint"] or any(digest(DATA / f"{scope}.json") != h for scope, h in check["manifests"].items()):
        raise ValueError("Audited MUSES data changed")
    lock = ROOT / "calibration.json"
    card = read(lock) if lock.exists() else calibrate(source, bank, raw, strength)
    if card["calibration_SHA256"] != digest(DATA / "calibrate.json") or card["trained_SHA256"] != digest(ROOT / "completed.json"):
        raise ValueError("Stale calibration")
    dataset = bdd_cases() if kind == "bdd" else all_cases(kind == "external")
    rows, panels, seen, groups = [], [], Counter(), set()
    for data, key, group, _, x, gt, frame in dataset:
        seedless, fixed, cue, rgb, matched = forward_scene(source, raw, strength, x, gt.shape)
        labels = {("author_raw", 0): seedless.argmax(1)[0]}
        labels.update({("fixed_raw", s): t.argmax(1)[0] for s, t in fixed.items()})
        for (method, seed), head in bank.items():
            delta = resize(head(cue[method]), gt.shape)
            labels[method, seed] = (fixed[seed] + card["chosen"][method] * delta).argmax(1)[0]
            if method == PRIMARY:
                labels[PRIMARY + "_unshrunk", seed] = (fixed[seed] + delta).argmax(1)[0]
        for (method, seed), pred in labels.items():
            rows.append(dict(dataset=data, id=key, group=group, method=method, seed=seed,
                             counts=confusion_matrix(pred, gt, 19).tolist()))
        show = len(panels) < 4 if kind == "bdd" else group not in groups and (kind == "muses" or not seen[data])
        if show:
            tracks = ("fixed_raw", "rgb_fusion", PRIMARY)
            overlay = [blend(frame, labels[m, SEEDS[0]].byte().cpu().numpy()) for m in tracks]
            titles = ("Fog", "Restored RGB", "Color matched", "Fixed raw adapter", "Restored-RGB cue", "Spatial update", "Reference")
            panels.append((f"{data} / {group} / {key}", [frame, to_rgb(rgb), to_rgb(matched), *overlay, paint(gt.byte().cpu().numpy())], titles))
        seen[data] += 1
        groups.add(group)
        if seen[data] % 20 == 0:
            print(dict(dataset=data, scored=seen[data]), flush=True)
    if kind == "bdd":
        expected = dict(bdd_fog=len(read(BDD_MANIFEST)["pairs"]))
    elif kind == "muses":
        expected = dict(muses_full=58)
    else:
        expected = dict(foggy_zurich_testv2=40, foggy_driving_fine=33)
    if seen != expected:
        raise ValueError(f"Test population mismatch: {seen}")
    comparisons = [(PRIMARY, control) for control in ("author_raw", "fixed_raw", *(a for a in ARMS if a != PRIMARY))]
    summary = aggregate(rows)
    intervals = paired_iou_intervals(rows, comparisons, "method", "counts")
    individual, picture = ROOT / f"{kind}-individual.json", ROOT / f"{kind}-predictions.png"
    atomic_write(individual, rows)
    draw(picture, panels)
    atomic_write(ROOT / f"{kind}-test.json", dict(counts=dict(seen), metrics=summary, intervals=intervals, rows=len(rows),
                 table_SHA256=digest(individual), pictures_SHA256=digest(picture), protocol_SHA256=digest(ROOT / "protocol.json"),
                 calibration_SHA256=digest(lock), data_SHA256=digest(BDD_MANIFEST if kind == "bdd" else DATA / "audit.json"), finished_utc=now()))
    print({key: round(value["miou"], 4) for key, value in summary.items()}, flush=True)


def check_math():
    base = torch.randn(1, 48, 9, 12)
    positive = torch.rand(1, 48, 1, 1) + .5
    model = normalize_spatial(base)
    changed = normalize_spatial(base * positive + torch.randn_like(positive))
    torch.testing.assert_close(common_signed(model, changed), model, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(common_signed(model, -model), torch.zeros_like(model))
    torch.testing.assert_close(common_signed(model, changed), common_signed(changed, model))
    print("Spatial cue preserves affine channel invariance, symmetry and rejection of opposite signs.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("math", "smoke", "fit", "muses", "external", "bdd"))
    stage = p.parse_args().stage
    if stage == "math":
        check_math()
    elif stage in ("smoke", "fit"):
        fit(smoke=stage == "smoke")
    else:
        test(stage)

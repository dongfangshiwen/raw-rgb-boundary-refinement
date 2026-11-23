"""Check whether the restoration cue adds value to a fixed, real-fog-adapted segmenter."""
import argparse
from collections import Counter, defaultdict
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import (
    CROP, DATA, INDEX, SEEDS, SIZE, STRENGTHS, Frame, aggregate, all_cases, cases, patch, paths
)
from boundary_refinement.evaluation.semantic.restoration_cue import ROOT as PREVIOUS, SceneViews, evidence, models as adapters
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/anchored-cue-v72"
MODES = ("ordinary", "color_only", "subtracted_ce", "subtracted_protected", "shifted")
PRIMARY = "subtracted_protected"
STEPS = 400


class GuidedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(51, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.Sigmoid())
        self.evidence = nn.Conv2d(51, 24, 3, padding=1, bias=False)
        self.output = nn.Conv2d(24, 19, 3, padding=1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, data):
        unit, signal = data.chunk(2, dim=1)
        update = self.output(self.gate(unit) * F.silu(self.evidence(signal)))
        return 3. * (update / 3.).tanh()


def plan():
    return dict(code_SHA256=digest(__file__), fit_SHA256=digest(DATA / "fit.json"), registration_SHA256=digest(INDEX),
                anchor_SHA256=digest(PREVIOUS / "completed.json"), anchor_scale_SHA256=digest(PREVIOUS / "calibration.json"),
                seeds=list(SEEDS), modes=list(MODES), steps=STEPS, strengths=list(STRENGTHS), crop=CROP,
                hypothesis="Keep the trained raw-only V71 adaptation and its calibration strength fixed. A no-bias cue "
                           "path has exactly zero contribution on zero cue, and raw features can modulate but cannot add "
                           "a separate class update. Compare ordinary restoration, color only, and color-subtracted "
                           "restoration, with spatial shift and removal of the supervised harm penalty. The penalty "
                           "adds cost if the new true-class NLL increases where the common anchor is correct.",
                scope="Mechanism follow-up after V70 and V71. The same MUSES recordings and external sets have been scored. "
                      "Use only training masks for fitting and the separate four training recordings for calibration.")


def cue_for(mode):
    return "subtracted" if mode in ("subtracted_ce", PRIMARY) else mode


def anchor_fit():
    scale = read(PREVIOUS / "calibration.json")
    if scale["training_SHA256"] != digest(PREVIOUS / "completed.json") or scale["calib_SHA256"] != digest(DATA / "calibrate.json"):
        raise ValueError("The raw-only anchor is not the recorded V71 match")
    return adapters(), scale["chosen"]["raw"]


@torch.no_grad()
def raw_records(engine, smoke):
    groups, counts = defaultdict(list), torch.zeros(19, dtype=torch.float64)
    for n, (_, group, _, rgb, gt, _) in enumerate(paths("fit"), start=1):
        _, raw, context, _, _ = engine(rgb)
        mask = F.interpolate(gt[None, None].float(), raw.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        labelled = mask != 255
        if not bool(labelled.any()):
            raise ValueError("A training mask has no labelled pixels")
        counts += torch.bincount(mask[labelled].long(), minlength=19)
        signals = {tag: evidence(context, tag).half().cpu() for tag in ("raw", "ordinary", "color_only", "subtracted", "shifted")}
        groups[group].append((raw.half().cpu(), signals, mask))
        if n % 24 == 0:
            print(dict(anchored_training_images=n), flush=True)
        if smoke and n == 4:
            break
    if sum(map(len, groups.values())) != (4 if smoke else SIZE["fit"]):
        raise ValueError("Anchored training cases are incomplete")
    weights = ((counts.sum() / 19 + 1.) / (counts + 1.)).pow(.20).clamp(.65, 2.).cuda().float()
    return tuple(groups.values()), weights


@torch.no_grad()
def anchored_records(data, raw, scale):
    return [[((base.cuda().float() + scale * raw(signals["raw"].cuda().float())).half().cpu(), signals, labels)
             for base, signals, labels in group] for group in data]


def objective(model, state, weights, arm):
    update = model(state.features)
    calibrated = state.logits + update
    label = state.labels[None]
    main = F.cross_entropy(calibrated, label, weight=weights, ignore_index=255)
    if arm != "subtracted_ce":
        good = state.logits.argmax(1) == label
        original = F.cross_entropy(state.logits, label, ignore_index=255, reduction="none")
        end = F.cross_entropy(calibrated, label, ignore_index=255, reduction="none")
        penalty = ((end - original).clamp_min(0.) * good).sum() / good.sum().clamp_min(1)
        main = main + .65 * penalty
    return main + .010 * update.square().mean()


def fit(smoke):
    configure_cuda()
    lock = ROOT / "protocol.json"
    recipe = plan()
    if not smoke:
        if lock.exists() and read(lock) != recipe:
            raise ValueError("Registered anchored comparison has changed")
        atomic_write(lock, recipe)
    source = SceneViews()
    fixed, scale = anchor_fit()
    raw, weights = raw_records(source, smoke)
    logs = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        groups = anchored_records(raw, fixed["raw", seed], scale)
        for mode in MODES:
            path = ROOT / f"{mode}-{seed}.pth"
            card = path.with_suffix(".json")
            if not smoke and card.exists():
                meta = read(card)
                if digest(path) != meta["weights_SHA256"] or digest(lock) != meta["protocol_SHA256"]:
                    raise ValueError(f"Previously fitted head changed: {path.name}")
                logs.append(meta)
                continue
            set_seed(seed)
            net = GuidedResidual().cuda()
            rng = torch.Generator().manual_seed(seed + 72007)
            opt = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00006)
            for step in range(3 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                losses = []
                for logits, signals, labels in cases(groups, rng):
                    feature = signals[cue_for(mode)]
                    state = patch(Frame(logits, feature, feature[:, :0], labels), rng)
                    losses.append(objective(net, state, weights, mode))
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
                opt.step()
                schedule.step()
            meta = dict(method=mode, seed=seed, steps=step + 1, loss=float(loss.detach()), parameters=sum(p.numel() for p in net.parameters()))
            if not smoke:
                torch_save(path, {k: t.cpu() for k, t in net.state_dict().items()})
                meta.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
                atomic_write(card, meta)
                logs.append(meta)
            print(meta, flush=True)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=logs, steps=sum(r["steps"] for r in logs), protocol_SHA256=digest(lock),
                     gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now()))


def load():
    result = read(ROOT / "completed.json")
    protocol = ROOT / "protocol.json"
    if digest(protocol) != result["protocol_SHA256"] or read(protocol) != plan():
        raise ValueError("The frozen anchored experiment inputs changed")
    heads = {}
    for run in result["runs"]:
        r, seed = run["method"], run["seed"]
        path = ROOT / f"{r}-{seed}.pth"
        if digest(path) != run["weights_SHA256"]:
            raise ValueError(f"Different checkpoint: {path}")
        model = GuidedResidual().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        heads[r, seed] = model
    if len(heads) != len(MODES) * len(SEEDS):
        raise ValueError("Incomplete anchored residual comparison")
    return heads


def proposals(engine, models, weights, scale, x, shape):
    logits, _, features, restored, matched = engine(x)
    log_dense = resize(logits, shape)
    raw = evidence(features, "raw")
    cues = {mode: evidence(features, cue_for(mode)) for mode in MODES}
    start = {s: log_dense + scale * resize(weights["raw", s](raw), shape) for s in SEEDS}
    corrections = {(mode, s): models[mode, s](cues[mode]) for mode in MODES for s in SEEDS}
    return log_dense, start, corrections, restored, matched


@torch.inference_mode()
def calibrate(engine, models, anchor, amplitude):
    sums = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64))
    for _, _, _, x, truth, _ in paths("calibrate"):
        _, base, delta, _, _ = proposals(engine, models, anchor, amplitude, x, truth.shape)
        for (mode, seed), update in delta.items():
            update = resize(update, truth.shape)
            for scale in STRENGTHS:
                result = (base[seed] + scale * update).argmax(1)[0]
                sums[mode, seed, scale] += confusion_matrix(result, truth, 19)
    scores = {m: {str(s): 100. * mean(segmentation_metrics(sums[m, seed, s])["miou"] for seed in SEEDS) for s in STRENGTHS}
              for m in MODES}
    selected = {m: float(max(v, key=lambda k: (v[k], -float(k)))) for m, v in scores.items()}
    packet = dict(strengths=selected, scores=scores, calibration_SHA256=digest(DATA / "calibrate.json"), train_SHA256=digest(ROOT / "completed.json"),
                  finished_utc=now())
    atomic_write(ROOT / "calibration.json", packet)
    print(packet, flush=True)
    return packet


@torch.inference_mode()
def test(external):
    configure_cuda()
    audit = read(DATA / "audit.json")
    if not audit["recordings_disjoint"] or any(digest(DATA / f"{scope}.json") != h for scope, h in audit["manifests"].items()):
        raise ValueError("The earlier recording audit is no longer valid")
    layer, source = load(), SceneViews()
    anchor, amplitude = anchor_fit()
    lock = ROOT / "calibration.json"
    cal = read(lock) if lock.exists() else calibrate(source, layer, anchor, amplitude)
    if cal["calibration_SHA256"] != digest(DATA / "calibrate.json") or cal["train_SHA256"] != digest(ROOT / "completed.json"):
        raise ValueError("Anchored calibration no longer matches the training evidence")
    old = set(read(INDEX)["previous_scoring_ids"])
    seen, panels, groups, table = Counter(), [], set(), []
    for data, key, group, lighting, x, gt, src in all_cases(external):
        ref, start, change, rgb, sham = proposals(source, layer, anchor, amplitude, x, gt.shape)
        masks = {("author_raw", 0): ref.argmax(1)[0]}
        masks.update({("fixed_raw", s): log.argmax(1)[0] for s, log in start.items()})
        for (mode, seed), delta in change.items():
            delta = resize(delta, gt.shape)
            masks[mode, seed] = (start[seed] + cal["strengths"][mode] * delta).argmax(1)[0]
            if mode == PRIMARY:
                masks[PRIMARY + "_unshrunk", seed] = (start[seed] + delta).argmax(1)[0]
        for (name, seed), mask in masks.items():
            row = dict(dataset=data, id=key, group=group, lighting=lighting, method=name, seed=seed,
                       counts=confusion_matrix(mask, gt, 19).tolist())
            table.append(row)
            if not external and key not in old:
                table.append({**row, "dataset": "muses_first_use"})
        if group not in groups and (not external or not seen[data]):
            overlay = [blend(src, masks[mode, SEEDS[0]].cpu().numpy().astype("uint8")) for mode in ("fixed_raw", "ordinary", PRIMARY)]
            panels.append((f"{data} / {group} / {key}", [src, to_rgb(rgb), to_rgb(sham), *overlay, paint(gt.byte().cpu().numpy())],
                           ("Fog", "Restoration", "Matched color", "Fixed raw", "Ordinary cue", "Protected correction", "Reference")))
        groups.add(group)
        seen[data] += 1
    expected = dict(foggy_zurich_testv2=40, foggy_driving_fine=33) if external else dict(muses_full=58)
    if seen != expected:
        raise ValueError(f"Unexpected anchored evaluation: {seen}")
    comparisons = [(PRIMARY, key) for key in ("author_raw", "fixed_raw", "ordinary", "color_only", "subtracted_ce", "shifted")]
    scores = aggregate(table)
    intervals = paired_iou_intervals(table, comparisons, "method", "counts")
    name = "external" if external else "muses"
    dest, img = ROOT / f"{name}-individual.json", ROOT / f"{name}-predictions.png"
    atomic_write(dest, table)
    draw(img, panels)
    atomic_write(ROOT / f"{name}-test.json", dict(counts=dict(seen), rows=len(table), metrics=scores, intervals=intervals,
                 table_SHA256=digest(dest), images_SHA256=digest(img), data_SHA256=digest(DATA / "audit.json"),
                 calibration_SHA256=digest(lock), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in scores.items()}, flush=True)


def check_math():
    head = GuidedResidual().eval()
    nn.init.normal_(head.output.weight)
    x = torch.randn(1, 51, 12, 17)
    update = head(torch.cat((x, torch.zeros_like(x)), dim=1))
    torch.testing.assert_close(update, torch.zeros_like(update), atol=0., rtol=0.)
    print("Zero restoration cue produces exactly zero logit update.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("math", "smoke", "fit", "test", "external"))
    stage = parser.parse_args().stage
    if stage == "math":
        check_math()
    elif stage in ("smoke", "fit"):
        fit(stage == "smoke")
    else:
        test(stage == "external")

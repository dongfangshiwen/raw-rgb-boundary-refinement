"""Isolate a restoration feature from its global color and exposure counterpart."""
import argparse
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import AUTHOR_HASH, paired_iou_intervals
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import (
    CROP, DATA, INDEX, ROAD, SEEDS, SIZE, STRENGTHS, Frame, aggregate, all_cases, cases, center_logits, patch, paths
)
from scripts.road_transition_v68 import Engine, geometry, render_images
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/restoration-cue-v71"
MODES = ("raw", "ordinary", "color_only", "subtracted", "shifted")
PRIMARY = "subtracted"
STEPS = 400


def match_channels(raw, restored, knots=65):
    """Map each raw channel monotonically to the restored image's quantiles."""
    cut = torch.linspace(0., 1., knots, device=raw.device)
    global_values = torch.stack((raw[0], restored[0])).flatten(2)
    # The range endpoints include every pixel. Both images use the same fixed grid for intermediate knots.
    uniform = global_values[..., ::4].contiguous()
    measured = torch.quantile(uniform, cut[1:-1], dim=-1).permute(1, 2, 0)
    boundaries = torch.cat((global_values.amin(-1, keepdim=True), measured, global_values.amax(-1, keepdim=True)), dim=-1)
    source, destination = boundaries.unbind(0)
    orig = global_values[0].contiguous()
    above = torch.searchsorted(source.contiguous(), orig, right=True).clamp(1, knots - 1)
    below = above - 1
    a, b = source.gather(-1, below), source.gather(-1, above)
    ratio = ((orig - a) / (b - a).clamp_min(1e-7)).clamp(0., 1.)
    low, high = destination.gather(-1, below), destination.gather(-1, above)
    return (low + ratio * (high - low)).reshape_as(raw).clamp(0., 1.)


def evidence(features, mode):
    original, restored, color = features.split(1)
    if mode == "raw":
        hint = torch.zeros_like(original)
    elif mode == "ordinary":
        hint = restored - original
    elif mode == "color_only":
        hint = color - original
    else:
        hint = restored - color
        if mode == "shifted":
            hint = hint.roll(tuple(n // 2 for n in hint.shape[-2:]), dims=(-2, -1))
    return torch.cat((original, hint), dim=1)


class Adapter(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(102, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                         nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(), nn.Conv2d(32, 19, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def forward(self, cue):
        return 4. * (super().forward(cue) / 4.).tanh()


class SceneViews:
    def __init__(self):
        self.base = Engine()

    @torch.no_grad()
    def __call__(self, image):
        engine = self.base
        restored = render_images(geometry(image, engine.rgb), engine.bank, SEEDS[0])["utility_tone"]
        reference = match_channels(image, restored)
        entries = [engine.semantic((x - engine.mean) / engine.std) for x in (image, restored, reference)]
        shape = tuple(n // 8 for n in image.shape[-2:])
        logits = torch.cat([resize(x["logits"].float(), shape) for x in entries])
        features = torch.cat([resize(x["g_adverse"]["s8"].float(), shape) for x in entries])
        context = torch.cat((center_logits(logits), features.clamp(-10., 10.)), dim=1)
        return entries[0]["logits"].float(), logits[:1], context, restored, reference


def register():
    return dict(source_SHA256=digest(__file__), real_fit_SHA256=digest(DATA / "fit.json"), data_SHA256=digest(INDEX),
                rgb_training_SHA256=digest(ROAD / "rgb-completed.json"), lidseg_SHA256=AUTHOR_HASH,
                arms=list(MODES), steps=STEPS, seeds=list(SEEDS), crop=CROP,
                scope="The same frozen LiDSeg is run on raw, full restoration and raw with global per-channel "
                      "quantiles matched to the restoration. Compare same head with no correction cue, full restoration "
                      "minus raw, color control minus raw, restoration minus color control, and spatially shifted "
                      "subtracted difference. Center class logits to remove additive nonidentifiability. "
                      "Every model uses the same dense masks and balanced update draws. Freeze before calibration and scoring.",
                primary="Three-seed mean effect of subtraction versus ordinary restoration and raw-only adaptation, with "
                        "paired recording-resampling on all official MUSES validation and separately on first-use frames. "
                        "The color-only and shifted controls identify possible appearance or alignment explanations.")


@torch.no_grad()
def capture(engine, smoke):
    groups = defaultdict(list)
    counts = torch.zeros(19, dtype=torch.float64)
    for n, (_, group, _, x, target, _) in enumerate(paths("fit"), start=1):
        _, coarse, context, _, _ = engine(x)
        labels = F.interpolate(target[None, None].float(), coarse.shape[-2:], mode="nearest-exact")[0, 0].long().cpu()
        if not bool((labels != 255).any()):
            raise ValueError("Precise MUSES labels are missing at the training grid")
        counts += torch.bincount(labels[labels != 255], minlength=19)
        aligned = evidence(context, PRIMARY)
        shifted = evidence(context, "shifted")
        # Precompute the aligned and rolled cues on the whole frame so their spatial interventions match inference.
        group_entries = {name: value for name, value in ((PRIMARY, aligned), ("shifted", shifted))}
        for name in MODES[:3]:
            group_entries[name] = evidence(context, name)
        groups[group].append((coarse.half().cpu(), {k: v.half().cpu() for k, v in group_entries.items()}, labels.byte()))
        if n % 24 == 0:
            print(dict(encoded=n), flush=True)
        if smoke and n == 4:
            break
    if sum(map(len, groups.values())) != (4 if smoke else SIZE["fit"]):
        raise ValueError("The V71 fit population did not match registration")
    return tuple(groups.values()), ((counts.sum() / 19. + 1.) / (counts + 1.)).pow(.20).clamp(.65, 2.).cuda().float()


def fit(smoke):
    configure_cuda()
    note = register()
    protocol = ROOT / "protocol.json"
    if not smoke:
        if protocol.exists() and read(protocol) != note:
            raise ValueError("The color control protocol changed")
        atomic_write(protocol, note)
    engine = SceneViews()
    pool, class_weights = capture(engine, smoke)
    stats = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            path = ROOT / f"{mode}-{seed}.pth"
            ledger = path.with_suffix(".json")
            if not smoke and ledger.exists():
                item = read(ledger)
                if digest(path) != item["weights_SHA256"] or digest(protocol) != item["protocol_SHA256"]:
                    raise ValueError(f"A completed checkpoint changed: {path.name}")
                stats.append(item)
                continue
            set_seed(seed)
            model = Adapter().cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
            lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00006)
            rng = torch.Generator().manual_seed(seed + 71007)
            for step in range(3 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                terms = []
                for logits, cues, labels in cases(pool, rng):
                    signal = cues[mode]
                    state = patch(Frame(logits, signal, signal[:, :0], labels), rng)
                    correction = model(state.features)
                    loss = F.cross_entropy(state.logits + correction, state.labels[None], weight=class_weights, ignore_index=255)
                    terms.append(loss + .010 * correction.square().mean())
                objective = torch.stack(terms).mean()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
                opt.step()
                lr.step()
            item = dict(method=mode, seed=seed, steps=step + 1, loss=float(objective.detach()),
                        parameters=sum(v.numel() for v in model.parameters()))
            if not smoke:
                torch_save(path, {k: v.cpu() for k, v in model.state_dict().items()})
                item.update(weights_SHA256=digest(path), protocol_SHA256=digest(protocol))
                atomic_write(ledger, item)
                stats.append(item)
            print(item, flush=True)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=stats, steps=sum(x["steps"] for x in stats), protocol_SHA256=digest(protocol),
                     torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def models():
    note, proto = read(ROOT / "completed.json"), ROOT / "protocol.json"
    if digest(proto) != note["protocol_SHA256"] or digest(__file__) != read(proto)["source_SHA256"]:
        raise ValueError("The frozen study file changed")
    bank = {}
    for record in note["runs"]:
        m, seed = record["method"], record["seed"]
        path = ROOT / f"{m}-{seed}.pth"
        if digest(path) != record["weights_SHA256"]:
            raise ValueError(f"Changed research checkpoint: {path}")
        model = Adapter().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[m, seed] = model
    if set(bank) != {(mode, seed) for mode in MODES for seed in SEEDS}:
        raise ValueError("Incomplete registered comparison")
    return bank


@torch.inference_mode()
def calibrate(engine, bank):
    collected = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64))
    for _, _, _, raw, mask, _ in paths("calibrate"):
        logits, _, context, _, _ = engine(raw)
        original = resize(logits, mask.shape)
        for mode in MODES:
            cue = evidence(context, mode)
            for seed in SEEDS:
                delta = resize(bank[mode, seed](cue), mask.shape)
                for size in STRENGTHS:
                    y = (original + size * delta).argmax(1)[0]
                    collected[mode, size, seed] += confusion_matrix(y, mask, 19)
    result = {mode: {str(s): 100. * sum(segmentation_metrics(collected[mode, s, seed])["miou"] for seed in SEEDS) / len(SEEDS)
                     for s in STRENGTHS} for mode in MODES}
    chosen = {mode: float(max(values, key=lambda k: (values[k], -float(k)))) for mode, values in result.items()}
    complete = dict(chosen=chosen, scores=result, calib_SHA256=digest(DATA / "calibrate.json"),
                    training_SHA256=digest(ROOT / "completed.json"), finished_utc=now())
    atomic_write(ROOT / "calibration.json", complete)
    print(complete, flush=True)
    return complete


@torch.inference_mode()
def test(external):
    configure_cuda()
    packet = read(DATA / "audit.json")
    if not packet["recordings_disjoint"] or any(digest(DATA / f"{key}.json") != v for key, v in packet["manifests"].items()):
        raise ValueError("The recording or data audit changed")
    engine, bank = SceneViews(), models()
    lock = ROOT / "calibration.json"
    recognized = read(lock) if lock.exists() else calibrate(engine, bank)
    if recognized["calib_SHA256"] != digest(DATA / "calibrate.json") or recognized["training_SHA256"] != digest(ROOT / "completed.json"):
        raise ValueError("The color control calibration changed")
    previous = set(read(INDEX)["previous_scoring_ids"])
    rows, panels, seen, groups = [], [], Counter(), set()
    for data, key, group, lighting, x, truth, src in all_cases(external):
        logits, _, context, restored, matched = engine(x)
        orig = resize(logits, truth.shape)
        masks = {("author_raw", 0): orig.argmax(1)[0]}
        for arm in MODES:
            data_cue = evidence(context, arm)
            scale = recognized["chosen"][arm]
            for seed in SEEDS:
                score = orig + scale * resize(bank[arm, seed](data_cue), truth.shape)
                masks[arm, seed] = score.argmax(1)[0]
        for (method, seed), mask in masks.items():
            row = dict(dataset=data, id=key, group=group, lighting=lighting, method=method, seed=seed,
                       counts=confusion_matrix(mask, truth, 19).tolist())
            rows.append(row)
            if not external and key not in previous:
                rows.append({**row, "dataset": "muses_first_use"})
        if group not in groups and (not external or not seen[data]):
            overlays = [blend(src, masks[arm, SEEDS[0]].byte().cpu().numpy()) for arm in ("raw", "ordinary", PRIMARY)]
            panels.append((f"{data} / {group} / {key}", [src, to_rgb(restored), to_rgb(matched), *overlays, paint(truth.byte().cpu().numpy())],
                           ("Fog", "Restored RGB", "Matched color", "Raw adapter", "Ordinary restoration", "Subtracted cue", "Reference")))
        groups.add(group)
        seen[data] += 1
    expected = dict(foggy_zurich_testv2=40, foggy_driving_fine=33) if external else dict(muses_full=58)
    if seen != expected:
        raise ValueError(f"Unexpected color-control evaluation counts: {seen}")
    comparisons = [(PRIMARY, arm) for arm in ("author_raw", "raw", "ordinary", "color_only", "shifted")]
    summary = aggregate(rows)
    ci = paired_iou_intervals(rows, comparisons, "method", "counts")
    prefix = "external" if external else "muses"
    file, picture = ROOT / f"{prefix}-individual.json", ROOT / f"{prefix}-predictions.png"
    atomic_write(file, rows)
    draw(picture, panels)
    atomic_write(ROOT / f"{prefix}-test.json", dict(counts=dict(seen), metrics=summary, intervals=ci, rows=len(rows),
                 table_SHA256=digest(file), pictures_SHA256=digest(picture), calibration_SHA256=digest(lock),
                 data_SHA256=digest(DATA / "audit.json"), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in summary.items()}, flush=True)


def check_math():
    frame = torch.linspace(0., 1., 160).reshape(1, 1, 10, 16).expand(1, 3, 10, 16)
    torch.testing.assert_close(match_channels(frame, frame), frame)
    bright = frame * .65 + .1
    torch.testing.assert_close(match_channels(frame, bright), bright, atol=2e-6, rtol=2e-6)
    result = match_channels(frame, frame.square())
    if (result.flatten(2).diff() < -1e-7).any():
        raise ArithmeticError("A monotone control changed pixel order")
    print("Matched-color control preserves the channel order and reproduces affine color-only changes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("math", "smoke", "fit", "test", "external"))
    stage = parser.parse_args().stage
    if stage == "math":
        check_math()
    elif stage in ("smoke", "fit"):
        fit(smoke=stage == "smoke")
    else:
        test(external=stage == "external")

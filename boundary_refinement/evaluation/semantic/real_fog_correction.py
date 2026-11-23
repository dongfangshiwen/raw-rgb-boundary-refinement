"""Test candidate-class probability transfer using recording-disjoint real-fog supervision."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.evaluation.controls.audit_dual_output import AUTHOR_HASH, AUTHOR_RELEASE, paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_pairs, resize_image, verified_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.parallel_utility import capture_group
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.road_heads import ROOT as ROAD, Engine, TransitionHead, geometry, render_images
from boundary_refinement.data.staging.stage_muses_real_fog import INDEX, ROOT as DATA, SIZE
from boundary_refinement.evaluation.paired.tonal_agreement import SEEDS
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/real-fog-correction-v70"
ADAPTERS = ("raw_adapter", "shared_adapter")
GATES = ("fusion", "transfer", "transfer_supervised", "shifted")
MAIN = "transfer_supervised"
STEPS = {**dict.fromkeys(ADAPTERS, 400), **dict.fromkeys(GATES, 400)}
STRENGTHS = (0., .25, .50, .75, 1.)
VIEWS = ("raw", "utility", "utility_tone")
CROP = 56


class ViewAdapter(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(51, 30, 3, padding=1), nn.GroupNorm(5, 30), nn.SiLU(),
                         nn.Conv2d(30, 30, 3, padding=1), nn.GroupNorm(5, 30), nn.SiLU(), nn.Conv2d(30, 19, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def forward(self, logits, features):
        y = torch.cat((center_logits(logits), features.clamp(-10., 10.)), dim=1)
        return 4. * (super().forward(y) / 4.).tanh()


@dataclass
class Frame:
    logits: torch.Tensor
    features: torch.Tensor
    edits: torch.Tensor
    labels: torch.Tensor


def center_logits(logits):
    return (logits - logits.mean(1, keepdim=True)).clamp(-20., 20.) / 4.


class FrozenViews:
    def __init__(self):
        self.base = Engine()

    @torch.no_grad()
    def __call__(self, raw):
        core = self.base
        rgb = render_images(geometry(raw, core.rgb), core.bank, SEEDS[0])
        images = {"raw": raw, **rgb}
        source = [core.semantic((images[tag] - core.mean) / core.std) for tag in VIEWS]
        logits = torch.cat([x["logits"].float() for x in source])
        if logits.shape != (3, 19, *raw.shape[-2:]):
            raise ValueError("The frozen LiDSeg tensor geometry changed")
        shape = tuple(n // 8 for n in raw.shape[-2:])
        guide = torch.cat([resize(x["g_adverse"]["s8"].float(), shape) for x in source])
        if guide.shape[1] != 32:
            raise ValueError("The published guidance feature count changed")
        edits = resize(torch.cat([4. * (images[tag] - raw) for tag in VIEWS[1:]], dim=1), shape)
        return logits, guide, edits, rgb["utility_tone"]


def lock_training():
    return dict(code_SHA256=digest(__file__), registration_SHA256=digest(INDEX), fit_SHA256=digest(DATA / "fit.json"),
                rgb_SHA256=digest(ROAD / "rgb-completed.json"), author_release=AUTHOR_RELEASE, author_SHA256=AUTHOR_HASH,
                seeds=list(SEEDS), steps=STEPS, crop=CROP, strengths=list(STRENGTHS),
                scenario="Freeze the two existing RGB corrections and LiDSeg. Train on 96 fine masks from nine official "
                         "training recordings, separately from calibration and evaluation recordings. Train same-architecture "
                         "raw-only and shared-view adapters with ordinary CE. All subsequent gates freeze and share the "
                         "shared-view adapter, receive identical logits and RGB edits, and use 23,274 parameters. Compare "
                         "learned convex fusion, candidate-class mass transfer, adding correction/harm event supervision, "
                         "and the same supervised objective with shifted restoration cues. Preserve uninvolved classes. "
                         "Calibrate blend strengths only on 24 disjoint training images. Test at the original label grid.",
                primary="First-use official-val frames. Need positive group-paired effects against both shared raw and "
                        "ordinary fusion, replicated on all-val, and compare the same-data raw-only adapter. "
                        "Five recordings limit inferential precision. No tested frame used to change weights or choose a strength.")


def paths(scope):
    note = read(DATA / f"{scope}.json")
    if note["registration_SHA256"] != digest(INDEX) or len(note["pairs"]) != SIZE[scope]:
        raise ValueError(f"Unregistered real fog set: {scope}")
    for row in note["pairs"]:
        a, b = row["frame_camera"], row["gt_semantic"]
        rgb = verified_image(a["file"], a["SHA256"], False)
        truth = verified_image(b["file"], b["SHA256"], True)
        raw, _ = resize_image(rgb)
        yield row["id"], row["group"], row["daytime"], raw, torch.from_numpy(truth.copy()).cuda().long(), rgb


@torch.no_grad()
def make_pool(source, smoke):
    groups = defaultdict(list)
    class_count = torch.zeros(19, dtype=torch.float64)
    for index, (_, group, _, raw, target, _) in enumerate(paths("fit"), start=1):
        full, feature, edits, _ = source(raw)
        small = resize(full, feature.shape[-2:])
        label = F.interpolate(target[None, None].float(), small.shape[-2:], mode="nearest-exact")[0, 0].long().cpu()
        valid = label != 255
        if not bool(valid.any()):
            raise ValueError(f"All labels disappeared in group {group}")
        class_count += torch.bincount(label[valid], minlength=19)
        groups[group].append(Frame(small.half().cpu(), feature.half().cpu(), edits.half().cpu(), label.byte()))
        if index % 24 == 0:
            print(dict(encoded_real_fit=index), flush=True)
        if smoke and index == 4:
            break
    records = sum(map(len, groups.values()))
    if records != (4 if smoke else SIZE["fit"]):
        raise ValueError("The real-fog mini-batch population is incomplete")
    weights = ((class_count.sum() / 19. + 1.) / (class_count + 1.)).pow(.20).clamp(.65, 2.0).cuda().float()
    return tuple(groups.values()), weights


def patch(frame, rng):
    height, width = frame.labels.shape
    size = min(CROP, height, width)
    top, left = (int(torch.randint(n - size + 1, (), generator=rng)) for n in frame.labels.shape)
    if not bool((frame.labels[top:top + size, left:left + size] != 255).any()):
        valid = (frame.labels != 255).nonzero()
        row, col = valid[torch.randint(len(valid), (), generator=rng)].tolist()
        top = min(height - size, max(0, row - size // 2))
        left = min(width - size, max(0, col - size // 2))
    view = (..., slice(top, top + size), slice(left, left + size))
    args = (frame.logits, frame.features, frame.edits)
    return Frame(*(x[view].cuda().float() for x in args), frame.labels[view].cuda().long())


def cases(pool, rng):
    # Sample recordings uniformly so a long burst does not dominate the fit.
    group_indices = torch.randint(len(pool), (2,), generator=rng).tolist()
    for i in group_indices:
        image_index = int(torch.randint(len(pool[i]), (), generator=rng))
        yield pool[i][image_index]


def adapter_loss(model, frame, class_weights, epoch_view, method):
    i = 0 if method == "raw_adapter" else epoch_view
    logits, feats = frame.logits[i:i + 1], frame.features[i:i + 1]
    shift = model(logits, feats)
    y = F.cross_entropy(logits + shift, frame.labels[None], weight=class_weights, ignore_index=255)
    return y + .010 * shift.square().mean()


def cues(logits, edits, shifted=False):
    centered = center_logits(logits)
    offset = torch.cat(((centered[1:] - centered[:1]).flatten(0, 1)[None], edits), dim=1)
    if shifted:
        offset = offset.roll(tuple(n // 2 for n in offset.shape[-2:]), dims=(-2, -1))
    return torch.cat((centered[:1], offset), dim=1)


def steps_from_evidence(value):
    score = value[:, 1] - value[:, 2] - .5 * value[:, 0] - 2.
    return torch.cat((torch.zeros_like(score[:1]), score))


def transfer_mass(probabilities, weights):
    initial = probabilities[0]
    start = initial.argmax(0, keepdim=True)
    recipient = probabilities[1:].argmax(1)
    sent = initial.gather(0, start) * weights[1:]
    return initial.scatter_add(0, recipient, sent).scatter_add(0, start, -sent.sum(0, keepdim=True))


def gate_loss(model, frame, class_weights, method):
    p = frame.logits.softmax(1)
    value = model(frame.features).unflatten(1, (2, 3))[0]
    weights = steps_from_evidence(value).softmax(0)
    mixed = (weights[:, None] * p).sum(0) if method == "fusion" else transfer_mass(p, weights)
    result = F.nll_loss(mixed.clamp_min(1e-8).log()[None], frame.labels[None], weight=class_weights, ignore_index=255)
    if method in (MAIN, "shifted"):
        predicted = frame.logits.argmax(1)
        valid = frame.labels != 255
        changes = (predicted[0] != predicted[1:]) & valid
        rescue = (predicted[1:] == frame.labels) & changes
        harm = (predicted[0] == frame.labels) & changes
        target = rescue.long() + 2 * harm.long()
        errors = F.cross_entropy(value, target, reduction="none")
        result = result + .20 * (errors * changes).sum() / changes.sum().clamp_min(1)
    return result


@torch.no_grad()
def precompute_gate(data, adaptation):
    packed = []
    for group in data:
        records = []
        for row in group:
            logits, features = row.logits.cuda().float(), row.features.cuda().float()
            corrected = logits + adaptation(logits, features)
            edits = row.edits.cuda().float()
            records.append(Frame(corrected.half().cpu(), cues(corrected, edits).half().cpu(),
                                 cues(corrected, edits, shifted=True).half().cpu(), row.labels))
        packed.append(records)
    return packed


def train_one(method, seed, pool, loss, smoke):
    set_seed(seed)
    model = (ViewAdapter if method in ADAPTERS else TransitionHead)().cuda()
    path = ROOT / f"{method}-{seed}.pth"
    meta = path.with_suffix(".json")
    lock = ROOT / "protocol.json"
    if not smoke and meta.exists():
        item = read(meta)
        if item["weights_SHA256"] != digest(path) or item["protocol_SHA256"] != digest(lock):
            raise ValueError(f"Run no longer matches its training evidence: {meta.name}")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return model.eval().requires_grad_(False), item
    images = torch.Generator().manual_seed(seed + 70019)
    view_rng = torch.Generator().manual_seed(seed + 70031)
    opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    anneal = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS[method], eta_min=.00006)
    for i in range(3 if smoke else STEPS[method]):
        opt.zero_grad(set_to_none=True)
        terms = []
        for original in cases(pool, images):
            small = patch(original, images)
            view = int(torch.randint(3, (), generator=view_rng))
            terms.append(loss(model, small, view))
        error = torch.stack(terms).mean()
        error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        anneal.step()
    item = dict(method=method, seed=seed, steps=i + 1, loss=float(error.detach()),
                parameters=sum(x.numel() for x in model.parameters()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        item.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
        atomic_write(meta, item)
    print(item, flush=True)
    return model.eval().requires_grad_(False), item


def fit(smoke):
    configure_cuda()
    lock = ROOT / "protocol.json"
    protocol = lock_training()
    if not smoke:
        if lock.exists() and read(lock) != protocol:
            raise ValueError("Real-fog training protocol changed during an existing run")
        atomic_write(lock, protocol)
    bank, class_weights = make_pool(FrozenViews(), smoke)
    logs = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for arm in ADAPTERS:
            def loss(net, sample, view):
                return adapter_loss(net, sample, class_weights, view, arm)
            trained, item = train_one(arm, seed, bank, loss, smoke)
            logs.append(item)
        gate_pool = precompute_gate(bank, trained)
        for arm in GATES:
            def loss(net, sample, _):
                if arm == "shifted":
                    sample = Frame(sample.logits, sample.edits, sample.edits, sample.labels)
                return gate_loss(net, sample, class_weights, arm)
            _, item = train_one(arm, seed, gate_pool, loss, smoke)
            logs.append(item)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=logs, steps=sum(x["steps"] for x in logs),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def load_fits():
    plan, summary = ROOT / "protocol.json", read(ROOT / "completed.json")
    if digest(plan) != summary["protocol_SHA256"] or not matches_registered(read(plan)['code_SHA256'], digest(__file__)):
        raise ValueError("Training code or provenance changed")
    bank = {}
    for row in summary["runs"]:
        label, seed = row["method"], row["seed"]
        path = ROOT / f"{label}-{seed}.pth"
        if digest(path) != row["weights_SHA256"]:
            raise ValueError(f"Corrupted or different gate checkpoint {label}/{seed}")
        model = (ViewAdapter if label in ADAPTERS else TransitionHead)().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[label, seed] = model
    if set(bank) != {(arm, seed) for arm in (*ADAPTERS, *GATES) for seed in SEEDS}:
        raise ValueError("The registered real-fog training arms are incomplete")
    return bank


def prepared(logits, features, edits, fit_bank, seed, shape):
    small = resize(logits, features.shape[-2:])
    adapter = fit_bank["shared_adapter", seed]
    changes = adapter(small, features)
    state = resize(logits, shape) + resize(changes, shape)
    p = state.softmax(1)
    raw = resize(logits[0:1], shape) + resize(fit_bank["raw_adapter", seed](small[0:1], features[0:1]), shape)
    return p, raw, (small + changes, edits)


def action(bank, seed, state, probabilities, method):
    signal = cues(*state, shifted=method == "shifted")
    values = bank[method, seed](signal).unflatten(1, (2, 3))[0]
    weights = resize(steps_from_evidence(values)[None], probabilities.shape[-2:]).softmax(1)[0]
    if method == "fusion":
        return (weights[:, None] * probabilities).sum(0)
    return transfer_mass(probabilities, weights)


@torch.inference_mode()
def calibrate(source, models):
    # Calculate calibration on the original label grid. Evaluation follows exactly the same resampling.
    budget = defaultdict(lambda: torch.zeros((19, 19), dtype=torch.int64))
    for _, _, _, x, truth, _ in paths("calibrate"):
        dense, f, delta, _ = source(x)
        for seed in SEEDS:
            p, _, state = prepared(dense, f, delta, models, seed, truth.shape)
            for mode in GATES:
                change = action(models, seed, state, p, mode) - p[0]
                for s in STRENGTHS:
                    y = (p[0] + s * change).argmax(0)
                    budget[mode, s, seed] += confusion_matrix(y, truth, 19)
    quality = {method: {str(s): 100. * mean(segmentation_metrics(budget[method, s, seed])["miou"] for seed in SEEDS)
                        for s in STRENGTHS} for method in GATES}
    chosen = {name: float(max(quality[name], key=lambda s: (quality[name][s], -float(s)))) for name in GATES}
    result = dict(chosen=chosen, scores=quality, calibration_SHA256=digest(DATA / "calibrate.json"),
                  training_SHA256=digest(ROOT / "completed.json"), finished_utc=now())
    atomic_write(ROOT / "calibration.json", result)
    print(result, flush=True)
    return result


def all_cases(external):
    if not external:
        yield from (("muses_full", *item) for item in paths("held"))
        return
    for data in ("foggy_zurich_testv2", "foggy_driving_fine"):
        for key, group, source, mask in load_pairs(data):
            x, _ = resize_image(source)
            yield data, key, group or capture_group(key), "", x, torch.from_numpy(mask.copy()).cuda().long(), source


def aggregate(rows):
    all_scores = defaultdict(list)
    for item in rows:
        all_scores[item["dataset"], item["method"], item["seed"]].append(item["counts"])
    out = defaultdict(dict)
    for (data, method, seed), values in all_scores.items():
        c = np.sum(values, axis=0, dtype=np.int64)
        out[data, method][str(seed)] = segmentation_metrics(torch.from_numpy(c))["miou"] * 100.
    return {"/".join(key): dict(miou=mean(v.values()), by_seed=v) for key, v in out.items()}


@torch.inference_mode()
def test(external):
    configure_cuda()
    audit = read(DATA / "audit.json")
    if not audit["recordings_disjoint"] or any(digest(DATA / f"{name}.json") != checksum for name, checksum in audit["manifests"].items()):
        raise ValueError("Real fog imagery or grouping audits changed")
    source, models = FrozenViews(), load_fits()
    path = ROOT / "calibration.json"
    choices = read(path) if path.exists() else calibrate(source, models)
    if choices["calibration_SHA256"] != digest(DATA / "calibrate.json") or choices["training_SHA256"] != digest(ROOT / "completed.json"):
        raise ValueError("Calibration is stale")
    prev = set(read(INDEX)["previous_scoring_ids"])
    rows, panels, seen, by_group = [], [], Counter(), Counter()
    for data, key, group, lighting, raw, truth, src in all_cases(external):
        logits, f, edits, rgb = source(raw)
        start = resize(logits[:1], truth.shape).argmax(1)[0]
        predictions = {("author_raw", 0): start}
        for seed in SEEDS:
            p, corrected, state = prepared(logits, f, edits, models, seed, truth.shape)
            predictions["raw_adapter", seed] = corrected.argmax(1)[0]
            predictions["shared_adapter", seed] = p[0].argmax(0)
            for mode in GATES:
                final = action(models, seed, state, p, mode)
                used = choices["chosen"][mode]
                y = (p[0] + used * (final - p[0])).argmax(0)
                predictions[mode, seed] = y
                if mode == MAIN:
                    predictions[MAIN + "_unshrunk", seed] = final.argmax(0)
        valid, raw_ok = truth != 255, start == truth
        for (method, seed), label in predictions.items():
            note = dict(dataset=data, id=key, group=group, lighting=lighting, method=method, seed=seed,
                        counts=confusion_matrix(label, truth, 19).tolist(),
                        corrected=int((valid & ~raw_ok & (label == truth)).sum()),
                        damaged=int((valid & raw_ok & (label != truth)).sum()))
            rows.append(note)
            if not external and key not in prev:
                rows.append({**note, "dataset": "muses_first_use"})
        if not by_group[group] and (not external or not seen[data]):
            keys = (("author_raw", 0), ("raw_adapter", SEEDS[0]), ("fusion", SEEDS[0]), (MAIN, SEEDS[0]))
            overlay = [blend(src, predictions[m].byte().cpu().numpy()) for m in keys]
            panels.append((f"{data} / {group} / {key}", [src, to_rgb(rgb), *overlay, paint(truth.byte().cpu().numpy())],
                           ("Fog image", "Restoration", "Author LiDSeg", "Raw adapter", "Fusion", "Candidate transfer", "Reference")))
        seen[data] += 1
        by_group[group] += 1
        if seen[data] % 20 == 0:
            print(dict(scored=data, images=seen[data]), flush=True)
    expected = dict(foggy_zurich_testv2=40, foggy_driving_fine=33) if external else dict(muses_full=58)
    if seen != expected:
        raise ValueError(f"Evaluation population changed: {seen}")
    controls = ("author_raw", "raw_adapter", "shared_adapter", "fusion", "transfer", "shifted")
    comparisons = [(MAIN, method) for method in controls]
    summary = aggregate(rows)
    ci = paired_iou_intervals(rows, comparisons, "method", "counts")
    suffix = "external" if external else "muses"
    table, image = ROOT / f"{suffix}-individual.json", ROOT / f"{suffix}-predictions.png"
    atomic_write(table, rows)
    draw(image, panels)
    atomic_write(ROOT / f"{suffix}-test.json", dict(counts=dict(seen), rows=len(rows), metrics=summary, contrasts=ci,
                 calibration_SHA256=digest(path), protocol_SHA256=digest(ROOT / "protocol.json"), data_SHA256=digest(DATA / "audit.json"),
                 table_SHA256=digest(table), pictures_SHA256=digest(image), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in summary.items()}, flush=True)


def check_math():
    torch.manual_seed(70097)
    p = torch.randn(3, 19, 5, 7).softmax(1)
    weights = torch.randn(3, 5, 7).softmax(0)
    value = transfer_mass(p, weights)
    torch.testing.assert_close(value.sum(0), torch.ones(5, 7))
    if value.min() < -1e-6:
        raise ArithmeticError("Probability transfer produced negative mass")
    unchanged = torch.ones_like(p[0], dtype=torch.bool)
    unchanged.scatter_(0, p.argmax(1), False)
    torch.testing.assert_close(value[unchanged], p[0][unchanged])
    torch.testing.assert_close(transfer_mass(p[:1].expand_as(p), weights), p[0])
    print("Verified conservation, identity when views agree, and unchanged uninvolved class probabilities.", flush=True)


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

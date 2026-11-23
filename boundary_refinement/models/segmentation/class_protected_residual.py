"""Train class-protected residual segmentation under a frozen restoration branch."""
import argparse
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.anchored_cue import GuidedResidual
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import CROP, Frame, aggregate, cases, patch
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band, compress_scores, segmentation_loss
from boundary_refinement.models.segmentation.restoration_proposals import EVAL_SIZE, FIT_SIZE, SEEDS, STEPS, base_offset, parents, pool
from boundary_refinement.models.segmentation.soft_proposals import EXTERNAL, ROOT as V79, evaluation_cases
from boundary_refinement.evaluation.semantic.spatial_restoration import RestorationFeatures
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/class-protected-residual-v81"
ARMS = {
    "raw_ce": ("raw_refined", "ce"), "rgb_ce": ("rgb_fusion", "ce"),
    "head_ce": ("correction_heads", "ce"), "head_natural_ce": ("correction_heads", "ce"),
    "head_lovasz": ("correction_heads", "lovasz"), "head_pooled": ("correction_heads", "pooled"),
    "head_protected": ("correction_heads", "protected"), "raw_protected": ("raw_refined", "protected"),
    "rgb_protected": ("rgb_fusion", "protected"), "shift_protected": ("shift_heads", "protected"),
}
PRIMARY = "head_protected"
RARE_FRACTION = .02
PRIOR_MASS = .75


def protocol(split):
    return dict(source_SHA256=digest(__file__), parent_SHA256=digest(V79 / split / "protocol.json"), split=split,
                arms={name: list(value) for name, value in ARMS.items()}, steps=STEPS, params=26208, nuisance="No RGB weight updates",
                trials=list(SEEDS), crop=CROP, rare_threshold=RARE_FRACTION, prior_mass=PRIOR_MASS, fixed_strength=1.,
                mechanism="Train unrestricted residual logits for all classes. Compute classwise changes in soft intersection "
                          "and union relative to a frozen raw anchor, shrinking patch counts toward its natural training statistics. "
                          "Penalize negative IoU changes on classes below 2% training prevalence. In each matched update, pair "
                          "one recording-uniform crop with one class-uniform rare-centred crop. Fit raw, RGB, aligned and shifted "
                          "feature controls, pooled IoU without protection and Lovasz-Softmax. Check both balanced and natural CE.",
                scoring="Use a fixed unit amplitude, both reused recording-disjoint MUSES splits and repeated external testing. "
                        "Score full 19-class mIoU, training-defined present rare classes and their recall. The two old MUSES sets "
                        "motivated the design and are reused evidence, not a blind confirmation.")


def soft_counts(logits, labels):
    valid = labels != 255
    one_hot = F.one_hot(labels.clamp(0, 18), 19).movedim(-1, 1) * valid[:, None]
    probability = logits.softmax(1) * valid[:, None]
    axes = (0, 2, 3)
    intersection = (probability * one_hot).sum(axes)
    classes = one_hot.sum(axes)
    return intersection, probability.sum(axes) + classes - intersection, classes, valid.sum()


def prior_iou(before, after, labels, reference):
    a, u, _, _ = soft_counts(before, labels)
    b, v, _, n = soft_counts(after, labels)
    prior_i, prior_u, _, _ = reference
    shared_i, shared_union = PRIOR_MASS * n * prior_i, PRIOR_MASS * n * prior_u
    initial = (a + shared_i) / (u + shared_union).clamp_min(1e-4)
    final = (b + shared_i) / (v + shared_union).clamp_min(1e-4)
    return final - initial


def lovasz_softmax(logits, target):
    """Lovasz extension averaged over target classes present in the mini-batch."""
    p = logits.softmax(1).movedim(1, -1)
    valid = target != 255
    p, y = p[valid], target[valid]
    losses = []
    for k in torch.unique(y).tolist():
        truth = (y == k).float()
        error, order = (truth - p[:, k]).abs().sort(descending=True)
        sorted_truth = truth[order]
        positives = sorted_truth.sum()
        remaining = positives - sorted_truth.cumsum(0)
        union = positives + (1. - sorted_truth).cumsum(0)
        jaccard = 1. - remaining / union
        steps = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        losses.append((error * steps).sum())
    return torch.stack(losses).mean()


@torch.no_grad()
def anchor_data(data, frozen):
    totals = torch.zeros(3, 19, device="cuda", dtype=torch.float64)
    known = 0
    groups = []
    for original in data:
        frames = []
        for low, cues, labels in original:
            props = {k: cues[k].cuda().float() for k in ("raw", "raw_refined")}
            baseline = low.cuda().float() + base_offset(props, frozen)
            i, u, c, n = soft_counts(baseline, labels.cuda()[None].long())
            totals += torch.stack((i, u, c)).double()
            known += int(n)
            frames.append((baseline.half().cpu(), cues, labels))
        groups.append(frames)
    i, u, prevalence = (totals / known).float()
    eligible = (prevalence > 0) & (prevalence < RARE_FRACTION)
    return groups, (i, u, prevalence, eligible)


def class_examples(groups, rare):
    ids = rare.nonzero().flatten().tolist()
    items = {k: [] for k in ids}
    for group in groups:
        for frame in group:
            for k in ids:
                positions = (frame[-1] == k).nonzero()
                if len(positions):
                    items[k].append((frame, positions))
    if not items or any(not choices for choices in items.values()):
        raise ValueError("The fit-class sampling list is empty")
    return tuple(items.values())


def group_sample(groups, examples, rng, natural):
    random_frames = list(cases(groups, rng))
    yield random_frames[0], None
    if natural:
        yield random_frames[1], None
        return
    samples = examples[int(torch.randint(len(examples), (), generator=rng))]
    frame, pixels = samples[int(torch.randint(len(samples), (), generator=rng))]
    point = pixels[int(torch.randint(len(pixels), (), generator=rng))].tolist()
    yield frame, point


def training_crop(frame, signal, rng, point):
    base, cues, truth = frame
    sample = cues[signal]
    if point is None:
        return patch(Frame(base, sample, sample[:, :0], truth), rng)
    h, w = truth.shape
    size = min(CROP, h, w)
    dy, dx = torch.randint(max(1, size // 2), (2,), generator=rng).tolist()
    y = max(0, min(h - size, point[0] - size // 4 - dy))
    x = max(0, min(w - size, point[1] - size // 4 - dx))
    cut = (..., slice(y, y + size), slice(x, x + size))
    signal = sample[cut].cuda().float()
    return Frame(base[cut].cuda().float(), signal, signal[:, :0], truth[cut].cuda().long())


def objective(mode, before, offset, labels, weight, prior):
    full = segmentation_loss(before, offset, labels, weight)
    if mode == "lovasz":
        return full + 1.5 * lovasz_softmax(before + offset, labels)
    if mode == "ce":
        return full
    gain = prior_iou(before, before + offset, labels, prior)
    present = prior[2] > 0
    loss = full - 1.5 * gain[present].mean()
    if mode == "protected":
        loss = loss + 3. * (-gain[prior[3]]).relu().mean()
    return loss


def fit_one(root, arm, seed, records, examples, weight, prior, smoke):
    file = root / f"{arm}-{seed}.pth"
    ledger, lock = file.with_suffix(".json"), root / "protocol.json"
    if not smoke and ledger.is_file():
        old = read(ledger)
        if old["weights_SHA256"] != digest(file) or old["protocol_SHA256"] != digest(lock):
            raise ValueError(f"The matched fit does not match the registered scope {file.name}")
        return old
    set_seed(seed)
    model = GuidedResidual().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00006)
    rng = torch.Generator().manual_seed(seed + 81003)
    signal, mode = ARMS[arm]
    steps = 3 if smoke else STEPS
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        block = [training_crop(frame, signal, rng, centre) for frame, centre in
                 group_sample(records, examples, rng, arm == "head_natural_ce")]
        before = torch.cat([r.logits for r in block])
        delta = model(torch.cat([r.features for r in block]))
        y = torch.stack([r.labels for r in block])
        loss = objective(mode, before, delta, y, weight, prior)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    out = dict(method=arm, seed=seed, updates=steps, params=sum(v.numel() for v in model.parameters()), loss=float(loss.detach()))
    if not smoke:
        torch_save(file, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        out.update(weights_SHA256=digest(file), protocol_SHA256=digest(lock))
        atomic_write(ledger, out)
    print(out, flush=True)
    return out


def fit(split, smoke):
    configure_cuda()
    root = ROOT / split
    lock, note = root / "protocol.json", protocol(split)
    if not smoke:
        if lock.is_file() and read(lock) != note:
            raise ValueError("The class-level experiment inputs changed")
        atomic_write(lock, note)
    raw = parents(split)
    cached, weights = pool(RestorationFeatures(), split, smoke)
    records, priors = [], {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        anchored, priori = anchor_data(cached, raw[seed])
        locations = class_examples(anchored, priori[3])
        priors[str(seed)] = dict(intersection=priori[0].tolist(), union=priori[1].tolist(), prevalence=priori[2].tolist(),
                                 rare=priori[3].nonzero().flatten().tolist())
        records.extend(fit_one(root, arm, seed, anchored, locations, weights, priori, smoke) for arm in ARMS)
    if not smoke:
        atomic_write(root / "completed.json", dict(runs=records, steps=sum(x["updates"] for x in records), priors=priors,
                     proto_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def load(split):
    root = ROOT / split
    lock, state = root / "protocol.json", read(root / "completed.json")
    if read(lock) != protocol(split) or digest(lock) != state["proto_SHA256"]:
        raise ValueError("The per-class fit scope is inconsistent")
    models = {}
    for row in state["runs"]:
        mode, seed = row["method"], row["seed"]
        file = root / f"{mode}-{seed}.pth"
        if digest(file) != row["weights_SHA256"] or row["params"] != 26208 or row["updates"] != STEPS:
            raise ValueError(f"Unexpected class-protected model {file.name}")
        net = GuidedResidual().cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(file, map_location="cpu", weights_only=True), strict=True)
        models[mode, seed] = net
    if set(models) != {(k, s) for k in ARMS for s in SEEDS}:
        raise ValueError("Missing class-protection controls")
    return models


@torch.inference_mode()
def score(split, part):
    from boundary_refinement.models.segmentation.restoration_affinity import displacement

    configure_cuda()
    root = ROOT / split
    engine, bank, frozen = RestorationFeatures(), load(split), parents(split)
    counts, rows, panels = Counter(), [], {}
    for dataset, key, group, lighting, x, truth, original in evaluation_cases(split, part):
        dense, _, features, rgb, _ = engine(x)
        features["shift_heads"] = displacement(features)
        base = resize(dense, truth.shape)
        edge = None if dataset == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        for seed in SEEDS:
            z = base + resize(base_offset(features, frozen[seed]), truth.shape)
            outputs = {"author_raw": base.argmax(1)[0], "raw_anchor": z.argmax(1)[0]}
            for name, (signal, _) in ARMS.items():
                out = bank[name, seed](features[signal])
                outputs[name] = (z + resize(out, truth.shape)).argmax(1)[0]
            for name, pred in outputs.items():
                row = dict(dataset=dataset, id=key, group=group, method=name, seed=seed,
                           counts=confusion_matrix(pred, truth, 19).tolist())
                if edge is not None:
                    row["edge"] = confusion_matrix(pred, edge, 19).tolist()
                rows.append(row)
            g = dataset, group
            if seed == SEEDS[0] and g not in panels and sum(k[0] == dataset for k in panels) < (5 if part == "held" else 2):
                show = [blend(original, outputs[m].byte().cpu().numpy()) for m in ("raw_ce", "rgb_ce", "head_ce", PRIMARY, "shift_protected")]
                panels[g] = (f"{key} / {lighting or dataset}", [original, to_rgb(rgb), *show, paint(truth.byte().cpu().numpy())],
                             ("Fog", "Restored RGB", "Raw CE", "RGB fusion", "Internal CE", "Class protection", "Shifted protection", "Reference"))
        counts[dataset] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(split=split, stage=part, scored=sum(counts.values())), flush=True)
    if (part == "held" and sum(counts.values()) != EVAL_SIZE[split]["held"]) or (part == "external" and counts != EXTERNAL):
        raise ValueError("The class-level evaluation cohort changed")
    full = rows + [{**r, "dataset": "foggy_driving_101"} for r in rows if r["dataset"].startswith("foggy_driving_")]
    boundaries = [{**r, "counts": r["edge"]} for r in rows if "edge" in r]
    comparison = [(PRIMARY, m) for m in ("author_raw", "raw_anchor", *ARMS) if m != PRIMARY]
    comparison += [("head_ce", "head_natural_ce"), ("raw_protected", "raw_ce"), ("rgb_protected", "rgb_ce")]
    summary, intervals = {}, {}
    for region, data in (("full", full), ("boundary", boundaries)):
        summary[region] = aggregate(data)
        intervals[region] = paired_iou_intervals(data, comparison, "method", "counts")
    file, preview = root / f"{part}-individual.json.gz", root / f"{part}-predictions.png"
    compress_scores(file, rows)
    draw(preview, list(panels.values()))
    atomic_write(root / f"{part}-test.json", dict(counts=dict(counts), metrics=summary, intervals=intervals,
                 weights_SHA256=digest(root / "completed.json"), rows_SHA256=digest(file), picture_SHA256=digest(preview),
                 RGB_frozen=True, finished_utc=now()))
    print({d: round(s["miou"], 4) for d, s in summary["full"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("smoke", "fit", "held", "external"))
    parser.add_argument("split", choices=tuple(FIT_SIZE))
    setup = parser.parse_args()
    if setup.part in ("smoke", "fit"):
        fit(setup.split, setup.part == "smoke")
    else:
        score(setup.split, setup.part)

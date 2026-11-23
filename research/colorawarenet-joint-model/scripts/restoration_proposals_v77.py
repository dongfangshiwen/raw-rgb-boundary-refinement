"""Test whether restoration-exclusive class proposals can repair fog segmentation."""
import argparse
from collections import Counter, defaultdict
from statistics import mean

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from dehaze_seg.utils import set_seed
from scripts.anchored_cue_v72 import GuidedResidual, anchor_fit
from scripts.audit_dual_output_v66 import paired_iou_intervals
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import DATA, SEEDS, Frame, aggregate, cases, patch, paths
from scripts.recording_restore_v74 import DATA as DATA_REFIT, ROOT as REFIT, check_data, model_bank, pairs
from scripts.restoration_affinity_v75 import (
    HELD, boundary_band, compress_scores, displacement, neighbors, registered_cases, segmentation_loss,
)
from scripts.restoration_cue_v71 import ROOT as RAW_ANCHOR
from scripts.restoration_descent_v76 import ROOT as DESCENT, trained
from scripts.spatial_restoration_v73 import RestorationFeatures
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/restoration-proposals-v77"
STEPS = 400
FIT_SIZE = {"large": 96, "refit": 104}
EVAL_SIZE = {"large": {"calibrate": 24, "held": 186}, "refit": {"calibrate": 9, "held": 44}}
DIRECT = dict(raw_residual="raw_refined", rgb_residual="rgb_fusion", head_residual="correction_heads")
RISK = dict(head_exclusive="correction_heads", raw_exclusive="raw_refined", rgb_exclusive="rgb_fusion",
            shift_exclusive="shift_heads", head_all="correction_heads", color_exclusive="color_match")
METHODS = {**DIRECT, **RISK}
PRIMARY = "head_exclusive_iou"
RISK_MODES = {tag + "_" + value: (tag, value) for tag in RISK for value in ("accuracy", "iou")}
SWITCH = (None, 0., .025, .05, .10, .20, .35, .50, .75)
SCALE = (0., .25, .50, .75, 1.)
QUOTA = .10
QUOTA_ARMS = ("head_exclusive_iou", "head_exclusive_accuracy", "raw_exclusive_iou", "rgb_exclusive_iou",
              "shift_exclusive_iou", "probability_average")


def data_dependencies(split):
    if split == "large":
        return [DATA / "fit.json", DATA / "calibrate.json", HELD, RAW_ANCHOR / "completed.json",
                RAW_ANCHOR / "calibration.json", DESCENT / "completed.json"]
    return [DATA_REFIT / f"{part}.json" for part in ("fit", "calibrate", "held", "audit")] + [
        DATA / "held.json", REFIT / "completed.json", REFIT / "protocol.json"]


def protocol(split):
    return dict(code_SHA256=digest(__file__), split=split,
                dependencies={str(p): digest(p) for p in data_dependencies(split)}, seeds=list(SEEDS),
                methods=METHODS, steps=STEPS, calibration_residuals=list(SCALE), calibration_switches=list(SWITCH),
                quota=QUOTA, union_floor=.02, parameters=26208,
                design="Start all arms from the same frozen raw refinement on each of two group-disjoint splits. "
                       "Train equal-width heads on the same batches. Give the native-grid RGB nominee a chance to "
                       "replace the raw label when none of eight raw neighbors proposes its class and the matched "
                       "appearance view disagrees. Estimate repair, harm and neither with collapsed label likelihood. "
                       "Compare expected pixel correctness and first-order pooled class-IoU scores using the same "
                       "posterior. Measure raw, RGB, shifted and color attribution, the unrestricted dispute set, "
                       "and direct residuals. Choose confidence and residual scales on the specified calibration "
                       "recordings only. Audit the top 10% of identical eligible RGB positions for matched activity.",
                decision="Evaluate previously scored recordings with V77 weights locked before scoring. Evidence for "
                         "the extra mechanism needs gains over raw refinement, ordinary fusion and the important "
                         "matched controls, with the second recording split and external checks. Oracle uses labels "
                         "only to describe accessible corrections. No test score selects weights or thresholds.")


def iterator(split, part):
    if part == "external":
        yield from registered_cases(part)
    elif split == "refit":
        yield from ((f"muses_refit_{part}", key, group, "", x, y, image) for key, group, x, y, image in pairs(part))
    elif part in ("fit", "calibrate"):
        yield from ((f"muses_large_{part}", *row) for row in paths(part))
    else:
        yield from registered_cases(part)


def parents(split):
    if split == "large":
        low, strength = anchor_fit()
        strong = trained()
        return {s: (low["raw", s], strength, strong["raw_direct", s]) for s in SEEDS}
    check_data()
    bank = model_bank()
    return {s: (bank["raw", s], 1., bank["raw_refined", s]) for s in SEEDS}


def base_offset(features, frozen):
    first, scale, second = frozen
    return scale * first(features["raw"]) + second(features["raw_refined"])


def alternate_logits(features):
    context = features["raw_refined"][:, :19]
    alternate = (features[key][:, 51:70] + context for key in ("rgb_fusion", "color_match"))
    return tuple(4. * view for view in alternate)


def proposals(low, high, rgb, color):
    raw = high.argmax(1, keepdim=True)
    shape = high.shape[-2:]
    restored = resize(rgb, shape).argmax(1, keepdim=True)
    matched = resize(color, shape).argmax(1, keepdim=True)
    nearby = neighbors(low.argmax(1, keepdim=True), fill=255).squeeze(2)
    nearby = F.interpolate(nearby.float(), shape, mode="nearest-exact").long()
    disagreement = restored != raw
    add = disagreement & (restored != matched) & ~(restored == nearby).any(1, keepdim=True)
    color_add = (matched != raw) & (restored != matched) & ~(matched == nearby).any(1, keepdim=True)
    return raw, restored, matched, disagreement, add, color_add


def event_prob(q, raw, alternative):
    return q.gather(1, raw), q.gather(1, alternative)


def event_loss(posterior, truth, raw, alternative, eligible):
    logp = posterior.log_softmax(1)
    harm, repair = event_prob(logp, raw, alternative)
    neither = (1. - harm.exp() - repair.exp()).clamp_min(1e-7).log()
    observed = torch.where(truth == raw, harm, torch.where(truth == alternative, repair, neither))
    valid = eligible & (truth != 255)
    return -(observed * valid).sum() / valid.sum().clamp_min(1)


def gains(q, raw, next_class, prior, objective):
    keep, repair = event_prob(q, raw, next_class)
    if objective == "accuracy":
        return repair - keep
    inverse_union, iou = prior
    add = inverse_union[next_class] * ((1. + iou[next_class]) * repair - iou[next_class])
    leave = inverse_union[raw] * (iou[raw] - (1. + iou[raw]) * keep)
    return add + leave


def acceptance(gain, eligible, cutoff):
    return torch.zeros_like(eligible) if cutoff is None else eligible & (gain > cutoff)


def take_quota(gain, eligible):
    positions = torch.where(eligible.reshape(-1))[0]
    size = int(QUOTA * len(positions))
    selected = torch.topk(gain.reshape(-1)[positions], size).indices
    allow = eligible.new_zeros(eligible.numel())
    allow[positions[selected]] = True
    return allow.reshape(eligible.shape)


@torch.no_grad()
def pool(engine, split, smoke):
    records = defaultdict(list)
    hist = torch.zeros(19, dtype=torch.float64)
    keep = {"raw", "shift_heads", *METHODS.values(), "color_match"}
    for number, (_, _, group, _, image, truth, _) in enumerate(iterator(split, "fit"), 1):
        _, small, features, _, _ = engine(image)
        target = F.interpolate(truth[None, None].float(), small.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        valid = target != 255
        if not bool(valid.any()):
            raise ValueError("A fitting frame lost its semantic labels")
        hist += torch.bincount(target[valid].long(), minlength=19)
        features["shift_heads"] = displacement(features)
        records[group].append((small.half().cpu(), {k: features[k].half().cpu() for k in keep}, target))
        if number % 24 == 0:
            print(dict(split=split, fit_cached=number), flush=True)
        if smoke and number == 4:
            break
    actual = sum(len(x) for x in records.values())
    if actual != (4 if smoke else FIT_SIZE[split]):
        raise ValueError(f"Unexpected fit count {actual}")
    scales = ((hist.sum() / 19. + 1.) / (hist + 1.)).pow(.20).clamp(.65, 2.).cuda().float()
    return tuple(records.values()), scales


@torch.no_grad()
def anchored_data(records, frozen):
    output = []
    total = torch.zeros(19, 19, dtype=torch.int64)
    for group in records:
        ready = []
        for old, cues, truth in group:
            tensors = {k: cues[k].cuda().float() for k in ("raw", "raw_refined", "rgb_fusion", "color_match")}
            base = old.cuda().float() + base_offset(tensors, frozen)
            raw, rgb, color, all_edits, add, color_add = proposals(base, base, *alternate_logits(tensors))
            total += confusion_matrix(raw[0, 0], truth.cuda(), 19)
            packed = torch.cat((raw, rgb, color, all_edits, add, color_add), dim=1).byte().cpu()
            ready.append((base.half().cpu(), cues, packed, truth))
        output.append(ready)
    hit = total.diagonal()
    union = total.sum(0) + total.sum(1) - hit
    relative = (union / total.sum().clamp_min(1)).clamp_min(.02)
    return output, dict(iou=(hit / union.clamp_min(1)).tolist(), inverse_union=(.02 / relative).tolist(),
                        train_union=union.tolist())


def loss(mode, module, batch, weights):
    anchor = torch.cat([row.logits for row in batch])
    label = torch.stack([row.labels for row in batch])
    update = module(torch.cat([row.features for row in batch]))
    if mode in DIRECT:
        return segmentation_loss(anchor, update, label, weights)
    raw, rgb, color, all_edits, add, color_add = torch.cat([row.edits for row in batch]).split(1, dim=1)
    chosen = color if mode == "color_exclusive" else rgb
    eligible = {"color_exclusive": color_add, "head_all": all_edits}.get(mode, add).bool()
    posterior = anchor / 2. + update
    extra = F.cross_entropy(posterior, label, ignore_index=255)
    return event_loss(posterior, label[:, None], raw.long(), chosen.long(), eligible) + .20 * extra + .01 * update.square().mean()


def fit_model(root, mode, seed, records, weight, smoke):
    path, lock = root / f"{mode}-{seed}.pth", root / "protocol.json"
    log = path.with_suffix(".json")
    if not smoke and log.is_file():
        result = read(log)
        if result["weights_SHA256"] != digest(path) or result["protocol_SHA256"] != digest(lock):
            raise ValueError(f"The existing fit for {mode}/{seed} changed")
        return result
    set_seed(seed)
    model = GuidedResidual().cuda()
    optimiser = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, STEPS, eta_min=.00006)
    generator = torch.Generator().manual_seed(seed + 77003)
    steps = 3 if smoke else STEPS
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        batch = [patch(Frame(anchor, cue[METHODS[mode]], edit, target), generator)
                 for anchor, cue, edit, target in cases(records, generator)]
        objective = loss(mode, model, batch, weight)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        optimiser.step()
        scheduler.step()
    result = dict(method=mode, seed=seed, steps=steps, parameters=sum(p.numel() for p in model.parameters()),
                  loss=float(objective.detach()))
    if not smoke:
        torch_save(path, {key: value.detach().cpu() for key, value in model.state_dict().items()})
        result.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
        atomic_write(log, result)
    print(result, flush=True)
    return result


def fit(split, smoke):
    configure_cuda()
    root, note = ROOT / split, protocol(split)
    lock = root / "protocol.json"
    if not smoke:
        if lock.is_file() and read(lock) != note:
            raise ValueError(f"The registered training scope changed for {split}")
        atomic_write(lock, note)
    frozen = parents(split)
    data, weight = pool(RestorationFeatures(), split, smoke)
    results, stats = [], {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        prepared, stats[str(seed)] = anchored_data(data, frozen[seed])
        results.extend(fit_model(root, mode, seed, prepared, weight, smoke) for mode in METHODS)
    if not smoke:
        atomic_write(root / "completed.json", dict(runs=results, steps=sum(r["steps"] for r in results),
                     train_metrics=stats, protocol_SHA256=digest(lock), gpu=torch.cuda.get_device_name(),
                     torch=str(torch.__version__), finished_utc=now()))


def fits(split):
    root = ROOT / split
    lock, done = root / "protocol.json", read(root / "completed.json")
    if digest(lock) != done["protocol_SHA256"] or read(lock) != protocol(split):
        raise ValueError("The recorded proposal inputs changed")
    bank = {}
    for row in done["runs"]:
        key = row["method"], row["seed"]
        path = root / f"{key[0]}-{key[1]}.pth"
        if digest(path) != row["weights_SHA256"] or row["parameters"] != 26208 or row["steps"] != STEPS:
            raise ValueError(f"The fitted proposal head changed {key}")
        bank[key] = GuidedResidual().cuda().eval().requires_grad_(False)
        bank[key].load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if set(bank) != {(mode, seed) for mode in METHODS for seed in SEEDS}:
        raise ValueError(f"Missing curve controls for {split}")
    prior = {int(s): tuple(torch.tensor(v[name], device="cuda") for name in ("inverse_union", "iou"))
             for s, v in done["train_metrics"].items()}
    return bank, prior


def frame_options(engine, image, shape, frozen, bank, prior):
    dense, small, features, restored, matched = engine(image)
    features["shift_heads"] = displacement(features)
    rgb, color = alternate_logits(features)
    base = resize(dense, shape)
    for seed in SEEDS:
        delta = base_offset(features, frozen[seed])
        high = base + resize(delta, shape)
        raw, new, reference, disagreement, add, color_add = proposals(small + delta, high, rgb, color)
        direct = {mode: resize(bank[mode, seed](features[cue]), shape) for mode, cue in DIRECT.items()}
        vote = (high / 2.).softmax(1)
        averaging = .5 * (vote + (resize(rgb, shape) / 2.).softmax(1))
        scores = {"probability_average": (gains(averaging, raw, new, prior[seed], "accuracy"), new, add)}
        for mode, cue in RISK.items():
            posterior = (high / 2. + resize(bank[mode, seed](features[cue]), shape)).softmax(1)
            possible = {"head_all": disagreement, "color_exclusive": color_add}.get(mode, add)
            recipient = reference if mode == "color_exclusive" else new
            for stat in ("accuracy", "iou"):
                scores[mode + "_" + stat] = (gains(posterior, raw, recipient, prior[seed], stat), recipient, possible)
        yield seed, high, raw, direct, scores, restored, matched, (new, add, reference, color_add, disagreement)


def update_matrix(base, raw, truth, gain, target, eligible, cutoff):
    if cutoff is None:
        return base
    yes = acceptance(gain, eligible & (truth[None, None] != 255), cutoff)
    label = truth[None, None][yes] * 19
    loss = torch.bincount(label + raw[yes], minlength=361).reshape(19, 19).cpu()
    win = torch.bincount(label + target[yes], minlength=361).reshape(19, 19).cpu()
    return base - loss + win


def macro_score(counts):
    return 100. * mean(segmentation_metrics(cm)["miou"] for cm in counts)


@torch.inference_mode()
def calibrate(split):
    configure_cuda()
    root = ROOT / split
    bank, prior = fits(split)
    frozen, engine = parents(split), RestorationFeatures()
    sums = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64))
    frames = 0
    for _, _, _, _, image, target, _ in iterator(split, "calibrate"):
        for seed, high, raw, direct, scores, *_ in frame_options(engine, image, target.shape, frozen, bank, prior):
            original = confusion_matrix(raw[0, 0], target, 19)
            for mode, d in direct.items():
                for alpha in SCALE:
                    sums[mode, seed, alpha] += confusion_matrix((high + alpha * d).argmax(1)[0], target, 19)
            for mode, (value, label, eligible) in scores.items():
                for cutoff in SWITCH:
                    sums[mode, seed, cutoff] += update_matrix(original, raw, target, value, label, eligible, cutoff)
        frames += 1
    if frames != EVAL_SIZE[split]["calibrate"]:
        raise ValueError("The threshold calibration count changed")
    thresholds = {**dict.fromkeys(DIRECT, SCALE), **dict.fromkeys((*RISK_MODES, "probability_average"), SWITCH)}
    curves = {mode: [macro_score([sums[mode, seed, value] for seed in SEEDS]) for value in grid]
              for mode, grid in thresholds.items()}
    chosen = {mode: grid[max(range(len(grid)), key=lambda i: (curves[mode][i], -i))] for mode, grid in thresholds.items()}
    card = dict(frames=frames, chosen=chosen, scores=curves, grid={k: list(v) for k, v in thresholds.items()},
                weights_SHA256=digest(root / "completed.json"), finished_utc=now())
    atomic_write(root / "calibration.json", card)
    print(dict(calibration=split, choices=chosen), flush=True)


def labels_for(high, raw, direct, scores, choices, rgb_options, truth):
    new, exclusive, color, color_exclusive, disagreement = rgb_options
    out = {"raw_anchor": raw}
    for mode, offset in direct.items():
        out[mode] = (high + choices[mode] * offset).argmax(1, keepdim=True)
        if mode == "raw_residual":
            out["raw_residual_full"] = (high + offset).argmax(1, keepdim=True)
    for mode, (value, label, available) in scores.items():
        out[mode] = torch.where(acceptance(value, available, choices[mode]), label, raw)
        if mode in QUOTA_ARMS:
            out[mode + "_quota"] = torch.where(take_quota(value, available), label, raw)
        if mode == PRIMARY:
            out["head_exclusive_iou_zero"] = torch.where(acceptance(value, available, 0.), label, raw)
    reference = truth[None, None]
    for name, label, available in (("oracle_exclusive", new, exclusive), ("oracle_disagreement", new, disagreement),
                                    ("oracle_color_exclusive", color, color_exclusive)):
        out[name] = torch.where(available & (label == reference), label, raw)
    return out


def flux(raw, output, target):
    valid = target != 255
    changed = (raw != output) & valid
    return [int(x.sum()) for x in (changed, changed & (output == target), changed & (raw == target))]


@torch.inference_mode()
def evaluate(split, part):
    configure_cuda()
    root = ROOT / split
    bank, prior = fits(split)
    calibration = read(root / "calibration.json")
    if digest(root / "completed.json") != calibration["weights_SHA256"]:
        raise ValueError("The threshold calibration is stale")
    frozen, engine = parents(split), RestorationFeatures()
    rows, seen, panels = [], Counter(), {}
    for dataset, key, group, lighting, image, truth, photo in iterator(split, part):
        edge = truth.masked_fill(~boundary_band(truth), 255)
        options = frame_options(engine, image, truth.shape, frozen, bank, prior)
        for seed, high, raw, direct, scores, rgb, match, candidates in options:
            assembled = labels_for(high, raw, direct, scores, calibration["chosen"], candidates, truth)
            for mode, pred in assembled.items():
                value = pred[0, 0]
                rows.append(dict(dataset=dataset, id=key, group=group, method=mode, seed=seed,
                                 counts=confusion_matrix(value, truth, 19).tolist(),
                                 boundary=confusion_matrix(value, edge, 19).tolist(), flux=flux(raw[0, 0], value, truth)))
            panel_key = dataset, group
            if (seed == SEEDS[0] and panel_key not in panels and
                    sum(key[0] == dataset for key in panels) < (5 if part == "held" else 2)):
                names = ("raw_anchor", "raw_residual", PRIMARY, PRIMARY + "_quota")
                search = [blend(photo, assembled[n][0, 0].byte().cpu().numpy()) for n in names]
                panels[panel_key] = (f"{key} / {lighting or dataset}",
                    [photo, to_rgb(rgb), to_rgb(match), *search, paint(truth.byte().cpu().numpy())],
                    ("Fog", "Restored RGB", "Color matched", "Fixed raw", "More raw tuning", "Candidate", "Candidate quota", "Reference"))
        seen[dataset] += 1
        if sum(seen.values()) % 24 == 0:
            print(dict(split=split, stage=part, scored=sum(seen.values())), flush=True)
    if part == "held" and sum(seen.values()) != EVAL_SIZE[split]["held"]:
        raise ValueError("The held frames changed")
    if part == "external" and seen != dict(foggy_zurich_testv2=40, foggy_driving_fine=33):
        raise ValueError("The external frame count changed")
    main_controls = ("raw_anchor", *DIRECT, "raw_residual_full", *RISK_MODES, "probability_average", "head_exclusive_iou_zero")
    comparison = [(PRIMARY, k) for k in main_controls if k != PRIMARY]
    comparison.extend((PRIMARY + "_quota", k + "_quota") for k in QUOTA_ARMS if k != PRIMARY)
    paired = {field: paired_iou_intervals(rows, comparison, "method", field) for field in ("counts", "boundary")}
    metrics = {field: aggregate([{**r, "counts": r[field]} for r in rows]) for field in paired}
    totals = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        key = row["dataset"], row["method"]
        for k, value in enumerate(row["flux"]):
            totals[key][k] += value
    counts = {f"{d}/{m}": dict(zip(("edits", "repaired", "harmed"), values, strict=True)) for (d, m), values in totals.items()}
    data = root / f"{part}-individual.json.gz"
    image = root / f"{part}-predictions.png"
    compress_scores(data, rows)
    draw(image, list(panels.values()))
    sources = [ZURICH, DRIVING] if part == "external" else [HELD if split == "large" else DATA_REFIT / "held.json"]
    atomic_write(root / f"{part}-test.json", dict(counts=dict(seen), metrics=metrics, intervals=paired, flux=counts,
                 scores_SHA256=digest(data), evaluation_SHA256={str(p): digest(p) for p in sources},
                 picture_SHA256=digest(image), calibration_SHA256=digest(root / "calibration.json"), finished_utc=now()))
    notable = ("raw_anchor", "raw_residual", "rgb_residual", PRIMARY, "head_exclusive_accuracy", "oracle_exclusive")
    print({k: round(v["miou"], 4) for k, v in metrics["counts"].items() if k.split("/")[-1] in notable}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("smoke", "fit", "calibrate", "held", "external"))
    parser.add_argument("split", choices=tuple(FIT_SIZE))
    arg = parser.parse_args()
    if arg.part in ("fit", "smoke"):
        fit(arg.split, arg.part == "smoke")
    elif arg.part == "calibrate":
        calibrate(arg.split)
    else:
        evaluate(arg.split, arg.part)

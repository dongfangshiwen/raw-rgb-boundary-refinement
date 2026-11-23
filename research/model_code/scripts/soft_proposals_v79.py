"""Score all restoration disagreements with continuous neighborhood and tone evidence."""
import argparse
from collections import Counter, defaultdict
from math import log

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix
from dehaze_seg.utils import set_seed
from scripts.anchored_cue_v72 import GuidedResidual
from scripts.audit_dual_output_v66 import paired_iou_intervals
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, city_train_ids, resize_image, verified_image
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.parallel_utility_v32 import capture_group
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import Frame, aggregate, cases, patch
from scripts.restoration_affinity_v75 import boundary_band, compress_scores, neighbors
from scripts.restoration_proposals_v77 import (
    DIRECT, EVAL_SIZE, FIT_SIZE, ROOT as V77, SEEDS, STEPS, SWITCH, acceptance, alternate_logits,
    base_offset, event_loss, fits as previous, flux, gains, iterator, macro_score, parents, pool, proposals,
    take_quota, update_matrix,
)
from scripts.spatial_restoration_v73 import RestorationFeatures
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/soft-proposals-v79"
ARMS = ("soft", "rgb_pair", "no_color", "no_neighbor", "shifted", "raw_prior")
PRIMARY = "soft"
CHOICES = (*ARMS, "same_head_hard", "soft_iou", "probability_average")
QUOTAS = ("soft", "rgb_pair", "no_color", "no_neighbor", "shifted", "probability_average")
INHERITED = ("raw_anchor", *DIRECT, "v77_all", "v77_exclusive", "author_raw")
EXTERNAL = dict(foggy_zurich_testv2=40, foggy_driving_fine=33, foggy_driving_coarse=68)


def protocol(split):
    paths = [V77 / split / name for name in ("completed.json", "protocol.json", "calibration.json")]
    paths.extend((DRIVING, ZURICH))
    return dict(source_SHA256=digest(__file__), inputs={str(p): digest(p) for p in paths},
                split=split, methods=list(ARMS), seeds=list(SEEDS), steps=STEPS, params=26208,
                threshold_grid=list(SWITCH), primary=PRIMARY,
                experiment="Freeze the original and restored predictors. On every raw/restore disagreement, encode "
                           "the confidence in both options in the raw, restored and tone-matched views, plus "
                           "raw neighborhood mean, peak and vote. Fit one 26,208-parameter posterior head with "
                           "repair/harm/neither likelihood. Use an equal raw/restored probability prior. "
                           "Ablate the prior, color, neighborhood and spatial alignment with the same width. "
                           "Use the same head with the old exclusion rule as a mask-only comparison.",
                decision="Calibrate thresholds only on the designated recordings, use probability of repair minus "
                         "harm as primary and approximate mIoU influence as secondary. Compare at a fixed quota "
                         "as well. Re-score both previous MUSES splits, published LiDSeg and rescore Zurich and "
                         "the 101 official Foggy Driving frames, including the 68 coarse masks. Coarse boundaries "
                         "are excluded from boundary comparisons. Foggy Driving never enters V79 fitting or calibration. "
                         "Support requires gains over matched direct residuals, V77 and aligned cue controls." )


def log_prior(raw, restored):
    return torch.logaddexp((raw / 2.).log_softmax(1), (restored / 2.).log_softmax(1)) - log(2.)


def rolled(x):
    return x.roll(tuple(n // 2 for n in x.shape[-2:]), dims=(-2, -1))


def paired_features(low, rgb, color, features):
    p, r, c = ((value / 2.).softmax(1) for value in (low, rgb, color))
    source, target = p.argmax(1, keepdim=True), r.argmax(1, keepdim=True)
    votes = F.one_hot(source[:, 0], 19).permute(0, 3, 1, 2).to(p.dtype)
    norm = neighbors(torch.ones_like(p[:, :1])).sum(1).clamp_min(1.)
    close = neighbors(p)
    n = (close.sum(1) / norm, close.amax(1), neighbors(votes).sum(1) / norm)
    common = features["raw_refined"][:, :51]
    difference = features["rgb_fusion"][:, 51:70]
    label = F.one_hot(target[:, 0], 19).permute(0, 3, 1, 2).to(p.dtype)
    peaks = r.topk(2, dim=1).values
    margin = peaks[:, :1] - peaks[:, 1:2]
    configs = dict(soft=(c, n), rgb_pair=(p, (p, p, p)), no_color=(p, n), no_neighbor=(c, (p, p, p)),
                   shifted=(rolled(c), tuple(rolled(v) for v in n)))
    cues = {}
    for name, (tone, nearby) in configs.items():
        measurements = (p, r, tone, *nearby)
        summary = torch.cat([m.gather(1, k) for k in (source, target) for m in measurements], dim=1)
        cues[name] = torch.cat((common, difference, label, 2. * summary, 2. * margin), dim=1)
    return cues, source, target


@torch.no_grad()
def prepare(data, frozen):
    output = []
    for group in data:
        ready = []
        for small, cues, label in group:
            pack = {k: cues[k].cuda().float() for k in ("raw", "raw_refined", "rgb_fusion", "color_match")}
            low = small.cuda().float() + base_offset(pack, frozen)
            rgb, color = alternate_logits(pack)
            views, a, b = paired_features(low, rgb, color, pack)
            ready.append((log_prior(low, rgb).half().cpu(), (low / 2.).log_softmax(1).half().cpu(),
                          {k: v.half().cpu() for k, v in views.items()}, torch.cat((a, b), dim=1).byte().cpu(), label))
        output.append(ready)
    return output


def fit_one(root, mode, seed, data, smoke):
    path, lock = root / f"{mode}-{seed}.pth", root / "protocol.json"
    audit = path.with_suffix(".json")
    if not smoke and audit.is_file():
        state = read(audit)
        if state["weights_SHA256"] != digest(path) or state["protocol_SHA256"] != digest(lock):
            raise ValueError(f"The saved soft proposal fit differs {mode}/{seed}")
        return state
    set_seed(seed)
    model = GuidedResidual().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00006)
    rng = torch.Generator().manual_seed(seed + 77003)
    iterations = 3 if smoke else STEPS
    for _ in range(iterations):
        opt.zero_grad(set_to_none=True)
        batch = [patch(Frame(old if mode == "raw_prior" else prior, features["soft" if mode == "raw_prior" else mode], edit, y), rng)
                 for prior, old, features, edit, y in cases(data, rng)]
        logits = torch.cat([r.logits for r in batch])
        update = model(torch.cat([r.features for r in batch]))
        a, b = torch.cat([r.edits for r in batch]).long().split(1, dim=1)
        targets = torch.stack([r.labels for r in batch])
        posterior = logits + update
        loss = (event_loss(posterior, targets[:, None], a, b, a != b) + .20 * F.cross_entropy(posterior, targets, ignore_index=255)
                + .01 * update.square().mean())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    result = dict(method=mode, seed=seed, steps=iterations, params=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        result.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
        atomic_write(audit, result)
    print(result, flush=True)
    return result


def fit(split, smoke):
    configure_cuda()
    root = ROOT / split
    lock, card = root / "protocol.json", protocol(split)
    if not smoke:
        if lock.is_file() and read(lock) != card:
            raise ValueError("The registered soft proposal study changed")
        atomic_write(lock, card)
    frozen = parents(split)
    data, _ = pool(RestorationFeatures(), split, smoke)
    if sum(map(len, data)) != (4 if smoke else FIT_SIZE[split]):
        raise ValueError("The precomputed soft proposal frames differ from registration")
    runs = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        prepared = prepare(data, frozen[seed])
        runs.extend(fit_one(root, m, seed, prepared, smoke) for m in ARMS)
    if not smoke:
        atomic_write(root / "completed.json", dict(runs=runs, steps=sum(x["steps"] for x in runs),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def trained(split):
    root = ROOT / split
    card, lock = read(root / "completed.json"), root / "protocol.json"
    if digest(lock) != card["protocol_SHA256"] or read(lock) != protocol(split):
        raise ValueError("The locked model scope changed")
    models = {}
    for row in card["runs"]:
        path = root / f'{row["method"]}-{row["seed"]}.pth'
        if digest(path) != row["weights_SHA256"] or row["params"] != 26208 or row["steps"] != STEPS:
            raise ValueError(f"A soft proposal head differs {path.name}")
        model = GuidedResidual().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        models[row["method"], row["seed"]] = model
    if set(models) != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("The paired controls are incomplete")
    return models


def evaluation_cases(split, part):
    yield from iterator(split, part)
    if part != "external":
        return
    index = read(DRIVING)
    coarse = [r for r in index["rows"] if r["annotation_kind"] == "gtCoarse"]
    if not index["original_GT_unchanged"] or len(coarse) != 68:
        raise ValueError("The official coarse Foggy Driving records changed")
    for record in coarse:
        key = record["id"]
        photo = verified_image(record["rgb"], record["rgb_SHA256"], False)
        truth = city_train_ids(verified_image(record["annotation"], record["mask_SHA256"], True))
        count = np.bincount(truth[truth < 19], minlength=19)
        if photo.shape[:2] != truth.shape or count.tolist() != record["counts_19"]:
            raise ValueError(f"The coarse label-ID mapping or geometry differs {key}")
        x, _ = resize_image(photo)
        yield "foggy_driving_coarse", key, capture_group(key), "", x, torch.from_numpy(truth.copy()).cuda().long(), photo


class Evaluator:
    def __init__(self, split):
        self.engine = RestorationFeatures()
        self.frozen, self.models = parents(split), trained(split)
        self.controls, self.stats = previous(split)
        self.old = read(V77 / split / "calibration.json")["chosen"]

    def options(self, image, shape, with_controls):
        dense, small, bank, restored, tone = self.engine(image)
        coarse_rgb, coarse_tone = alternate_logits(bank)
        base = resize(dense, shape)
        rgb = resize(coarse_rgb, shape)
        for seed in SEEDS:
            offset = base_offset(bank, self.frozen[seed])
            low, high = small + offset, base + resize(offset, shape)
            a, b, _, wide, hard, _ = proposals(low, high, coarse_rgb, coarse_tone)
            cues, _, _ = paired_features(low, coarse_rgb, coarse_tone, bank)
            prior = log_prior(high, rgb)
            earlier = (high / 2.).log_softmax(1)
            stats = self.stats[seed]
            scores = {"probability_average": (gains(prior.exp(), a, b, stats, "accuracy"), wide)}
            for mode in ARMS:
                source = "soft" if mode == "raw_prior" else mode
                start = earlier if mode == "raw_prior" else prior
                q = (start + resize(self.models[mode, seed](cues[source]), shape)).softmax(1)
                value = gains(q, a, b, stats, "accuracy")
                scores[mode] = (value, wide)
                if mode == PRIMARY:
                    scores["same_head_hard"] = (value, hard)
                    scores["soft_iou"] = (gains(q, a, b, stats, "iou"), wide)
            direct = {"raw_anchor": a, "author_raw": base.argmax(1, keepdim=True)}
            if with_controls:
                for name, cue in DIRECT.items():
                    d = resize(self.controls[name, seed](bank[cue]), shape)
                    direct[name] = (high + self.old[name] * d).argmax(1, keepdim=True)
                for tag, mode, stat, allowed in (("v77_all", "head_all", "accuracy", wide),
                                                  ("v77_exclusive", "head_exclusive", "iou", hard)):
                    x = (earlier + resize(self.controls[mode, seed](bank["correction_heads"]), shape)).softmax(1)
                    pick = acceptance(gains(x, a, b, stats, stat), allowed, self.old[f"{mode}_{stat}"])
                    direct[tag] = torch.where(pick, b, a)
            yield seed, a, b, scores, direct, restored, tone


@torch.inference_mode()
def calibrate(split):
    configure_cuda()
    runner = Evaluator(split)
    sums, frames = defaultdict(lambda: torch.zeros(19, 19, dtype=torch.int64)), 0
    for _, _, _, _, image, y, _ in iterator(split, "calibrate"):
        for seed, a, b, values, *_ in runner.options(image, y.shape, False):
            old = confusion_matrix(a[0, 0], y, 19)
            for mode, (score, allowed) in values.items():
                for cutoff in SWITCH:
                    sums[mode, seed, cutoff] += update_matrix(old, a, y, score, b, allowed, cutoff)
        frames += 1
    if frames != EVAL_SIZE[split]["calibrate"]:
        raise ValueError("Unexpected calibration frames")
    curves = {m: [macro_score([sums[m, seed, t] for seed in SEEDS]) for t in SWITCH] for m in CHOICES}
    chosen = {m: SWITCH[max(range(len(SWITCH)), key=lambda k: (curve[k], -k))] for m, curve in curves.items()}
    root = ROOT / split
    atomic_write(root / "calibration.json", dict(frames=frames, thresholds=list(SWITCH), chosen=chosen, mIoU=curves,
                 weights_SHA256=digest(root / "completed.json"), finished_utc=now()))
    print(dict(calibrated=split, chosen=chosen), flush=True)


@torch.inference_mode()
def evaluate(split, part):
    configure_cuda()
    root = ROOT / split
    prior = read(root / "calibration.json")
    if prior["weights_SHA256"] != digest(root / "completed.json"):
        raise ValueError("Thresholds are stale for the loaded checkpoint")
    runner = Evaluator(split)
    rows, counts, panels = [], Counter(), {}
    for dataset, key, group, light, x, truth, photo in evaluation_cases(split, part):
        band = None if dataset == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        for seed, a, b, options, direct, rgb, tone in runner.options(x, truth.shape, True):
            predictions = dict(direct)
            for mode, (gain, available) in options.items():
                predictions[mode] = torch.where(acceptance(gain, available, prior["chosen"][mode]), b, a)
                if mode in QUOTAS:
                    predictions[mode + "_quota"] = torch.where(take_quota(gain, available), b, a)
            value, allowed = options[PRIMARY]
            predictions["soft_zero"] = torch.where(acceptance(value, allowed, 0.), b, a)
            predictions["oracle"] = torch.where(allowed & (b == truth[None, None]), b, a)
            for mode, pred in predictions.items():
                p = pred[0, 0]
                row = dict(dataset=dataset, id=key, group=group, method=mode, seed=seed,
                           counts=confusion_matrix(p, truth, 19).tolist(), flux=flux(a[0, 0], p, truth))
                if band is not None:
                    row["boundary"] = confusion_matrix(p, band, 19).tolist()
                rows.append(row)
            g = dataset, group
            if seed == SEEDS[0] and g not in panels and sum(z[0] == dataset for z in panels) < (5 if part == "held" else 2):
                shown = [blend(photo, predictions[m][0, 0].byte().cpu().numpy()) for m in ("raw_residual", "rgb_residual", PRIMARY, "same_head_hard")]
                panels[g] = (f"{key} / {light or dataset}", [photo, to_rgb(rgb), to_rgb(tone), *shown, paint(truth.byte().cpu().numpy())],
                             ("Fog", "Restored RGB", "Tone matched", "Raw residual", "RGB fusion", "Soft acceptance", "Hard acceptance", "Reference"))
        counts[dataset] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(split=split, stage=part, scored=sum(counts.values())), flush=True)
    if (part == "held" and sum(counts.values()) != EVAL_SIZE[split]["held"]) or (part == "external" and counts != EXTERNAL):
        raise ValueError("The radiometric proposal evaluation membership changed")
    comparisons = [(PRIMARY, m) for m in (*INHERITED, *CHOICES, "soft_zero") if m != PRIMARY]
    comparisons += [("soft_quota", f"{m}_quota") for m in QUOTAS if m != PRIMARY]
    main = rows + [{**row, "dataset": "foggy_driving_101"} for row in rows if row["dataset"].startswith("foggy_driving_")]
    boundary = [{**row, "counts": row["boundary"]} for row in rows if "boundary" in row]
    by_region = dict(counts=main, boundary=boundary)
    scores = {k: aggregate(v) for k, v in by_region.items()}
    intervals = {k: paired_iou_intervals(v, comparisons, "method", "counts") for k, v in by_region.items()}
    totals = defaultdict(lambda: [0, 0, 0])
    for row in main:
        key = f'{row["dataset"]}/{row["method"]}'
        for k, value in enumerate(row["flux"]):
            totals[key][k] += value
    ledger, picture = root / f"{part}-individual.json.gz", root / f"{part}-predictions.png"
    compress_scores(ledger, rows)
    draw(picture, list(panels.values()))
    atomic_write(root / f"{part}-test.json", dict(counts=dict(counts), metrics=scores, intervals=intervals, flux=dict(totals),
                 ledger_SHA256=digest(ledger), picture_SHA256=digest(picture), calibration_SHA256=digest(root / "calibration.json"),
                 finished_utc=now()))
    print({k: round(v["miou"], 4) for k, v in scores["counts"].items() if k.split("/")[-1] in
           ("raw_anchor", "raw_residual", "rgb_residual", "head_residual", PRIMARY, "same_head_hard", "oracle")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("smoke", "fit", "calibrate", "held", "external"))
    parser.add_argument("split", choices=tuple(FIT_SIZE))
    settings = parser.parse_args()
    if settings.part in ("smoke", "fit"):
        fit(settings.split, settings.part == "smoke")
    elif settings.part == "calibrate":
        calibrate(settings.split)
    else:
        evaluate(settings.split, settings.part)

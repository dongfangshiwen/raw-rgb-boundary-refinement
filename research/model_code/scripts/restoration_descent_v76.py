"""Predict semantic descent along anchored neighborhood proposals with a curvature bound."""
import argparse
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix
from dehaze_seg.utils import set_seed
from scripts.anchored_cue_v72 import GuidedResidual, anchor_fit, anchored_records
from scripts.audit_dual_output_v66 import paired_iou_intervals
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import DATA, SEEDS, Frame, aggregate, cases, patch
from scripts.restoration_affinity_v75 import (
    HELD, ROOT as AFFINITY, STEPS, boundary_band, compress_scores, displacement, feature_pool, neighbors,
    registered_cases, segmentation_loss,
)
from scripts.spatial_restoration_v73 import RestorationFeatures
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/restoration-descent-v76"
DIRECT = {"raw_direct": "raw_refined", "rgb_direct": "rgb_fusion", "head_direct": "correction_heads"}
DESCENT = {"raw_descent": "raw_refined", "rgb_descent": "rgb_fusion", "color_descent": "color_match",
           "head_descent": "correction_heads", "shift_descent": "shift_heads"}
ARMS = {**DIRECT, **DESCENT}
PRIMARY = "head_descent"


class Proposals:
    def __init__(self, logits):
        centered = logits - logits.mean(1, keepdim=True)
        exists = neighbors(torch.ones_like(centered[:, :1]))
        self.delta = (neighbors(centered) - centered[:, None]) * exists
        self.base = logits.softmax(1)
        # CE has second derivative Var_p(delta). The squared range divided by four bounds it everywhere.
        spread = self.delta.amax(2, keepdim=True) - self.delta.amin(2, keepdim=True)
        self.curvature = (.25 * spread.square()).clamp_min(.001)

    def update(self, target, fixed=None):
        utility = ((target - self.base)[:, None] * self.delta).sum(2, keepdim=True)
        step = (utility / self.curvature).clamp(0., 1.) if fixed is None else fixed
        gain = (step * (utility - .5 * self.curvature * step)).clamp_min(0.)
        weights = gain / (.025 + gain.sum(1, keepdim=True))
        return (weights * step * self.delta).sum(1)


def protocol():
    return dict(code_SHA256=digest(__file__), affinity_protocol_SHA256=digest(AFFINITY / "protocol.json"),
                affinity_code_SHA256=digest(REPO / "scripts/restoration_affinity_v75.py"), seeds=list(SEEDS),
                fit_data_SHA256=digest(DATA / "fit.json"), steps=STEPS, methods=ARMS, parameters=26208,
                objective="At each pixel, use a predicted class posterior to estimate the initial decrease in CE "
                          "along eight centered raw-neighbor proposals. Bound the curvature by the squared logit "
                          "range divided by four. Compute each bounded step analytically and mix proposals using "
                          "their estimated decrease. Train with the same CE, harm and posterior losses for every arm. "
                          "Direct arms use the posterior as unrestricted class offsets. The fixed-step diagnostic "
                          "reuses head-descent weights. True-label proposals are an oracle diagnostic only.",
                decision="Start from the same raw anchor, initialization, samples, parameters and steps in every arm. "
                         "Use the V75 training population and fixed test memberships. The 24-image calibration subset "
                         "motivated testing anticipated loss decrease after same-class affinity reduced mIoU. Require "
                         "gains over raw, restored RGB, color, shift and unrestricted head controls in the fresh frames "
                         "and at least one external benchmark. No held labels enter fitted prediction or model choice.")


def file(mode, seed):
    return ROOT / f"{mode}-{seed}.pth"


def train(mode, seed, groups, weight, smoke):
    weights_path = file(mode, seed)
    log, lock = weights_path.with_suffix(".json"), ROOT / "protocol.json"
    if not smoke and log.is_file():
        old = read(log)
        if digest(weights_path) != old["weights_SHA256"] or digest(lock) != old["protocol_SHA256"]:
            raise ValueError("The recorded descent model has different provenance")
        return old
    set_seed(seed)
    head = GuidedResidual().cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.0005, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00006)
    rng = torch.Generator().manual_seed(seed + 75003)
    iterations = 3 if smoke else STEPS
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        batch = [patch(Frame(p, z[ARMS[mode]], z[ARMS[mode]][:, :0], y), rng) for p, z, y in cases(groups, rng)]
        logits = torch.cat([r.logits for r in batch])
        labels = torch.stack([r.labels for r in batch])
        offset = head(torch.cat([r.features for r in batch]))
        prediction = logits + offset
        update = offset if mode in DIRECT else Proposals(logits).update(prediction.softmax(1))
        posterior = F.cross_entropy(prediction, labels, weight=weight, ignore_index=255)
        loss = segmentation_loss(logits, update, labels, weight) + .40 * posterior
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 2.5, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
    result = dict(method=mode, seed=seed, steps=iterations, loss=float(loss.detach()),
                  parameters=sum(p.numel() for p in head.parameters()))
    if not smoke:
        torch_save(weights_path, {k: p.detach().cpu() for k, p in head.state_dict().items()})
        result.update(weights_SHA256=digest(weights_path), protocol_SHA256=digest(lock))
        atomic_write(log, result)
    print(result, flush=True)
    return result


def fit(smoke):
    configure_cuda()
    if not smoke:
        lock, plan = ROOT / "protocol.json", protocol()
        if lock.exists() and read(lock) != plan:
            raise ValueError("The descent protocol changed")
        atomic_write(lock, plan)
    frozen, strength = anchor_fit()
    pool, weight = feature_pool(RestorationFeatures(), smoke)
    cards = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        base = anchored_records(pool, frozen["raw", seed], strength)
        cards.extend(train(mode, seed, base, weight, smoke) for mode in ARMS)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=cards, steps=sum(r["steps"] for r in cards),
                     gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                     protocol_SHA256=digest(lock), finished_utc=now()))


def trained():
    card, lock = read(ROOT / "completed.json"), ROOT / "protocol.json"
    if card["protocol_SHA256"] != digest(lock) or read(lock) != protocol():
        raise ValueError("The recorded descent fits do not match their protocol")
    models = {}
    for row in card["runs"]:
        key = row["method"], row["seed"]
        path = file(*key)
        if digest(path) != row["weights_SHA256"] or row["parameters"] != 26208 or row["steps"] != STEPS:
            raise ValueError(f"Invalid trained descent fit: {key}")
        models[key] = GuidedResidual().cuda().eval().requires_grad_(False)
        models[key].load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if set(models) != {(mode, seed) for mode in ARMS for seed in SEEDS}:
        raise ValueError("Missing descent comparisons")
    return models


def oracle_update(proposal, truth, size):
    labels = F.interpolate(truth[None, None].float(), size, mode="nearest-exact").long()
    known = F.one_hot(labels[:, 0].clamp_max(18), 19).permute(0, 3, 1, 2).to(proposal.base.dtype)
    ideal = torch.where(labels == 255, proposal.base, known)
    return proposal.update(ideal)


@torch.inference_mode()
def score(stage):
    configure_cuda()
    engine, models = RestorationFeatures(), trained()
    anchor, strength = anchor_fit()
    rows, panels, counts = [], {}, Counter()
    for dataset, key, group, lighting, image, truth, original in registered_cases(stage):
        dense, small, cues, restored, _ = engine(image)
        cues["shift_heads"] = displacement(cues)
        whole = resize(dense, truth.shape)
        region = boundary_band(truth)
        boundary = truth.masked_fill(~region, 255)
        interior = truth.masked_fill(region, 255)
        outputs = {}
        for seed in SEEDS:
            residual = strength * anchor["raw", seed](cues["raw"])
            low, high = small + residual, whole + resize(residual, truth.shape)
            proposal = Proposals(low)
            outputs["raw", seed] = high.argmax(1)[0]
            for mode, feature in ARMS.items():
                offset = models[mode, seed](cues[feature])
                if mode in DESCENT:
                    target = (low + offset).softmax(1)
                    offset = proposal.update(target)
                    if mode == PRIMARY:
                        fixed = resize(proposal.update(target, fixed=.50), truth.shape)
                        outputs["head_fixed_step", seed] = (high + fixed).argmax(1)[0]
                outputs[mode, seed] = (high + resize(offset, truth.shape)).argmax(1)[0]
            oracle = oracle_update(proposal, truth, low.shape[-2:])
            outputs["label_oracle", seed] = (high + resize(oracle, truth.shape)).argmax(1)[0]
        for (mode, seed), mask in outputs.items():
            values = [confusion_matrix(mask, y, 19).tolist() for y in (truth, boundary, interior)]
            rows.append(dict(dataset=dataset, group=group, id=key, seed=seed, method=mode,
                             counts=values[0], boundary=values[1], interior=values[2]))
        cap = 5 if stage == "held" else 2
        group_id = dataset, group
        if group_id not in panels and sum(ds == dataset for ds, _ in panels) < cap:
            shown = [blend(original, outputs[m, SEEDS[0]].byte().cpu().numpy())
                     for m in ("raw_direct", "head_direct", "raw_descent", PRIMARY, "shift_descent")]
            panels[group_id] = (f"{key} / {lighting or dataset}", [original, to_rgb(restored), *shown,
                                paint(truth.byte().cpu().numpy())],
                                ("Fog", "Restored RGB", "Raw direct", "Feature direct", "Raw descent",
                                 "Restoration descent", "Shifted descent", "Reference"))
        counts[dataset] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(stage=stage, scored=sum(counts.values())), flush=True)
    source = read(AFFINITY / f"{stage}-test.json")
    if counts != source["counts"] or len({row["id"] for row in rows}) != sum(counts.values()):
        raise ValueError(f"The recorded evaluation cohort has changed: {counts}")
    controls = [(PRIMARY, m) for m in ("raw", *ARMS, "head_fixed_step") if m != PRIMARY]
    fields = ("counts", "boundary", "interior")
    intervals = {region: paired_iou_intervals(rows, controls, "method", region) for region in fields}
    metrics = {region: aggregate([{**row, "counts": row[region]} for row in rows]) for region in fields}
    matrix, figure = ROOT / f"{stage}-individual.json.gz", ROOT / f"{stage}-predictions.png"
    compress_scores(matrix, rows)
    draw(figure, list(panels.values()))
    atomic_write(ROOT / f"{stage}-test.json", dict(counts=dict(counts), metrics=metrics, intervals=intervals,
                 matching_cohort_SHA256=digest(AFFINITY / f"{stage}-test.json"), scores_SHA256=digest(matrix),
                 picture_SHA256=digest(figure), weights_SHA256=digest(ROOT / "completed.json"),
                 data_SHA256=digest(HELD) if stage == "held" else None, finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in metrics["counts"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "held", "external", "development", "calibrate"))
    stage = parser.parse_args().stage
    if stage in ("smoke", "fit"):
        fit(stage == "smoke")
    else:
        score(stage)

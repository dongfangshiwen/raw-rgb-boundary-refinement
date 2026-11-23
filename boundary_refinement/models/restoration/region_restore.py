"""Fit coarse RGB correction with color subspace, region and paired-view controls."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import defaultdict
from math import sqrt
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.training.crop_context import originals
from boundary_refinement.evaluation.photographic.evaluate_author_mct import load_author_reference
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_parallel_rgb import bind
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.source_replay import PROTOCOL as RW_FIT
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD
from boundary_refinement.training.structure_transfer import SEEDS, training_scenes
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.paired.veil_response import NH_ROOT, nh


ROOT = REPO / "results/region-restore-v48"
MODES = ("direct", "global", "scalar", "split", "rotated", "color", "semantic", "shifted", "paired", "cross_scene")
STEPS, GRID = 480, 48
FLOOR_DB, MIN_GAIN = -.25, .10
SCALES = (0., .25, .5, .75, 1., 1.25)


def view_size(x, maximum, minimum=32):
    shape = x.shape[-2:]
    scale = min(1., maximum / max(shape))
    return tuple(max(minimum, round(scale * n)) for n in shape)


def color_assignments(value):
    """Deterministic same-cardinality photometric grouping with soft RGB assignments."""
    flat = value.flatten(2).transpose(1, 2)
    rank = flat.mean(-1).argsort(dim=1, stable=True)
    points = torch.linspace(.025, .975, 19, device=value.device)
    positions = rank[:, (points * (flat.shape[1] - 1)).round().long()]
    centers = flat.gather(1, positions[..., None].expand(-1, -1, 3))
    for _ in range(4):
        square = (flat[:, :, None] - centers[:, None]).square().sum(-1)
        assignment = (-30. * square).softmax(-1)
        weights = assignment.transpose(1, 2)
        centers = (weights @ flat) / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    square = (flat[:, :, None] - centers[:, None]).square().sum(-1)
    return (-30. * square).softmax(-1).transpose(1, 2).reshape(len(value), 19, *value.shape[-2:])


@torch.no_grad()
def prepare(raw, base, second, semantics):
    """Prepare target-free image evidence on the grid used by the correction head."""
    x, b, m = (F.adaptive_avg_pool2d(v, view_size(raw, GRID, 8)) for v in (raw, base, second))
    var = (F.adaptive_avg_pool2d(raw.square(), x.shape[-2:]) - x.square()).clamp_min(0.)
    cues = torch.cat((2. * x - 1., 2. * b - 1., 2. * m - 1., 4. * var.sqrt()), dim=1)
    model, mean_rgb, std_rgb = semantics
    preview = resize(raw, view_size(raw, 416), antialias=True)
    scores = model(pixel_values=(preview - mean_rgb) / std_rgb).logits.float()
    probs = F.adaptive_avg_pool2d(scores.softmax(1), x.shape[-2:])
    return cues, b, m, color_assignments(x), probs


def share_by_region(coef, assignment):
    membership = assignment.flatten(2)
    values = coef.flatten(2)
    average = (membership @ values.transpose(1, 2)) / membership.sum(2, keepdim=True).clamp_min(1e-6)
    pooled = (average.transpose(1, 2) @ membership).reshape_as(coef)
    return .50 * (coef + pooled)


def separate_colors(color, rotated=False):
    xy = (1., -1., 1.) if rotated else (1., 1., 1.)
    axis = color.new_tensor(xy).view(1, 3, 1, 1) / sqrt(3.)
    achromatic = (color * axis).sum(dim=1, keepdim=True) * axis
    return achromatic, color - achromatic


class LowField(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(nn.Conv2d(12, 28, 3, padding=1), nn.GroupNorm(4, 28), nn.SiLU())
        self.spatial = nn.Sequential(nn.Conv2d(56, 28, 3, padding=2, dilation=2), nn.GroupNorm(4, 28),
                                     nn.SiLU(), nn.Conv2d(28, 3, 1))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)

    def forward(self, batch, mode):
        inputs, base, second, color, semantic = batch
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=inputs.is_cuda):
            local = self.first(inputs)
            context = local.mean((-1, -2), keepdim=True).expand_as(local)
            code = self.spatial(torch.cat((local, context), dim=1)).float()
        if mode == "direct":
            return .18 * code.tanh()
        direction = second - base
        if mode in ("scalar", "global"):
            gate = (code.mean(1, keepdim=True) - 2.).sigmoid()
            if mode == "global":
                gate = gate.mean((-1, -2), keepdim=True)
            return gate * direction
        coef = (.5 * (code[:, :2] + code[:, 2:]) - 2.).sigmoid()
        if mode == "color":
            coef = share_by_region(coef, color)
        elif mode in ("semantic", "shifted", "paired", "cross_scene"):
            if mode == "shifted":
                semantic = semantic.roll(tuple(n // 2 for n in semantic.shape[-2:]), dims=(-2, -1))
            coef = share_by_region(coef, semantic)
        luma, chroma = separate_colors(direction, mode == "rotated")
        return coef[:, :1] * luma + coef[:, 1:2] * chroma


def load_anchors():
    rgb, rgb_hash = rgb_checkpoint()
    bind()
    second, second_card = load_author_reference()
    model, mean_rgb, std_rgb, seg_card = load_segmenter()
    return rgb, second, (model, mean_rgb, std_rgb), dict(pgl_SHA256=rgb_hash, mct=second_card, segmentation=seg_card)


@torch.no_grad()
def inputs(fog, clear, anchors):
    rgb, second, semantics = anchors
    pair = torch.stack((fog, clear)).cuda().float() / 255.
    a, y = resize(pair, view_size(pair, 768), antialias=True).split(1)
    base, mct = pgl_rgb(rgb, a).clamp(0, 1), author_rgb(second, a).clamp(0, 1)
    features = prepare(a, base, mct, semantics)
    truth = F.adaptive_avg_pool2d(y, features[1].shape[-2:])
    return features, truth


@torch.no_grad()
def training_bank(anchors, smoke):
    repeated = training_scenes()
    lm = [tuple(inputs(r["frames"][j], r["clear"], anchors) for j in (1, 3, 5, 7))
          for r in (repeated[:3] if smoke else repeated)]
    source = originals()
    rw = [inputs(*pair, anchors) for pair in (source[:3] if smoke else source)]
    nh_pairs = [(a, b) for _, _, _, a, b in nh("fit")]
    nhaze = [inputs(*pair, anchors) for pair in (nh_pairs[:3] if smoke else nh_pairs)]
    actual = [len(lm), sum(map(len, lm)), len(rw), len(nhaze)]
    if actual != ([3, 12, 3, 3] if smoke else [32, 128, 86, 36]):
        raise ValueError(f"Unexpected RGB fit population {actual}")
    print(dict(lmhaze_scenes=actual[0], lmhaze_views=actual[1], rw=actual[2], nh=actual[3]), flush=True)
    return lm, rw, nhaze


def relative_loss(network, item, mode):
    features, target = item
    base = features[1]
    delta = network(features, mode)
    error = (base + delta).clamp(0., 1.) - target
    normal = (base - target).square().mean().clamp_min(.0025)
    return (error.square().mean() / normal + .25).log(), error


def draw_examples(banks, rng):
    lm, rw, nhaze = banks
    groups = torch.randperm(len(lm), generator=rng)[:2].tolist()
    paired = [lm[i][j] for i in groups for j in torch.randperm(len(lm[i]), generator=rng)[:2].tolist()]
    single = [items[int(torch.randint(len(items), (), generator=rng))] for items in (rw, nhaze)]
    return paired, single


def paired_penalty(errors, wrong):
    # Valid same-reference residuals need no geometrical warping. Resize only to the
    # same descriptor size so the cross-scene control is treated identically.
    vectors = [F.adaptive_avg_pool2d(field, (8, 12)) for field in errors]
    partners = ((0, 3), (2, 1)) if wrong else ((0, 1), (2, 3))
    # A common smooth L1 avoids penalizing small JPEG residuals disproportionately.
    return sum(F.smooth_l1_loss(vectors[a], vectors[b], beta=.10) for a, b in partners) / 2.


def fit(smoke=False):
    configure_cuda()
    rgb, mct, semantics, anchors = load_anchors()
    capacity = sum(p.numel() for p in LowField().parameters())
    plan = dict(code_SHA256=digest(__file__), modes=list(MODES), seeds=list(SEEDS), steps=STEPS,
        trainable=capacity, anchor=anchors, grid=GRID, data={str(p): digest(p) for p in (TRAIN_CARD, RW_FIT, NH_ROOT / "data_manifest.json")},
        training="Four LMHaze views from two distinct scenes per update plus one RW2AH and one NH-HAZE pair. Fixed augmentation-free samples, equal compute schedules. Frozen backbones. Each row supervises coarse RGB means only.",
        mechanism="Coarse residuals compare unconstrained RGB to shared scalar and separated achromatic/opponent mixing of the native PGL and MCT predictions. Region variants average the coefficients through color assignments, SegFormer probabilities or spatially rolled SegFormer probabilities. The paired penalty uses aligned same-scene RGB fitting residuals; the cross_scene control compares the residuals from mismatched scenes with equal count and weight.",
        selection=dict(amplitudes=list(SCALES), rw_floor_db=FLOOR_DB, useful_gain_db=MIN_GAIN,
            scope="Calibrate only on 64 LMHaze, 47 RW2AH and 9 NH-HAZE pairs. Rank by equal mean paired gain on LMHaze and NH-HAZE, subject to the RW floor. REVIDE does not enter calibration or fitting."),
    )
    lock = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if lock.exists() and read(lock) != plan:
            raise ValueError("The registered coarse-restoration experiment changed")
        if not lock.exists():
            atomic_write(lock, plan)
    bank = training_bank((rgb, mct, semantics), smoke)
    receipts = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            file = ROOT / f"{mode}-{seed}.pth"
            receipt = file.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["checkpoint_SHA256"] != digest(file) or row["protocol_SHA256"] != digest(lock):
                    raise ValueError("An existing training artifact was modified")
                receipts.append(row)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 48041)
            model = LowField().cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.0017, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.0001)
            started = perf_counter()
            for i in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                views, other = draw_examples(bank, rng)
                terms = [relative_loss(model, item, mode) for item in views]
                rw, broad = [relative_loss(model, item, mode)[0] for item in other]
                value = sum(score for score, _ in terms) / 4. + .75 * rw + .75 * broad
                if mode in ("paired", "cross_scene"):
                    value = value + 1.50 * paired_penalty([error for _, error in terms], mode == "cross_scene")
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                opt.step()
                scheduler.step()
            output = dict(mode=mode, seed=seed, updates=i + 1, loss=value.item(), seconds=round(perf_counter() - started, 2))
            print(output, flush=True)
            if not smoke:
                torch_save(file, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
                row = dict(output, checkpoint_SHA256=digest(file), protocol_SHA256=digest(lock), finished_utc=now())
                atomic_write(receipt, row)
                receipts.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=receipts, updates=sum(r["updates"] for r in receipts),
            protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def load_heads(modes=MODES, seeds=SEEDS):
    lock = ROOT / "protocol.json"
    registered = read(ROOT / "completed.json")
    if registered["protocol_SHA256"] != digest(lock) or not matches_registered(read(lock)['code_SHA256'], digest(__file__)):
        raise ValueError("The training source or registration changed")
    if {(r["mode"], r["seed"]) for r in registered["runs"]} != {(m, s) for m in MODES for s in SEEDS}:
        raise ValueError("Missing matched checkpoint")
    result = {}
    for row in registered["runs"]:
        mode, seed = row["mode"], row["seed"]
        if mode in modes and seed in seeds:
            path = ROOT / f"{mode}-{seed}.pth"
            if row["checkpoint_SHA256"] != digest(path):
                raise ValueError("An RGB checkpoint changed")
            head = LowField().cuda()
            head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            result[mode, seed] = head.eval().requires_grad_(False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    fit(parser.parse_args().smoke)

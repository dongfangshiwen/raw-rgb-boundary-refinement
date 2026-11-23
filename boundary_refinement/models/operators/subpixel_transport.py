"""Train and ablate local, mass-conserving subpixel transport on a frozen dual-output system."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from itertools import islice

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import DATA, paths
from boundary_refinement.evaluation.semantic.resolution_interaction import DualScaleFeatures, ROOT as ANCHOR, SEEDS, load as anchors
from boundary_refinement.evaluation.semantic.restoration_cue import match_channels
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/subpixel-transport-v92"
AUTHOR = REPO.parent / "benchmark_data/dysample-author-v92/dysample.py"
AUTHOR_SHA = "b9069bb414da4ca4448cec9f778c678701812564539b5b3034ac662d6290ab7b"
ARMS = ("raw_free", "raw_mass", "dual_free", "dual_mass", "tone_mass", "shift_mass", "dysample")
PRIMARY = "dual_mass"
STEPS, SIDE, UPSCALE = 320, 28, 4
RANGES = (.02, .05, .10, .20)


def protocol():
    return dict(code_SHA256=digest(__file__), frozen_anchor_SHA256=digest(ANCHOR / "completed.json"),
                fit_SHA256=digest(DATA / "fit.json"), author_SHA256=AUTHOR_SHA,
                author_commit="81a1de5caa95d55a0f5488425fa53ec7ef47f8f0",
                arms=list(ARMS), seeds=list(SEEDS), steps=STEPS, crop=SIDE, upscale=UPSCALE, ranges=list(RANGES),
                primary=PRIMARY, primary_comparators=["bilinear", "raw_mass", "dual_free", "tone_mass", "dysample"],
                primary_endpoint="Six-native-pixel valid semantic boundary band, pooled confusion mIoU over 19 "
                "classes; main fog diagnostic is Foggy Driving Fine. Also report full mIoU, traffic light IoU and "
                "85%-recall false positive rate; no thresholds chosen with test labels. Coarse annotations have no "
                "boundary endpoint. One GoPro recording yields no confidence interval.",
                transfer="Freeze before scoring all recording-disjoint public MUSES snow validation frames. Recheck "
                "MUSES rain, Foggy Zurich and full Foggy Driving from previous rounds. Attribute to dehazing only "
                "if supported against raw and tone guides, and conservation only against same-guide free transport. "
                "Describe snow as cross-weather transfer, and report contradictory comparisons.",
                sampling="96 densely annotated MUSES fog frames in nine training recordings. Choose one recording-uniform "
                "patch and one equally sampled rare-class/label-boundary centered patch per update. All arms receive "
                "the same two patches and the same weighted CE with extra weight on true boundaries. "
                "Frozen V84 raw two-scale logits per seed. No partial labels in this experiment.",
                formula="On a four-times grid, guide affinities reweight the bilinear stencil on coarse class transitions. "
                "In the mass arms project 16 child rows to unit sum and nine donor columns to the original bilinear "
                "column sums. The resulting local sum constraint applies to logits, not to thresholded mask area. "
                "At arbitrary native shapes add the resized correction to the direct native bilinear prediction.",
                optimizers="AdamW, cosine to one tenth; 0.035 on nine affinity scalars, 0.002 on author's 640 "
                "LP offset weights, weight decay zero. Equal numbers of samples and steps.")


def author_class():
    if digest(AUTHOR) != AUTHOR_SHA:
        raise ValueError("DySample source differs from the pinned author checkout")
    spec = importlib.util.spec_from_file_location("author_dysample_v92", AUTHOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DySample


def geometry(device):
    axis = (torch.arange(4, device=device) + .5) / 4. - .5
    query = torch.cartesian_prod(axis, axis)
    neighbor = torch.arange(-1, 2, device=device)
    stencil = torch.cartesian_prod(neighbor, neighbor)
    diff = query[:, None] - stencil
    return (1. - diff.abs()).clamp_min(0.).prod(-1), stencil, diff.square().sum(-1)


def conserve(plan, target):
    # Three scaling iterations followed by exact nonnegative marginal rounding.
    for _ in range(3):
        plan = plan * (target / plan.sum(-2, keepdim=True).clamp_min(1e-12))
        plan = plan / plan.sum(-1, keepdim=True).clamp_min(1e-12)
    plan = plan * (target / plan.sum(-2, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
    row_deficit = (1. - plan.sum(-1, keepdim=True)).clamp_min(0.)
    col_deficit = (target - plan.sum(-2, keepdim=True)).clamp_min(0.)
    return plan + row_deficit * (col_deficit / col_deficit.sum(-1, keepdim=True).clamp_min(1e-12))


class ColorTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self.range_weights = nn.Parameter(torch.full((2, len(RANGES)), -.5))
        self.temperature = nn.Parameter(torch.zeros(()))

    def forward(self, first, second, base, balanced):
        weight = 3. * self.range_weights.sigmoid()
        cost = (first @ weight[0]) + (second @ weight[1])
        temperature = .25 + 1.5 * self.temperature.sigmoid()
        kernel = (base.clamp_min(1e-10).log() * temperature - cost).masked_fill(base == 0, -torch.inf)
        plan = kernel.softmax(-1)
        return conserve(plan, base.sum(0)) if balanced else plan


def new_model(arm):
    return (author_class()(19, scale=4, groups=1, style="lp") if arm == "dysample" else ColorTransport()).cuda()


def pack(image, height, width):
    return image.reshape(image.shape[0], height, 4, width, 4).permute(1, 3, 2, 4, 0).reshape(height * width, 16, -1)


def unpack(values, h, w):
    return values.reshape(h, w, 4, 4, -1).permute(4, 0, 2, 1, 3).reshape(1, -1, 4 * h, 4 * w)


def render_guides(rgb, restored, grid):
    rgb = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255.
    fine = resize(rgb, restored.shape[-2:], antialias=True)
    tone = match_channels(fine, restored)
    size = tuple(4 * x for x in grid)
    raw, actual, matched = (resize(v, size, antialias=True) for v in (rgb, restored, tone))
    shift = tuple(4 * (x // 2) for x in grid)
    wrong = (raw + (actual - raw).roll(shift, dims=(-2, -1))).clamp(0., 1.)
    return torch.cat((raw, actual, matched, wrong)).clamp(0., 1.).mul(255.).round().byte()


def guide_stencil(guides, labels, only_active=True):
    h, w = labels.shape
    base, offsets, distance = geometry(labels.device)
    idx = torch.arange(h * w, device=labels.device)
    neighbor = ((idx[:, None] // w + offsets[:, 0]).clamp(0, h - 1) * w +
                (idx[:, None] % w + offsets[:, 1]).clamp(0, w - 1))
    if only_active:
        idx = idx[(labels.flatten()[neighbor] != labels.flatten()[:, None]).any(-1)]
    neighbor = neighbor[idx]
    bands = torch.tensor(RANGES, device=labels.device)
    features = []
    for image in guides:
        subpixel = pack(image, h, w).float() / 255.
        local = subpixel[idx, :, None] - subpixel.mean(1)[neighbor][:, None]
        difference = local.square().mean(-1, keepdim=True)
        features.append(1. - (-.5 * difference / bands.square()).exp())
    return idx, neighbor, base, distance, features


def prediction(arm, model, logits, stencil):
    h, w = logits.shape[-2:]
    canonical = F.interpolate(logits, scale_factor=4., mode="bilinear", align_corners=False)
    if arm == "dysample":
        return model(logits) - canonical
    idx, neighbor, base, distance, features = stencil
    raw = features[0]
    if arm == "jbu":
        # Fixed 3x3 Gaussian joint bilateral interpolation, spatial sigma .65 cells, RGB RMS sigma .05.
        color_kernel = (1. - raw[..., 1]).clamp_min(1e-12)
        weights = color_kernel * (-.5 * distance / .65 ** 2).exp()
        plan = weights / weights.sum(-1, keepdim=True)
    else:
        second = {"raw": 0, "dual": 1, "tone": 2, "shift": 3}[arm.split("_")[0]]
        plan = model(raw, features[second], base, arm.endswith("_mass"))
    z = logits[0].permute(1, 2, 0).reshape(h * w, 19)
    # Subtract the base on the same stencil before resampling. The output is a correction only.
    delta = (plan - base) @ z[neighbor]
    empty = logits.new_zeros((h * w, 16, 19))
    return unpack(empty.index_copy(0, idx, delta), h, w)


@dataclass
class Frame:
    group: str
    guides: torch.Tensor
    logits: list
    target: torch.Tensor
    edges: torch.Tensor
    rare: torch.Tensor


def coarse_locations(target, h, w):
    label = target.long()
    valid = label < 19
    left = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    up = (label[1:] != label[:-1]) & valid[1:] & valid[:-1]
    boundary = F.pad(left, (1, 0)) | F.pad(left, (0, 1)) | F.pad(up, (0, 0, 1, 0)) | F.pad(up, (0, 0, 0, 1))
    rare = ((label == 6) | (label == 7) | (label == 14))
    return tuple(pack(v[None], h, w).any(1).reshape(h, w).nonzero().short().cpu() for v in (boundary, rare))


@torch.no_grad()
def train_frames(engine, raw, smoke):
    bank, counts = defaultdict(list), Counter()
    for key, group, _, _, truth, rgb in islice(paths("fit"), 4 if smoke else None):
        views, restored, _ = engine(rgb)
        h, w = views.fine.shape[-2:]
        guides = render_guides(rgb, restored, (h, w)).cpu()
        labels = F.interpolate(truth[None, None].float(), (4 * h, 4 * w), mode="nearest-exact")[0, 0].byte()
        edges, rare = coarse_locations(labels, h, w)
        seed_logits = [raw_anchor(raw["raw_detail", s], views).half().cpu() for s in SEEDS]
        bank[group].append(Frame(group, guides, seed_logits, labels.cpu(), edges, rare))
        counts[group] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(encoded=sum(counts.values())), flush=True)
    if not smoke and (sum(counts.values()), len(bank)) != (96, 9):
        raise ValueError("Fog training group mismatch")
    return list(bank.values())


def choose(rng, values):
    return values[int(rng.integers(len(values)))]


def train_crop(pool, rng, trial, focused, step):
    frame = choose(rng, choose(rng, pool))
    _, _, h, w = frame.logits[trial].shape
    if focused:
        focus = frame.rare if step % 2 == 0 and len(frame.rare) else frame.edges
    else:
        focus = ()
    if len(focus):
        centre = choose(rng, focus).numpy().astype(int)
        corner = np.clip(centre - rng.integers(SIDE // 4, 3 * SIDE // 4, 2), 0, [h - SIDE, w - SIDE])
    else:
        corner = [int(rng.integers(n - SIDE + 1)) for n in (h, w)]
    y, x = corner
    lo = (..., slice(y, y + SIDE), slice(x, x + SIDE))
    hi = (..., slice(4 * y, 4 * (y + SIDE)), slice(4 * x, 4 * (x + SIDE)))
    return frame.logits[trial][lo].cuda().float(), frame.guides[hi].cuda(), frame.target[hi].cuda().long()


def crop_loss(arm, model, example):
    z, guides, truth = example
    features = None if arm == "dysample" else guide_stencil(guides, z.argmax(1)[0])
    y = F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False) + prediction(arm, model, z, features)
    inside = (..., slice(8, -8), slice(8, -8))
    target = truth[inside]
    from boundary_refinement.models.segmentation.restoration_affinity import boundary_band
    edge = boundary_band(truth)[inside]
    weight = (target < 19).float() * (1. + 3. * edge + 2. * ((target == 6) | (target == 7) | (target == 14)))
    ce = F.cross_entropy(y[inside], target[None], ignore_index=255, reduction="none")[0]
    return (ce * weight).sum() / weight.sum().clamp_min(1.)


def fit_one(arm, seed, pool, smoke):
    trial = SEEDS.index(seed)
    set_seed(seed)
    net = new_model(arm)
    file = ROOT / f"{arm}-{seed}.pth"
    receipt = file.with_suffix(".json")
    lock = None if smoke else digest(ROOT / "protocol.json")
    if not smoke and receipt.exists():
        old = read(receipt)
        if digest(file) != old["weights_SHA256"] or lock != old["protocol_SHA256"]:
            raise ValueError("Frozen transport checkpoint changed")
        return old
    lr = .002 if arm == "dysample" else .035
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=lr / 10.)
    rng = np.random.default_rng(92001 + seed)
    n = 3 if smoke else STEPS
    for step in range(n):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(crop_loss(arm, net, train_crop(pool, rng, trial, focus, step)) for focus in (False, True)) / 2.
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
    row = dict(method=arm, seed=seed, steps=n, params=sum(p.numel() for p in net.parameters()), loss=float(loss.detach()),
               sampler_end=sha256(str(rng.bit_generator.state).encode()).hexdigest())
    if not smoke:
        torch_save(file, {k: v.cpu() for k, v in net.state_dict().items()})
        row.update(weights_SHA256=digest(file), protocol_SHA256=lock)
        atomic_write(receipt, row)
    print(row, flush=True)
    return row


def fit(smoke):
    configure_cuda()
    cfg = protocol()
    lock = ROOT / "protocol.json"
    if not smoke:
        if lock.exists() and read(lock) != cfg:
            raise ValueError("Training protocol changed")
        atomic_write(lock, cfg)
    frames = train_frames(DualScaleFeatures(), anchors(), smoke)
    runs = [fit_one(a, s, frames, smoke) for s in (SEEDS[:1] if smoke else SEEDS) for a in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=runs, total_updates=sum(r["steps"] for r in runs),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def load():
    path = ROOT / "protocol.json"
    card = read(ROOT / "completed.json")
    if digest(path) != card["protocol_SHA256"] or not matches_registered(read(path), protocol()):
        raise ValueError("Frozen transport experiment changed")
    models = {}
    for row in card["runs"]:
        name = row["method"], row["seed"]
        net = new_model(name[0]).eval().requires_grad_(False)
        f = ROOT / f"{name[0]}-{name[1]}.pth"
        if digest(f) != row["weights_SHA256"] or row["steps"] != STEPS:
            raise ValueError(f"Unexpected transport fit {name}")
        net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True), strict=True)
        models[name] = net
    if set(models) != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Incomplete fitted arms")
    return models


def check():
    configure_cuda()
    torch.manual_seed(923)
    z = torch.randn(1, 19, 7, 9, device="cuda")
    guides = torch.randint(256, (4, 3, 28, 36), device="cuda", dtype=torch.uint8)
    idx, neighbor, base, _, features = guide_stencil(guides, z.argmax(1)[0], only_active=False)
    baseline = unpack(base @ z[0].permute(1, 2, 0).reshape(-1, 19)[neighbor], 7, 9)
    error = float((baseline - F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False)).abs().max())
    net = new_model(PRIMARY)
    weights = net(features[0], features[1], base, True)
    row = float((weights.sum(-1) - 1.).abs().max().detach())
    col = float((weights.sum(-2) - base.sum(0)).abs().max().detach())
    assert weights.min() >= 0 and max(error, row, col) < 5e-6
    constant = z[:, :, :1, :1].expand_as(z)
    none = guide_stencil(guides, constant.argmax(1)[0])
    assert len(none[0]) == 0 and prediction(PRIMARY, net, constant, none).abs().max() == 0
    assert torch.allclose(unpack(pack(guides[0], 7, 9), 7, 9)[0], guides[0])
    return dict(bilinear=error, row_sum=row, column_sum=col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "fit"))
    mode = parser.parse_args().mode
    print(check(), flush=True)
    if mode != "check":
        fit(mode == "smoke")

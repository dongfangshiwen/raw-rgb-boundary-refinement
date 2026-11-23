"""Fit a local restoration solver using overlap-weighted candidate covariance."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.cross_scale_rgb_v44 import resize
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_parallel_rgb_v8 import bind
from scripts.orthogonal_risk_v52 import anchors as primary_anchors, encode as rgb_candidates
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.region_restore_v48 import SEEDS, view_size
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save
from scripts.veil_response_v45 import NH_ROOT, RW_FIT, TRAIN_CARD, training_pairs


ROOT = REPO / "results/overlap-quadratic-v53"
FITTING = ("direct", "alignment")
STEPS, MAX_GRID = 480, 12


class Partition:
    """Adjoint of a non-negative separable bilinear partition of unity."""
    def __init__(self, image, grid):
        self.grid = grid
        axes = [resize(torch.eye(g, device=image.device, dtype=image.dtype)[None, :, :, None], (n, 1))
                [0, :, :, 0].T for g, n in zip(grid, image.shape[-2:], strict=True)]
        self.rows, self.cols = axes
        self.mass = self.rows.sum(0)[:, None] * self.cols.sum(0)[None, :]

    def average(self, value):
        return ((self.rows.T @ value) @ self.cols) / self.mass.clamp_min(1e-8)

    def expand(self, value):
        return (self.rows @ value) @ self.cols.T


@dataclass
class Geometry:
    features: torch.Tensor
    gram: torch.Tensor
    scale: torch.Tensor
    partition: Partition
    base: torch.Tensor
    deltas: torch.Tensor


def references():
    bind()
    mct, release = load_author_reference()
    return primary_anchors(), mct, release


@torch.no_grad()
def geometry(x, locked):
    candidates, mct, _ = locked
    base, delta, _, _, _ = rgb_candidates(x, candidates)
    direct = (base + delta).clamp(0, 1)
    second = author_rgb(mct, x).clamp(0, 1)
    shifts = torch.cat((direct - base, second - base, x - base), dim=0)
    basis = Partition(x, view_size(x, MAX_GRID, 4))
    gram = basis.average((shifts[:, None] * shifts[None, :]).mean(2)).permute(2, 3, 0, 1)[None]
    diag = gram.diagonal(dim1=-1, dim2=-2).clamp_min(1e-8).sqrt()
    norms = diag.permute(0, 3, 1, 2)
    corr = (gram / (diag[..., :, None] * diag[..., None, :])).clamp(-1., 1.)
    edges = torch.stack([corr[..., i, j] for i, j in ((0, 1), (0, 2), (1, 2))], dim=1)
    colors = basis.average(torch.cat((2.*x - 1., 2.*base - 1., *(4. * d[None] for d in shifts)), dim=1))
    contrast = (basis.average(x.square()) - basis.average(x).square()).clamp_min(0).sqrt()
    features = torch.cat((colors, 4.*norms, edges, 4.*contrast), dim=1)
    return Geometry(features, gram, diag, basis, base, shifts)


def label(context, truth):
    residual = truth - context.base
    dot = context.partition.average((residual * context.deltas).mean(1)).permute(1, 2, 0)[None]
    error = context.partition.average(residual.square().mean(1))
    return dot, error


class LocalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv2d(24, 3, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x).permute(0, 2, 3, 1)


def direct(value):
    logits = value - 2.
    return torch.cat((logits.new_zeros((*logits.shape[:-1], 1)), logits), dim=-1).softmax(-1)[..., 1:]


def simplex(point):
    """Project on non-negative weights whose sum is <= one (anchor gets the rest)."""
    positive = point.clamp_min(0.)
    order = point.sort(descending=True).values
    rank = torch.arange(1, order.shape[-1]+1, device=order.device, dtype=order.dtype)
    threshold = (order.cumsum(-1) - 1.) / rank
    level = threshold.amax(-1, keepdim=True)
    fixed = (point - level).clamp_min(0.)
    return torch.where(positive.sum(-1, keepdim=True) > 1., fixed, positive)


def quadratic(gram, linear, weight):
    return ((gram @ weight.unsqueeze(-1)).squeeze(-1) * weight).sum(-1) - 2. * (linear * weight).sum(-1)


def solve(gram, linear, diagonal=False):
    diag = gram.diagonal(dim1=-1, dim2=-2)
    shape = torch.diag_embed(diag) if diagonal else gram
    ridge = (.02 * diag.mean(-1, keepdim=True)).clamp_min(1e-6)
    metric = shape + torch.diag_embed(ridge.expand_as(diag))
    norm = metric.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-6)
    # The max row sum bounds the spectral norm for these symmetric matrices.
    value = torch.zeros_like(linear)
    for _ in range(32):
        grad = (metric @ value.unsqueeze(-1)).squeeze(-1) - linear
        value = simplex(value - grad / norm)
    return value


def preference(net, context):
    return .15 * net(context.features) * context.scale


def decoded(context, knots, strength=1., hard=False):
    knots = knots.permute(0, 3, 1, 2)
    weights = F.interpolate(knots, context.base.shape[-2:], mode="nearest") if hard else context.partition.expand(knots)
    addition = (context.deltas * weights[0, :, None]).sum(0, keepdim=True)
    return (context.base + strength * addition).clamp(0, 1)


def upper_bound(context, weights, dot):
    return (quadratic(context.gram, dot, weights)[0] * context.partition.mass).sum() / context.partition.mass.sum()


@torch.no_grad()
def data(locked, smoke):
    banks = defaultdict(list)
    counts = Counter()
    for domain, _, _, fog, clean in training_pairs():
        if smoke and counts[domain] >= 3:
            continue
        images = torch.cat((as_float(fog), as_float(clean)))
        frame = resize(images, view_size(images, 640), antialias=True)
        # The optional gamma-transformed view supplies additional low-light exposure conditions to every arm.
        gamma = 1.65 + .35 * (counts[domain] % 3) / 2.
        for variant in (frame, frame.pow(gamma)):
            x, y = variant.split(1)
            fields = geometry(x, locked)
            gt_dot, gt_error = label(fields, y)
            banks[domain].append((fields.features, fields.gram, fields.scale, gt_dot, gt_error))
        counts[domain] += 1
    if counts != (dict(lm=3, rw=3, nh=3) if smoke else dict(lm=128, rw=86, nh=36)):
        raise ValueError("Unexpected overlap-aware fitting population")
    print(dict(pairs=dict(counts), views={k: len(v) for k, v in banks.items()}), flush=True)
    return list(banks.values())


def objective(net, arm, sample):
    features, gram, norm, target, mse = sample
    estimate = net(features)
    if arm == "direct":
        return (quadratic(gram, target, direct(estimate)) / mse.clamp_min(.003)).mean()
    truth = (target / norm).clamp(-.35, .35)
    error = F.smooth_l1_loss(.15*estimate, truth, beta=.05, reduction="none")
    return (error / (mse.clamp_min(.004).sqrt()[..., None])).mean()


def fit(smoke):
    configure_cuda()
    locked = references()
    parent = REPO / "results/controlled-rgb-band-v51"
    plan = dict(code_SHA256=digest(__file__), seeds=list(SEEDS), modes=list(FITTING), steps=STEPS,
        parameters=sum(p.numel() for p in LocalHead().parameters()), backbone_SHA256=locked[0][-1], author=locked[2],
        files={str(p): digest(p) for p in (parent / "completed.json", parent / "calibration.json", TRAIN_CARD, RW_FIT,
             NH_ROOT / "data_manifest.json")},
        idea="Use a partition of unity to blend per-node candidate directions. Derive a true image MSE upper bound from a weighted local Gram matrix. Estimate signed target alignment and optimize each local simplex using observed covariance. Apply the same learned alignments in the diagonal and full-covariance designs.",
        controls="Direct local simplex gating under the same analytic training upper bound, diagonal-only covariance, full covariance, and nearest-grid rendering of the same coefficients. Retain a frozen equal mixture and each individual prediction as strong views.",
        scope="LMHaze/RW2AH/NH-HAZE fit only, with equal gamma augmentation and fixed optimization steps across modes. Calibration uses the old 120 validation pairs. RRSHID test, NT-HAZE published paired data and HyperHazeOff RGB are used only for evaluation here. HyperHazeOff has been scored by V52 on a different algorithm.")
    protocol = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if protocol.exists() and read(protocol) != plan:
            raise ValueError("This preregistered algorithm changed")
        if not protocol.exists():
            atomic_write(protocol, plan)
    banks = data(locked, smoke)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for arm in FITTING:
            path = ROOT / f"{arm}-{seed}.pth"
            info = path.with_suffix(".json")
            if not smoke and info.exists():
                row = read(info)
                if row["weights_SHA256"] != digest(path) or row["protocol_SHA256"] != digest(protocol):
                    raise ValueError("A matched local checkpoint changed")
                ledger.append(row)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 53129)
            net = LocalHead().cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00010)
            for step in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                losses = [objective(net, arm, pool[int(torch.randint(len(pool), (), generator=rng))]) for pool in banks]
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
                opt.step()
                schedule.step()
            out = dict(mode=arm, seed=seed, updates=step + 1, loss=loss.item())
            print(out, flush=True)
            if not smoke:
                torch_save(path, {k: v.cpu().clone() for k, v in net.state_dict().items()})
                out.update(weights_SHA256=digest(path), protocol_SHA256=digest(protocol))
                atomic_write(info, out)
                ledger.append(out)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, protocol_SHA256=digest(protocol),
            updates=sum(r["updates"] for r in ledger), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def fitted():
    d = ROOT / "protocol.json"
    receipt = read(ROOT / "completed.json")
    if digest(__file__) != read(d)["code_SHA256"] or digest(d) != receipt["protocol_SHA256"]:
        raise ValueError("The source lock has changed")
    choices = {}
    for row in receipt["runs"]:
        arm, seed = row["mode"], row["seed"]
        file = ROOT / f"{arm}-{seed}.pth"
        if digest(file) != row["weights_SHA256"]:
            raise ValueError("A local checkpoint changed")
        net = LocalHead().cuda()
        net.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"), strict=True)
        choices[arm, seed] = net.eval().requires_grad_(False)
    if set(choices) != {(m, s) for m in FITTING for s in SEEDS}:
        raise ValueError("Incomplete matched checkpoint matrix")
    return choices


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--smoke", action="store_true")
    fit(cli.parse_args().smoke)

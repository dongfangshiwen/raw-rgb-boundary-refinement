"""Fit signed residual projections for an analytic error-guarded RGB edit."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.models.restoration.controlled_rgb_band import ROOT as PARENT, band, models, signals
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.region_restore import SEEDS, view_size
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.paired.veil_response import NH_ROOT, RW_FIT, training_pairs


ROOT = REPO / "results/orthogonal-risk-v52"
MODES = ("ordinary_gate", "signed_gate", "scalar", "split")
LOWER, UPPER = -.65, 1.25
COVERAGE, MIN_EDIT = .80, 1. / 255.
STEPS = 480


def components(delta):
    average = delta.mean((-1, -2), keepdim=True).expand_as(delta)
    parts = torch.cat((average, delta - average), dim=0)
    norm = parts.square().mean((1, 2, 3)).sqrt()
    vectors = parts / norm.clamp_min(1e-8)[:, None, None, None]
    return vectors, norm


def inner(first, second):
    return (first * second).mean((-3, -2, -1))


def observations(raw, anchor, residual, unit, norm):
    images = torch.cat((raw, anchor, anchor - raw, residual), dim=1)
    spread = images.flatten(2)
    moments = (spread.mean(2), spread.std(2, unbiased=False), spread.abs().mean(2),
               F.adaptive_avg_pool2d(images, (2, 2)).flatten(1))
    raw_gap = anchor - raw
    alignment = inner(raw_gap, unit)[None]
    strength = inner(raw_gap, raw_gap).sqrt().reshape(1, 1)
    return torch.cat((*moments, alignment, norm[None], strength), dim=1)[0]


def anchors():
    cal = PARENT / "calibration.json"
    lock = read(cal)
    if lock["fit_SHA256"] != digest(PARENT / "completed.json"):
        raise ValueError("Unexpected parent amplitude registration")
    restorer, signature = rgb_checkpoint()
    bank = models()
    return restorer, [bank["coarse", seed] for seed in SEEDS], lock["amplitudes"]["coarse"], signature


def encode(x, locked):
    restorer, adapters, scale, _ = locked
    anchor = pgl_rgb(restorer, x).clamp(0., 1.)
    signal = signals(x, anchor)
    field = torch.stack([band(net(signal, "direct"), "coarse") for net in adapters]).mean(0)
    change = scale * resize(field, x.shape[-2:])
    unit, norm = components(change)
    cues = observations(x, anchor, change, unit, norm)
    return anchor, change, unit, norm, cues


def target(native, clear, unit):
    clean_gap = clear - native
    return inner(clean_gap, unit), inner(clean_gap, clean_gap)[0]


class TrustHead(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        a, b = self.layers(features).split(2, dim=-1)
        return a, .006 + .15 * b.sigmoid()


def direction(norm):
    return norm / norm.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)


def pack_predictions(net, cues, norm, mode):
    logits, uncertainty = net(cues)
    if mode == "scalar":
        return .10 * logits.mean(-1, keepdim=True), uncertainty.mean(-1, keepdim=True)
    if mode in ("ordinary_gate", "signed_gate"):
        a = UPPER * (logits - 2.).sigmoid()
        if mode == "signed_gate":
            a = LOWER + (UPPER - LOWER) * (logits - .526093).sigmoid()
        return a * norm, uncertainty
    return .10 * logits, uncertainty


def nll(loc, scale, label):
    # Student-t with two degrees of freedom, up to an additive constant.
    return (scale.log() + 1.5 * ((label - loc).square() / (2. * scale.square())).log1p()).mean()


def loss(net, mode, cues, norm, truth, base):
    values, scale = pack_predictions(net, cues, norm, mode)
    if mode in ("ordinary_gate", "signed_gate"):
        change = ((values.square() - 2. * values * truth).sum(-1) / base.clamp_min(.0025)).mean()
        return change + .05 * nll(values.detach(), scale, truth)
    label = (truth * direction(norm)).sum(-1, keepdim=True) if mode == "scalar" else truth
    return nll(values, scale, label)


def nonconformity(values, scale, oracle, norm, mode):
    real = (oracle * direction(norm)).sum(-1, keepdim=True) if mode == "scalar" else oracle
    return torch.linalg.vector_norm((real - values) / scale, dim=-1)


def quantile(scores, coverage):
    """Split-conformal upper radius with the plus-one rank for exchangeable groups."""
    ranked = sorted(scores)
    index = ceil((len(ranked) + 1) * coverage)
    return ranked[index - 1] if 1 <= index <= len(ranked) else float("inf")


def guarded_step(prediction, scale, feasible, radius):
    """Exact optimum on the segment from the anchor to a feasible coefficient vector."""
    if not radius < float("inf"):
        return torch.zeros_like(feasible), feasible.new_zeros(feasible.shape[:-1])
    energy = feasible.square().sum(-1)
    radius_support = radius * torch.linalg.vector_norm(scale * feasible, dim=-1)
    lower_alignment = (prediction * feasible).sum(-1) - radius_support
    amount = (lower_alignment / energy.clamp_min(1e-12)).clamp(0., 1.)
    coeff = amount[..., None] * feasible
    lower_gain = 2. * amount * lower_alignment - amount.square() * energy
    return coeff, lower_gain


def coefficients(values, scale, norms, mode, bound=0.):
    if mode in ("ordinary_gate", "signed_gate"):
        return values, norms.new_zeros(norms.shape[:-1])
    if mode == "scalar":
        length = norms.square().sum(-1, keepdim=True).sqrt()
        feasible = values.clamp(min=LOWER * length, max=UPPER * length)
        coeff, lower = guarded_step(values, scale, feasible, bound)
        return coeff * direction(norms), lower
    feasible = values.clamp(min=LOWER * norms, max=UPPER * norms)
    return guarded_step(values, scale, feasible, bound)


def reconstruct(native, coeff, basis):
    return (native + (coeff[:, None, None, None] * basis).sum(0, keepdim=True)).clamp(0., 1.)


@torch.no_grad()
def training(locked, smoke):
    banks = defaultdict(list)
    for domain, name, group, fog, clean in training_pairs():
        if smoke and len(banks[domain]) >= 3:
            continue
        pair = torch.cat((as_float(fog), as_float(clean)))
        raw, truth = resize(pair, view_size(pair, 768), antialias=True).split(1)
        anchor, _, basis, norm, cues = encode(raw, locked)
        z, mse = target(anchor, truth, basis)
        banks[domain].append((cues, norm, z, mse))
    counts = {k: len(v) for k, v in banks.items()}
    if counts != (dict(lm=3, rw=3, nh=3) if smoke else dict(lm=128, rw=86, nh=36)):
        raise ValueError(f"Haze projection training population changed {counts}")
    tables = [tuple(torch.stack(col) for col in zip(*bank, strict=True)) for bank in banks.values()]
    features = torch.cat([b[0] for b in tables])
    center, scale = features.mean(0), features.std(0, unbiased=False).clamp_min(.003)
    tables = [(((f - center) / scale).clamp(-5., 5.), *v) for f, *v in tables]
    state = dict(center=center.cpu(), scale=scale.cpu())
    print(dict(training=counts, feature_dim=len(center)), flush=True)
    return tables, state


def fit(smoke):
    configure_cuda()
    locked = anchors()
    lock = ROOT / "protocol.json"
    registration = dict(code_SHA256=digest(__file__), rgb_backbone_SHA256=locked[-1], seeds=list(SEEDS), modes=list(MODES),
        steps=STEPS, input_SHA256={str(p): digest(p) for p in (TRAIN_CARD, RW_FIT, NH_ROOT / "data_manifest.json",
             PARENT / "completed.json", PARENT / "calibration.json")},
        mechanism="Split a frozen V51 correction into two orthogonal image-space components. Learn the clean residual's projection on each unit direction. Use a group-calibrated ellipsoid and maximize the lower squared-error gain along the line segment from zero to a feasible clipped prediction.",
        alternatives="Same data and head size. Compare ordinary and signed end-to-end gates, a scalar projection on the same complete correction, and the orthogonal projection with and without calibration.",
        fitting="Train only small heads on registered LMHaze fit, RW2AH fit, and NH-HAZE fit. Every update samples equal count from each dataset. The V51 base and proposals are frozen, averaged over its three seeds.",
        evaluation="Use earlier calibration data only for radii and equal-coverage controls. Report selective harm, useful edits, PSNR and coverage on already observed development domains and new RGB collections. A 0.10 dB PSNR loss counts as harm. The claimed lower gain is conditional on the true projections being in the uncertainty region. Cross-domain exchangeability is unverified.",
        relative_amplitude_bounds=[LOWER, UPPER], conformal_coverage=COVERAGE, edit_rms_threshold=MIN_EDIT)
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if lock.exists() and read(lock) != registration:
            raise ValueError("The registered risk experiment changed")
        if not lock.exists():
            atomic_write(lock, registration)
    tables, state = training(locked, smoke)
    normpath = ROOT / "normalizer.pth"
    if not smoke:
        if normpath.exists():
            old = torch.load(normpath, weights_only=True, map_location="cpu")
            if any(not torch.allclose(old[k], v, rtol=1e-4, atol=1e-5) for k, v in state.items()):
                raise ValueError("Previously registered cue normalization changed")
        else:
            torch_save(normpath, state)
    width = len(state["center"])
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            path = ROOT / f"{mode}-{seed}.pth"
            info = path.with_suffix(".json")
            if not smoke and info.exists():
                record = read(info)
                if record["weights_SHA256"] != digest(path) or record["protocol_SHA256"] != digest(lock):
                    raise ValueError("A fitted trust head changed")
                ledger.append(record)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 52039)
            net = TrustHead(width).cuda()
            op = torch.optim.AdamW(net.parameters(), lr=.0012, weight_decay=.01)
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(op, STEPS, eta_min=.00008)
            for step in range(2 if smoke else STEPS):
                op.zero_grad(set_to_none=True)
                batch = []
                for bank in tables:
                    idx = torch.randint(len(bank[0]), (12,), generator=rng).cuda()
                    batch.append([col[idx] for col in bank])
                values = [torch.cat(col) for col in zip(*batch, strict=True)]
                total = loss(net, mode, *values)
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3., error_if_nonfinite=True)
                op.step()
                decay.step()
            row = dict(mode=mode, seed=seed, updates=step + 1, loss=total.item(), trainable=sum(p.numel() for p in net.parameters()))
            print(row, flush=True)
            if not smoke:
                torch_save(path, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                row.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
                atomic_write(info, row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, optimizer_updates=sum(v["updates"] for v in ledger),
            code_SHA256=digest(__file__), protocol_SHA256=digest(lock), norm_SHA256=digest(normpath),
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now()))


def fitted():
    lock = ROOT / "protocol.json"
    card = read(ROOT / "completed.json")
    norm = ROOT / "normalizer.pth"
    if not matches_registered((card['code_SHA256'], card['protocol_SHA256'], card['norm_SHA256']), (digest(__file__), digest(lock), digest(norm))):
        raise ValueError("Mismatched trust experiment receipt")
    state = torch.load(norm, map_location="cuda", weights_only=True)
    bank = {}
    for row in card["runs"]:
        mode, seed = row["mode"], row["seed"]
        path = ROOT / f"{mode}-{seed}.pth"
        if digest(path) != row["weights_SHA256"]:
            raise ValueError("A trust-model checkpoint changed")
        net = TrustHead(len(state["center"])).cuda()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[mode, seed] = net.eval().requires_grad_(False)
    if set(bank) != {(m, s) for m in MODES for s in SEEDS}:
        raise ValueError("Matched trust estimates are incomplete")
    return state, bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    fit(parser.parse_args().smoke)

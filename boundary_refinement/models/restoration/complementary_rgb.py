"""Learn signed RGB corrections outside the space of the existing restoration candidates."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.training.decision_overlap import MANIFEST, exact, rr_records
from boundary_refinement.training.exact_overlap import OFFSETS, neighbor_axis
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.training.overlap_quadratic import Partition, SEEDS, decoded, geometry, references
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.region_restore import view_size
from boundary_refinement.models.restoration.local_utility import ROOT as PREVIOUS, coefficients as mixture, fitted, residual_actions
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.paired.veil_response import TRAIN_CARD, training_pairs


ROOT = REPO / "results/complementary-rgb-v59"
MODES = ("global", "local", "projected")
STEPS = 480
PRIOR_SEED = SEEDS[0]
MAX_RMS = .13


class ColorHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(31, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv2d(24, 4, 1),
        )
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


@dataclass
class ColorGeometry:
    anchor: torch.Tensor
    fields: torch.Tensor
    partition: Partition
    features: torch.Tensor
    transform: torch.Tensor
    raw_rms: torch.Tensor
    projected_rms: torch.Tensor
    mass: torch.Tensor


def tonal_fields(anchor):
    brightness = (anchor * anchor.new_tensor((.25, .5, .25))[None, :, None, None]).sum(1, keepdim=True)
    window = 4. * brightness * (1. - brightness)
    contrast = window * (2. * brightness - 1.)
    color = anchor.new_tensor(((1., 1., 1.), (1., 1., 1.), (1., -1., 0.), (-.5, -.5, 1.)))
    features = torch.cat((window, contrast, window, window), dim=0)
    return features * color[:, :, None, None]


def projection(moment):
    """Remove the locally resolvable span of the three prior candidate differences."""
    g, b = moment[..., :3, :3], moment[..., :3, 3:]
    values, vectors = torch.linalg.eigh(g.double())
    cutoff = (.0005 * values[..., -1:]).clamp_min(2e-8)
    keep = values > cutoff
    reciprocal = torch.where(keep, values.clamp_min(2e-8).reciprocal(), 0.)
    u = ((vectors * reciprocal.unsqueeze(-2)) @ vectors.transpose(-1, -2)) @ b.double()
    t = torch.cat((-u, torch.eye(4, device=moment.device, dtype=u.dtype).expand(*u.shape[:-2], 4, 4)), dim=-2)
    covariance = t.transpose(-1, -2) @ moment.double() @ t
    raw, clean = (x.diagonal(dim1=-2, dim2=-1).clamp_min(0.).sqrt()
                  for x in (moment[..., 3:, 3:], covariance))
    return t.to(moment), raw.to(moment), clean.to(moment)


def field_moments(fields):
    return torch.einsum("kchw,lchw->klhw", fields, fields) / fields.shape[1]


@torch.no_grad()
def context(raw, restorers, heads, summaries=False):
    base = geometry(raw, restorers)
    part = base.partition
    mass = part.mass / part.mass.sum()
    center = mixture(heads["global", PRIOR_SEED], base.features, mass, global_mixture=True)
    weight, _ = residual_actions(heads["local_utility", PRIOR_SEED], base.features, base.gram, center, "local_utility")
    anchor = decoded(base, weight)
    fields = torch.cat((base.deltas, tonal_fields(anchor)))
    energy = field_moments(fields)
    gram = part.average(energy).permute(2, 3, 0, 1).unsqueeze(0)
    transform, raw_rms, clean_rms = projection(gram)
    cues = torch.cat((base.features, 2. * part.average(anchor)-1., 3. * clean_rms.permute(0,3,1,2)), dim=1)
    result = ColorGeometry(anchor, fields, part, cues, transform, raw_rms, clean_rms, mass)
    if summaries:
        return result, energy
    return result, base.base + base.deltas[1:2]


def code(net, features, transform, raw_rms, projected_rms, mass, mode):
    value = net(features).permute(0, 2, 3, 1)
    if mode == "global":
        value = (value * mass[None, ..., None]).sum(dim=(1, 2), keepdim=True).expand_as(value)
        basis_rms = (raw_rms.square() * mass[None, ..., None]).sum(dim=(1, 2), keepdim=True).sqrt()
    else:
        basis_rms = projected_rms if mode == "projected" else raw_rms
    amplitude = MAX_RMS * value.tanh()
    direction = amplitude / basis_rms.clamp_min(.055)
    if mode == "projected":
        a = (transform @ direction.unsqueeze(-1)).squeeze(-1)
    else:
        a = torch.cat((direction.new_zeros((*direction.shape[:-1], 3)), direction), dim=-1)
    return a, amplitude


def reveal(ctx, weights):
    pixel_weights = ctx.partition.expand(weights[0].permute(2,0,1)).unsqueeze(1)
    return ctx.anchor + (ctx.fields * pixel_weights).sum(0, keepdim=True)


def adjustment(ctx, model, mode):
    terms = (ctx.features, ctx.transform, ctx.raw_rms, ctx.projected_rms, ctx.mass)
    weights, _ = code(model, *terms, mode)
    return reveal(ctx, weights)


@torch.no_grad()
def examples(restorers, prior, smoke):
    records = defaultdict(list)
    counts = Counter()
    expected = dict(lm=128, rw=86, nh=36, rr=240)
    for domain, _, _, low, high in chain(training_pairs(), rr_records("train")):
        if smoke and counts[domain] == 3:
            continue
        pair = torch.cat((as_float(low), as_float(high)))
        reduced = resize(pair, view_size(pair, 576), antialias=True)
        for pair in (reduced,) if domain == "rr" else (reduced, reduced.pow(1.8)):
            x, y = pair.split(1)
            ctx, product = context(x, restorers, prior, summaries=True)
            part = ctx.partition
            row = {i: neighbor_axis(part.rows, i) for i in (-1,0,1)}
            col = {i: neighbor_axis(part.cols, i) for i in (-1,0,1)}
            metric = [(row[i].T @ product @ col[j]).div(part.mass).permute(2,3,0,1)[None] for i,j in OFFSETS]
            difference = y-ctx.anchor
            dot = part.average((ctx.fields * difference).mean(1)).permute(1,2,0).unsqueeze(0)
            mse = difference.square().mean().clamp_min(.0025)
            records[domain].append((ctx.features,ctx.transform,ctx.raw_rms,ctx.projected_rms,ctx.mass,torch.stack(metric),dot,mse))
        counts[domain] += 1
        if counts[domain] in (32, 64, 128, 192, expected[domain]):
            print(dict(bank=domain, photos=counts[domain], overall=sum(counts.values())), flush=True)
    if counts != (dict.fromkeys(expected,3) if smoke else expected):
        raise ValueError(f"Unexpected fit cohort {counts}")
    return tuple(records.values())


def objective(net, sample, mode):
    features, transform, raw, projected, mass, metric, z, mse = sample
    a, amplitude = code(net, features, transform, raw, projected, mass, mode)
    return (exact(metric, a, z) * mass).sum() / mse + .35*amplitude.square().mean()


def protocol():
    return dict(code_SHA256=digest(__file__), fit_data_SHA256=digest(TRAIN_CARD), rr_data_SHA256=digest(MANIFEST),
        prior_SHA256=digest(PREVIOUS / "completed.json"), prior_seed=PRIOR_SEED, modes=list(MODES),
        params=sum(p.numel() for p in ColorHead().parameters()), seeds=list(SEEDS), steps=STEPS,
        design="Compare global and pixel-varying tone controls and local residualization into the complement of existing "
               "candidate fields. Same four deterministic tone fields, 31 features, fit pairs, neural head and objective. "
               "Both raw and projected direction code is signed. Fit exact squared error after overlapping reconstruction; "
               "RGB clipping is applied during inference. The first-stage RGB system is frozen.")


def fit(smoke=False):
    configure_cuda()
    plan = protocol()
    lock = ROOT/"protocol.json"
    if not smoke:
        if lock.is_file() and read(lock)!=plan:
            raise ValueError("V59 fitting requires the exact registered source")
        if not lock.is_file():
            atomic_write(lock,plan)
    bank = examples(references(), fitted(), smoke)
    rows = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            path = ROOT/f"{mode}-{seed}.pth"
            note = path.with_suffix(".json")
            if not smoke and note.is_file():
                known = read(note)
                if known["weights_SHA256"] != digest(path) or known["protocol_SHA256"] != digest(lock):
                    raise ValueError("A previously trained RGB correction changed")
                rows.append(known)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed+59387)
            model = ColorHead().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0011, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00009)
            for step in range(2 if smoke else STEPS):
                optimizer.zero_grad(set_to_none=True)
                losses = [objective(model, pool[int(torch.randint(len(pool),(),generator=rng))], mode) for pool in bank]
                value = torch.stack(losses).mean()
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            row = dict(mode=mode,seed=seed,steps=step+1,loss=float(value.detach()))
            if not smoke:
                torch_save(path, {k:v.detach().cpu() for k,v in model.state_dict().items()})
                row.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
                atomic_write(note,row)
                rows.append(row)
            print(row,flush=True)
    if not smoke:
        atomic_write(ROOT/"completed.json",dict(runs=rows,updates=sum(r["steps"] for r in rows),
            protocol_SHA256=digest(lock),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),finished_utc=now()))


def models():
    card = read(ROOT/"completed.json")
    lock = ROOT/"protocol.json"
    if not matches_registered(read(lock), protocol()) or card["protocol_SHA256"] != digest(lock):
        raise ValueError("V59 checkpoints need their original source")
    result = {}
    for row in card["runs"]:
        key = row["mode"], row["seed"]
        path = ROOT/f"{key[0]}-{key[1]}.pth"
        if row["weights_SHA256"] != digest(path):
            raise ValueError("A trained RGB checkpoint changed")
        model = ColorHead().cuda()
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        result[key] = model.eval().requires_grad_(False)
    if set(result) != {(m,s) for m in MODES for s in SEEDS} or len(card["runs"])!=len(result):
        raise ValueError("V59 ablation set is incomplete")
    return result


def check():
    torch.manual_seed(59)
    d = torch.randn(2,3,4,80,7,dtype=torch.float64)
    d[0,...,2] = d[0,...,0]+d[0,...,1]
    d[1,0,0,...,:3] = 0.
    gram = d.transpose(-1,-2) @ d / d.shape[-2]
    t, before, after = projection(gram)
    value = d @ t
    leak = (d[...,:3].transpose(-1,-2) @ value / d.shape[-2]).abs().max().item()
    torch.testing.assert_close(after, value.square().mean(-2).sqrt())
    if leak > 1e-10 or (after > before+1e-10).any():
        raise ArithmeticError("The independent RGB candidate complement failed")
    print(dict(local_rank_cases=gram.numel()//49, residual_dot_max_error=leak),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit","smoke","check"))
    command=parser.parse_args().stage
    if command=="check":
        check()
    else:
        fit(command=="smoke")

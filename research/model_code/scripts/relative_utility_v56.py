"""Learn local candidate utility relative to a shared global RGB mixture."""
import argparse
from collections import Counter, defaultdict
from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.utils import set_seed
from scripts.cross_scale_rgb_v44 import resize
from scripts.decision_overlap_v55 import MANIFEST, ROOT as SOURCE, exact, rr_records
from scripts.exact_overlap_v54 import OFFSETS, matrix_blocks
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.overlap_quadratic_v53 import SEEDS, geometry, label, quadratic, references, simplex
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.region_restore_v48 import view_size
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save
from scripts.veil_response_v45 import TRAIN_CARD, training_pairs


ROOT = REPO / "results/relative-utility-v56"
ARMS = ("global", "local_direct", "residual_direct", "local_utility")
STEPS = 420
EPSILON = 1e-6


class MixtureHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(27, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(), nn.Conv2d(24, 4, 1)
        )
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def forward(self, features, center):
        hint = center.permute(0, 3, 1, 2).expand(-1, -1, *features.shape[-2:])
        return super().forward(torch.cat((features, hint), dim=1)).permute(0, 2, 3, 1)


def coefficients(head, features, mass, global_mixture):
    empty = features.new_zeros((features.shape[0], 1, 1, 3))
    logits = head(features, empty)
    if global_mixture:
        logits = (logits * mass[None, :, :, None]).sum(dim=(1, 2), keepdim=True)
    return logits.softmax(-1)[..., 1:]


def relative_geometry(gram, center):
    # The relative directions are all four existing candidates minus the learned mixture.
    basis = torch.eye(4, device=center.device, dtype=center.dtype)[:, 1:]
    offsets = basis - center.unsqueeze(-2)
    metric = offsets @ gram @ offsets.transpose(-1, -2)
    variance = metric.diagonal(dim1=-2, dim2=-1).clamp_min(0.)
    return offsets, metric, variance


def residual_mixture(center, offsets, weights):
    return center + (weights.unsqueeze(-1) * offsets).sum(-2)


def radial_update(slope, variance, smooth):
    """Select a candidate and minimize predicted error along its line from the center."""
    curvature = variance.clamp_min(EPSILON)
    step = (slope / curvature).clamp(0., 1.)
    gain = 2. * slope * step - variance * step.square()
    if smooth:
        temperature = .03 * variance.mean(-1, keepdim=True) + .0001
        probs = torch.cat((torch.zeros_like(gain[..., :1]), gain / temperature), dim=-1).softmax(-1)[..., 1:]
    else:
        index = gain.argmax(-1, keepdim=True)
        probs = torch.zeros_like(gain).scatter(-1, index, 1.)
    return probs * step


def quadratic_update(metric, slope, diagonal):
    variance = metric.diagonal(dim1=-2, dim2=-1).clamp_min(0.)
    shape = torch.diag_embed(variance) if diagonal else metric
    ridge = (.02 * variance.mean(-1, keepdim=True)).clamp_min(EPSILON)
    shape = shape + torch.diag_embed(ridge.expand_as(variance))
    step = shape.abs().sum(-1).amax(-1, keepdim=True).clamp_min(EPSILON).reciprocal()
    x = torch.zeros_like(slope)
    for _ in range(12):
        grad = (shape @ x.unsqueeze(-1)).squeeze(-1) - slope
        x = simplex(x - step * grad)
    return x


def residual_actions(head, features, gram, center, arm, smooth=False):
    offsets, metric, variance = relative_geometry(gram, center)
    logits = head(features, center)
    if arm == "residual_direct":
        probabilities = torch.cat((torch.zeros_like(logits[..., :1]), logits - 2.), dim=-1).softmax(-1)[..., 1:]
        return residual_mixture(center, offsets, probabilities), None
    projection = .22 * logits
    rms = variance.clamp_min(EPSILON).sqrt()
    slope = projection * rms
    if arm in ("utility_diagonal", "utility_full"):
        weights = quadratic_update(metric, slope, arm == "utility_diagonal")
    else:
        weights = radial_update(slope, variance, smooth)
    return residual_mixture(center, offsets, weights), (projection, offsets, rms)


def proposals(geom, bank):
    mass = geom.partition.mass / geom.partition.mass.sum()
    result = {}
    for seed in SEEDS:
        center = coefficients(bank["global", seed], geom.features, mass, global_mixture=True)
        result["global", seed] = center.expand(geom.gram.shape[:-1])
        result["local_direct", seed] = coefficients(
            bank["local_direct", seed], geom.features, mass, global_mixture=False
        )
        for arm in ("residual_direct", "local_utility", "utility_diagonal", "utility_full"):
            net = bank["residual_direct" if arm == "residual_direct" else "local_utility", seed]
            result[arm, seed], _ = residual_actions(net, geom.features, geom.gram, center, arm)
    return result


@torch.no_grad()
def fit_bank(restorers, smoke):
    bank = defaultdict(list)
    counts = Counter()
    stored = Counter()
    goals = dict(lm=128, rw=86, nh=36, rr=240)
    for domain, _, _, raw, clear in chain(training_pairs(), rr_records("train")):
        if smoke and counts[domain] == 3:
            continue
        pair = torch.cat((as_float(raw), as_float(clear)))
        pair = resize(pair, view_size(pair, 576), antialias=True)
        variants = (pair,) if domain == "rr" else (pair, pair.pow(1.8))
        for variant in variants:
            x, y = variant.split(1)
            geom = geometry(x, restorers)
            rhs, start = label(geom, y)
            matrices = matrix_blocks(geom)
            terms = torch.stack([matrices[step] for step in OFFSETS])
            mass = geom.partition.mass / geom.partition.mass.sum()
            bank[domain].append((geom.features, geom.gram, terms, rhs, start, mass))
            stored[domain] += 1
        counts[domain] += 1
        if counts[domain] in (32, 64, 128, 192, goals[domain]):
            print(dict(caching=domain, count=counts[domain], done=sum(counts.values())), flush=True)
    expected = dict.fromkeys(goals, 3) if smoke else goals
    if counts != expected:
        raise ValueError(f"Unexpected fitting pairs {counts}")
    print(dict(training_pairs=dict(counts), views=dict(stored)), flush=True)
    return tuple(bank.values())


def objective(net, sample, arm, global_head):
    features, gram, terms, rhs, start, mass = sample
    base_mse = (start * mass).sum()
    if arm in ("global", "local_direct"):
        weights = coefficients(net, features, mass, global_mixture=arm == "global")
        change = quadratic(gram, rhs, weights) if arm == "global" else exact(terms, weights, rhs)
        return (change * mass).sum() / base_mse.clamp_min(.0025)

    with torch.no_grad():
        center = coefficients(global_head, features, mass, global_mixture=True)
        at_center = quadratic(gram, rhs, center)
        global_mse = (base_mse + (at_center * mass).sum()).clamp_min(.0025)
    weights, signal = residual_actions(net, features, gram, center, arm, smooth=True)
    error = ((exact(terms, weights, rhs) - at_center) * mass).sum() / global_mse
    if signal is None:
        return error
    estimate, offsets, rms = signal
    with torch.no_grad():
        tangent = rhs - (gram @ center.unsqueeze(-1)).squeeze(-1)
        observed = (offsets @ tangent.unsqueeze(-1)).squeeze(-1) / rms
    alignment = F.smooth_l1_loss(estimate, observed, beta=.06, reduction="none")
    # Scale the per-direction label by the error units so domain groups have comparable weight.
    return error + .30 * (alignment.mean(dim=-1) * mass).sum() / global_mse.sqrt()


def input_protocol():
    return dict(
        implementation_SHA256=digest(__file__),
        weights_source_SHA256=digest(SOURCE / "completed.json"),
        source_manifest_SHA256=digest(TRAIN_CARD), rr_split_SHA256=digest(MANIFEST),
        seeds=list(SEEDS), arms=list(ARMS), steps=STEPS,
        optimization=dict(name="AdamW", lr=.0011, weight_decay=.01, cosine_min=.00009, clip_grad=2.5),
        parameters=sum(x.numel() for x in MixtureHead().parameters()),
        inputs="Reuse frozen candidate generators. Use the same 490 real training pairs and 740 views as V55.",
        rule="Fit global logits and a standalone spatial gate. Fit matching local residual heads from the frozen global mixture. "
             "Direct residual weights use exact reconstruction error. Utility also predicts the signed, RMS-normalized candidate projections "
             "relative to that global mixture and uses a line search along one direction at each region. "
             "Use same-weight diagonal and full quadratics at inference to isolate cross-candidate terms.",
        scopes="Use no score from V56 evaluation to select weights. Any optional validation shrinkage is selected on old validation and RRSHID val only."
    )


def load_head(path, recorded):
    if recorded["weights_SHA256"] != digest(path):
        raise ValueError(f"Mixture checkpoint differs from its log: {path.name}")
    head = MixtureHead().cuda()
    head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return head.eval().requires_grad_(False)


def fit(smoke=False):
    configure_cuda()
    protocol = input_protocol()
    lock = ROOT / "protocol.json"
    if not smoke:
        if lock.is_file() and read(lock) != protocol:
            raise ValueError("V56 changes a locked training protocol")
        if not lock.is_file():
            atomic_write(lock, protocol)
    samples = fit_bank(references(), smoke)
    finished = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        baseline = None
        for arm in ARMS:
            path = ROOT / f"{arm}-{seed}.pth"
            log = path.with_suffix(".json")
            if not smoke and log.is_file():
                row = read(log)
                if row["protocol_SHA256"] != digest(lock) or row["weights_SHA256"] != digest(path):
                    raise ValueError("A logged V56 run has changed")
                if arm == "global":
                    baseline = load_head(path, row)
                finished.append(row)
                continue
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed + 56741)
            head = MixtureHead().cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=.0011, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00009)
            for step in range(2 if smoke else STEPS):
                optimizer.zero_grad(set_to_none=True)
                losses = [objective(head, group[int(torch.randint(len(group), (), generator=generator))], arm, baseline)
                          for group in samples]
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.5, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
            if arm == "global":
                baseline = head.eval().requires_grad_(False)
            row = dict(arm=arm, seed=seed, steps=step+1, loss=float(loss.detach()))
            if not smoke:
                torch_save(path, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                row.update(weights_SHA256=digest(path), protocol_SHA256=digest(lock))
                atomic_write(log, row)
                finished.append(row)
            print(row, flush=True)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=finished, protocol_SHA256=digest(lock),
            updates=sum(row["steps"] for row in finished), gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__), finished_utc=now()))


def fitted():
    packet = read(ROOT / "completed.json")
    protocol = ROOT / "protocol.json"
    if packet["protocol_SHA256"] != digest(protocol) or read(protocol) != input_protocol():
        raise ValueError("V56 evaluator requires its frozen fit")
    bank = {(row["arm"], row["seed"]): load_head(ROOT / f"{row['arm']}-{row['seed']}.pth", row) for row in packet["runs"]}
    if set(bank) != {(a, s) for a in ARMS for s in SEEDS} or len(packet["runs"]) != len(bank):
        raise ValueError("Missing matched V56 checkpoints")
    return bank


def check_math():
    torch.manual_seed(55)
    d = torch.randn(2, 3, 5, 3, 3, dtype=torch.float64)
    g = (d.transpose(-1, -2) @ d) / 3.
    rhs = torch.randn(2, 3, 5, 3, dtype=d.dtype)
    center = torch.rand(2, 1, 1, 4, dtype=d.dtype).softmax(-1)[..., 1:]
    offsets, metric, variance = relative_geometry(g, center)
    slope = (offsets @ (rhs - (g @ center.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
    moves = radial_update(slope, variance, False)
    alpha = residual_mixture(center, offsets, moves)
    actual = quadratic(g, rhs, alpha) - quadratic(g, rhs, center)
    expected = -2. * (moves * slope).sum(-1) + ((metric @ moves.unsqueeze(-1)).squeeze(-1) * moves).sum(-1)
    step = torch.linspace(0., 1., 501, dtype=d.dtype)[:, None, None, None, None]
    brute = (variance * step.square() - 2. * slope * step).amin(dim=(0, -1))
    torch.testing.assert_close(actual, expected)
    if not ((actual <= brute + 1e-10).all() and (alpha >= -1e-12).all() and (alpha.sum(-1) <= 1.+1e-12).all()):
        raise ArithmeticError("A selected line edit is infeasible or loses to the line-search grid")
    print(dict(line_search_cases=actual.numel(), algebra_error=float((actual - expected).abs().amax())), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "smoke", "check"))
    args = parser.parse_args()
    if args.stage == "check":
        check_math()
    else:
        fit(args.stage == "smoke")

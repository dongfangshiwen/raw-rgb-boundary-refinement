"""Test the exact neighboring quadratic against the separable overlap bound."""
from collections import Counter, defaultdict
from itertools import product
from statistics import mean

import torch

from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_overlap_quadratic import POPULATION, calibrate, edge_score, samples, seams
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.overlap_quadratic import ROOT as LOCAL, SEEDS, decoded, fitted, geometry, label, preference, references, simplex, solve, upper_bound
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT = REPO / "results/exact-overlap-v54"
OFFSETS = tuple(product((-1, 0, 1), repeat=2))
MODES = dict(full_exact=0., half_penalty=.50)


def neighbor_axis(axis, direction):
    positions = torch.arange(axis.shape[1], device=axis.device) + direction
    exists = (positions >= 0) & (positions < axis.shape[1])
    return axis * axis[:, positions.clamp(0, axis.shape[1] - 1)] * exists


def matrix_blocks(context):
    """Only horizontal, vertical and diagonal neighboring hats have common support."""
    partition = context.partition
    residuals = context.deltas
    field = (residuals[:, None] * residuals[None, :]).mean(2)
    row = {i: neighbor_axis(partition.rows, i) for i in (-1, 0, 1)}
    column = {i: neighbor_axis(partition.cols, i) for i in (-1, 0, 1)}
    return {(i, j): ((row[i].T @ field @ column[j]) / partition.mass).permute(2, 3, 0, 1)[None] for i, j in OFFSETS}


def applies(blocks, a):
    return sum((part @ a.roll((-i, -j), dims=(1, 2)).unsqueeze(-1)).squeeze(-1)
               for (i, j), part in blocks.items())


def exact_solve(gram, links, linear, penalty):
    blocks = {ij: (1. - penalty) * block for ij, block in links.items()}
    diagonal = gram.diagonal(dim1=-1, dim2=-2)
    ridge = (.02 * diagonal.mean(-1, keepdim=True)).clamp_min(1e-6)
    blocks[0, 0] = blocks[0, 0] + penalty * gram + torch.diag_embed(ridge.expand_as(diagonal))
    # A row-sum bound controls the spectrum of the mass-weighted symmetric operator.
    scale = sum(h.abs().sum(-1) for h in blocks.values()).amax().clamp_min(1e-6)
    operator = torch.stack([blocks[k] for k in OFFSETS])
    coeff = torch.zeros_like(linear)
    for _ in range(48):
        neighbors = torch.stack([coeff.roll((-i, -j), dims=(1, 2)) for i, j in OFFSETS])
        gradient = (operator @ neighbors.unsqueeze(-1)).squeeze(-1).sum(0) - linear
        coeff = simplex(coeff - gradient / scale)
    return coeff


def oracle_objective(context, blocks, a, clean_dot):
    value = ((applies(blocks, a) - 2. * clean_dot) * a).sum(-1)[0]
    return (value * context.partition.mass).sum() / context.partition.mass.sum()


@torch.inference_mode()
def test():
    configure_cuda()
    cal = calibrate()
    ROOT.mkdir(parents=True, exist_ok=True)
    stamp = ROOT / "protocol.json"
    plan = dict(evaluator_SHA256=digest(__file__), source_SHA256=digest(LOCAL / "completed.json"),
        amplitude_SHA256=digest(LOCAL / "calibration.json"), registered_eval_SHA256=digest(LOCAL / "evaluation_protocol.json"),
        test_counts=POPULATION, common_amplitude=cal["scales"]["full"], penalty=MODES,
        aim="Use the frozen V53 learned alignment and identical candidate RGBs. Replace the local upper bound with the exact neighboring knot matrix, or half of the upper-bound penalty. No new ground truth enters fitting or selection. Lock the same amplitude for every comparison.",
        criteria="Report exact numerical energy identity, PSNR and grid-seam gradient error. Require a meaningful matched difference on a real held set before attributing a performance benefit to the cross-node terms.")
    if stamp.exists() and read(stamp) != plan:
        raise ValueError("A fixed overlap energy comparison changed")
    if not stamp.exists():
        atomic_write(stamp, plan)
    study = references()
    nets = fitted()
    rows, summary, by_group, examples, identity, counts = [], defaultdict(list), defaultdict(lambda: defaultdict(list)), [], defaultdict(list), Counter()
    amplitude = plan["common_amplitude"]
    for dataset, name, group, fog, clean, roi, condition in samples():
        x, y = as_float(fog), as_float(clean)
        ctx = geometry(x, study)
        blocks = matrix_blocks(ctx)
        target, _ = label(ctx, y)
        mask = roi.cuda() if roi is not None else None
        grid = seams(ctx, mask)
        baseline = area_psnr(ctx.base, y, mask)
        native_mse = (ctx.base-y).square().mean()
        learned = torch.cat([preference(nets["alignment", seed], ctx) for seed in SEEDS])
        trials = {"upper": solve(ctx.gram, learned)}
        trials.update({kind: exact_solve(ctx.gram, blocks, learned, ratio) for kind, ratio in MODES.items()})
        for i, seed in enumerate(SEEDS):
            measured, visual = {}, {}
            for kind, batch in trials.items():
                coeff = batch[i:i+1]
                rgb = decoded(ctx, coeff, amplitude)
                score = area_psnr(rgb, y, mask)
                error = edge_score(rgb-y, grid)
                true_energy = (rgb-y).square().mean() - native_mse
                exact = oracle_objective(ctx, blocks, amplitude*coeff, target)
                difference = abs(float(true_energy-exact))
                if difference > 2e-6:
                    raise ArithmeticError("The sparse overlapping quadratic is not numerically exact")
                margin = float(upper_bound(ctx, amplitude*coeff, target)-exact)
                identity[dataset].append(difference)
                row = dict(dataset=dataset, image=name, group=group, condition=condition, seed=seed, method=kind,
                    psnr=score, delta_db=score-baseline, seam=error, gap=margin, harm=score <= baseline-.10)
                rows.append(row)
                measured[kind], visual[kind] = row, rgb
                summary[dataset, kind].append(row)
            for kind in MODES:
                for metric in ("psnr", "seam", "harm"):
                    a, b = measured[kind][metric], measured["upper"][metric]
                    if a is not None and b is not None:
                        by_group[dataset, kind, metric][group].append(float(a)-float(b))
            if counts[dataset] == 0 and seed == SEEDS[0] and dataset in ("rrshid", "nt_haze", "realhyper_csnc"):
                pictures = [x, ctx.base, *(visual[n] for n in ("upper", "half_penalty", "full_exact")), y]
                examples.append((dataset + " / " + name, [to_rgb(t) for t in pictures],
                    ("Hazy input", "Anchor", "Local upper bound", "Half penalty", "Exact cross-node energy", "Clear reference")))
        counts[dataset] += 1
        if counts[dataset] % 130 == 0:
            print(dict(dataset=dataset, measured=counts[dataset]), flush=True)
    if counts != POPULATION:
        raise ValueError("The exact-overlap confirmation population changed")
    values = {"/".join(k): {m: mean(r[m] for r in v if r[m] is not None) for m in ("psnr", "delta_db", "seam", "gap", "harm")}
              for k, v in summary.items()}
    comparison = {"/".join(k): bootstrap({g: mean(v) for g, v in val.items()}) for k, val in by_group.items()}
    detail = ROOT / "individual.json"
    atomic_write(detail, rows)
    picture = ROOT / "predictions.png"
    draw(picture, examples)
    atomic_write(ROOT / "test.json", dict(counts=dict(counts), rows=len(rows), averages=values, versus_upper_bound=comparison,
        maximum_mse_identity_error={d: max(v) for d, v in identity.items()}, protocol_SHA256=digest(stamp),
        table_SHA256=digest(detail), panel_SHA256=digest(picture), finished_utc=now()))
    print(values, flush=True)


if __name__ == "__main__":
    test()

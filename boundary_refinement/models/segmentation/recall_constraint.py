"""Protect the relative ordering of sparse positive and verified negative scores."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.utils import set_seed
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.segmentation.false_positive_constraint import ROOT as V87, STEPS, TRACKED, flat_grad, loss_terms
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.segmentation.ranked_negatives import class_margin, flat_logits
from boundary_refinement.evaluation.semantic.resolution_interaction import DetailHead, DualScaleFeatures, SEEDS, batch, packets
from boundary_refinement.training.source_step import projected
from boundary_refinement.models.segmentation.sparse_recovery import DATA, ROOT as V86, edit, load as sparse_weights, make_sparse_pool, raw_models
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/recall-constraint-v91"
ARMS = ("project_zero_margin", "project_detached_quantile", "project_relative_quantile", "always_reduce_risk")
PRIMARY = "project_relative_quantile"
TEMPERATURE = 1.
MIN_POSITIVE = 8
BAND = (.10, .20)


def protocol():
    return dict(source_SHA256=digest(__file__), training_SHA256=digest(DATA / "fit-population.json"),
                prior_SHA256=digest(V86 / "completed.json"), projection_SHA256=digest(V87 / "completed.json"),
                seeds=list(SEEDS), steps=STEPS, arms=list(ARMS), active_class_ids=list(TRACKED),
                quantile_band=list(BAND), temperature=TEMPERATURE, min_positive_count=MIN_POSITIVE,
                design="Estimate the lower positive margin quantile from annotated positive pixels for each active "
                       "class. Approximate the negative exceedance rate at this point with a sigmoid on verified "
                       "dense negatives. Protect its first-order change under the sparse update only. Compare a "
                       "fixed argmax boundary, a stop-gradient positive quantile, and an equally sized correction "
                       "also applied when the sparse update already reduces the risk. Reuse V87 starting weights, "
                       "crop streams, optimizer and main losses. All configurations are fixed before rain evaluation.",
                primary="Traffic-light false positives per million at 85 percent recall against matched detached "
                        "and zero-margin constraints, V87 negative projection and strong raw baselines. "
                        "Zurich and Driving are repeated fog evaluations. New MUSES rain recordings test transfer.")


def positive_boundary(values):
    ordered = values.sort().values
    start = int(BAND[0] * len(ordered))
    stop = max(start + 1, ceil(BAND[1] * len(ordered)))
    return ordered[start:stop].mean()


def relative_exceedance(negative, positive, arm):
    if arm == "project_zero_margin":
        threshold = negative.new_zeros(())
    else:
        threshold = positive_boundary(positive)
        if arm == "project_detached_quantile":
            threshold = threshold.detach()
    return torch.sigmoid((negative - threshold) / TEMPERATURE).mean()


def recall_risk(logits, target, extra_logits, extra_target, arm):
    dense, sparse = flat_logits(logits), flat_logits(extra_logits)
    target, extra_target = target.flatten(), extra_target.flatten()
    terms, sizes = [], []
    for cls in TRACKED:
        pos = (extra_target == cls).nonzero().flatten()
        neg = ((target >= 0) & (target < 19) & (target != cls)).nonzero().flatten()
        sizes.append(len(pos) if len(pos) >= MIN_POSITIVE and len(neg) else 0)
        if not sizes[-1]:
            continue
        negative, _ = class_margin(dense[neg], cls)
        positive, _ = class_margin(sparse[pos], cls)
        terms.append(relative_exceedance(negative, positive, arm))
    value = torch.stack(terms).mean() if terms else (dense.sum() + sparse.sum()) * 0.
    return value, sizes


def fit_one(arm, seed, raw, initial, dense, sparse, weights, smoke):
    path = ROOT / f"{arm}-{seed}.pth"
    receipt = path.with_suffix(".json")
    checksum = None if smoke else digest(ROOT / "protocol.json")
    if not smoke and receipt.is_file():
        result = read(receipt)
        if result["weights_SHA256"] != digest(path) or result["protocol_SHA256"] != checksum:
            raise ValueError(f"Existing constraint checkpoint mismatch: {path.name}")
        return result
    set_seed(seed)
    model = DetailHead().cuda()
    model.load_state_dict(initial.state_dict(), strict=True)
    params = tuple(model.parameters())
    opt = torch.optim.SGD(params, lr=.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.003)
    dense_draw = torch.Generator().manual_seed(87001 + seed)
    sparse_draw = torch.Generator().manual_seed(87019 + seed)
    valid = {str(cls): 0 for cls in TRACKED}
    before, after = [], []
    n = 4 if smoke else STEPS
    for _ in range(n):
        views, labels = batch(*dense, dense_draw, "dense")
        extra, target = batch(*sparse, sparse_draw, "extra")
        extra = extra.cut(slice(1, 2))
        target = target[1:2]
        dense_loss, logits = loss_terms(model, raw, views, labels, weights)
        sparse_logits = raw_anchor(raw, extra) + edit(model, extra, "fusion_mixed")
        sparse_loss = .5 * F.cross_entropy(sparse_logits, target, weight=weights, ignore_index=255)
        estimate, support = recall_risk(logits, labels, sparse_logits, target, arm)
        for cls, size in zip(TRACKED, support, strict=True):
            valid[str(cls)] += int(size > 0)
        dense_grad = flat_grad(dense_loss, params, retain_graph=True)
        sparse_grad = flat_grad(sparse_loss, params, retain_graph=True)
        risk_grad = flat_grad(estimate, params)
        movement, dot = projected(-sparse_grad, risk_grad)
        if arm == "always_reduce_risk":
            movement = -sparse_grad - dot.abs() / risk_grad.dot(risk_grad).clamp_min(1e-18) * risk_grad
        before.append(dot)
        after.append(movement.dot(risk_grad))
        step = dense_grad - movement
        opt.zero_grad(set_to_none=True)
        for parameter, gradient in zip(params, step.split([p.numel() for p in params]), strict=True):
            parameter.grad = gradient.reshape_as(parameter)
        nn.utils.clip_grad_norm_(params, 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    result = dict(method=arm, seed=seed, steps=n, params=sum(p.numel() for p in params), class_active_steps=valid,
                  before_increases=int((torch.stack(before) > 1e-9).sum()),
                  after_increases=int((torch.stack(after) > 1e-7).sum()),
                  loss=float((dense_loss + sparse_loss).detach()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        result.update(weights_SHA256=digest(path), protocol_SHA256=checksum)
        atomic_write(receipt, result)
    print(result, flush=True)
    return result


def fit(smoke):
    configure_cuda()
    path, note = ROOT / "protocol.json", protocol()
    if not smoke:
        if path.is_file() and read(path) != note:
            raise ValueError("The recall-constraint settings changed")
        atomic_write(path, note)
    initial, raw, engine = sparse_weights(), raw_models(), DualScaleFeatures()
    dense, weights, rare = packets(engine, smoke)
    sparse, positions, _ = make_sparse_pool(engine, smoke)
    runs = [fit_one(m, s, raw["raw_detail", s], initial["fusion_dense", s], (dense, rare), (sparse, positions),
                    weights, smoke) for s in (SEEDS[:1] if smoke else SEEDS) for m in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=runs, total_updates=sum(x["steps"] for x in runs),
                     protocol_SHA256=digest(path), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), utc=now()))


def load():
    path, card = ROOT / "protocol.json", read(ROOT / "completed.json")
    if not matches_registered(read(path), protocol()) or digest(path) != card["protocol_SHA256"]:
        raise ValueError("The protected rank checkpoints do not match registration")
    bank = {}
    for row in card["runs"]:
        key = row["method"], row["seed"]
        file = ROOT / f"{key[0]}-{key[1]}.pth"
        net = DetailHead().cuda().eval().requires_grad_(False)
        if row["steps"] != STEPS or row["weights_SHA256"] != digest(file) or row["params"] != sum(
                p.numel() for p in net.parameters()):
            raise ValueError(f"Invalid protected rank checkpoint: {file.name}")
        net.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"), strict=True)
        bank[key] = net
    if set(bank) != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("Missing protected rank checkpoint")
    return bank


def check_math():
    positive = torch.linspace(-2., 3., 20)
    negative = torch.tensor([-2., 0., 2.])
    bias = torch.tensor(0., requires_grad=True)
    risk = relative_exceedance(negative + bias, positive + bias, PRIMARY)
    error, = torch.autograd.grad(risk, bias)
    assert float(error.abs()) < 1e-7
    detached = relative_exceedance(negative + bias, positive + bias, "project_detached_quantile")
    error, = torch.autograd.grad(detached, bias)
    assert float(error) > 0.
    torch.testing.assert_close(risk, relative_exceedance(negative - 7., positive - 7., PRIMARY))
    print("Joint margin shifts cancel and the detached control does not cancel.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("check", "smoke", "fit"))
    args = cli.parse_args()
    if args.part == "check":
        check_math()
    else:
        fit(args.part == "smoke")

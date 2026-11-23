"""Constrain positive-only adaptation using verified dense-background gradients."""
import argparse
from collections import Counter, defaultdict
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix
from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import paired_iou_intervals
from scripts.class_protected_residual_v81 import soft_counts
from scripts.contrastive_response_v85 import raw_anchor
from scripts.cross_scale_rgb_v44 import resize
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import aggregate
from scripts.resolution_interaction_v84 import DetailHead, DualScaleFeatures, SEEDS, add_global_cohort, batch, packets
from scripts.restoration_affinity_v75 import compress_scores
from scripts.source_step_v33 import projected
from scripts.sparse_recovery_v86 import ROOT as V86, edit, images, load as sparse_weights, make_sparse_pool, raw_models
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save
from scripts.weatherreasonseg_v85 import group_recall


ROOT = REPO / "results/false-positive-constraint-v87"
STEPS = 240
ARMS = ("dense_only", "plain_sparse", "total_loss_projection", "known_negative_projection")
PRIMARY = "known_negative_projection"
TRACKED = (6, 7, 14)


def registration():
    return dict(code_SHA256=digest(__file__), v86_SHA256=digest(V86 / "completed.json"),
                score_inputs={f: digest(V86 / f"{f}-test.json") for f in ("held", "external", "weather")},
                seeds=list(SEEDS), steps=STEPS, arms=list(ARMS), tracked=list(TRACKED),
                protocol="Continue each seed's common V86 dense-fusion model with the same SGD budget. Keep two dense "
                         "crops and one additional crop. The extra crop is dense for the dense-only control and uses "
                         "WeatherReasonSeg training positives for other arms. Compare the unmodified sparse update, "
                         "a PCGrad-style projection relative to all dense loss, and the identical projection relative "
                         "only to false positive probability mass on independently labelled dense negative pixels. "
                         "Preserve the dense update and never treat an unannotated sparse pixel as a negative. "
                         "Compare recall and precision on the previous real-fog cohorts. These are repeated evaluations.")


def loss_terms(model, anchor, views, labels, weights):
    delta = edit(model, views, "fusion_mixed")
    logits = raw_anchor(anchor, views) + delta
    hit, union, count, _ = soft_counts(logits, labels)
    loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
    loss = loss + .8 * (1. - hit[count > 0] / union[count > 0].clamp_min(1e-5)).mean()
    return loss + .015 * delta.square().mean(), logits


def false_positive_mass(logits, dense_labels, sparse_labels):
    present = F.one_hot(sparse_labels.clamp(0, 18), 19) * (sparse_labels < 19)[..., None]
    active = present.any(dim=(0, 1, 2))
    ids = torch.arange(19, device=logits.device)[None, :, None, None]
    known_negative = (dense_labels[:, None] != ids) & (dense_labels[:, None] < 19)
    count = known_negative.sum((0, 2, 3))
    mean_probability = (logits.softmax(1) * known_negative).sum((0, 2, 3)) / count.clamp_min(1)
    active = active & (count > 0)
    return (mean_probability * active).sum() / active.sum().clamp_min(1)


def flat_grad(loss, params, retain_graph=False):
    return nn.utils.parameters_to_vector(torch.autograd.grad(loss, params, retain_graph=retain_graph))


def fit_one(mode, seed, raw, initial, dense, partial, weights, smoke):
    checkpoint = ROOT / f"{mode}-{seed}.pth"
    receipt = checkpoint.with_suffix(".json")
    plan = digest(ROOT / "protocol.json") if not smoke else None
    if not smoke and receipt.is_file():
        saved = read(receipt)
        if digest(checkpoint) != saved["weights_SHA256"] or plan != saved["protocol_SHA256"]:
            raise ValueError(f"Unexpected constraint checkpoint {checkpoint.name}")
        return saved
    set_seed(seed)
    model = DetailHead().cuda()
    model.load_state_dict(initial.state_dict(), strict=True)
    parameters = tuple(model.parameters())
    opt = torch.optim.SGD(parameters, lr=.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.003)
    gen = torch.Generator().manual_seed(87001 + seed)
    extra_gen = torch.Generator().manual_seed(87019 + seed)
    n = 4 if smoke else STEPS
    corrections, protected = [], []
    for _ in range(n):
        views, mask = batch(*dense, gen, "dense")
        extra_pair, extra_mask = batch(*(dense if mode == "dense_only" else partial), extra_gen, "extra")
        extra, target = extra_pair.cut(slice(1, 2)), extra_mask[1:2]
        dense_loss, logits = loss_terms(model, raw, views, mask, weights)
        extra_logits = raw_anchor(raw, extra) + edit(model, extra, "fusion_mixed")
        additional = .5 * F.cross_entropy(extra_logits, target, weight=weights, ignore_index=255)
        protect = mode == PRIMARY
        dense_grad = flat_grad(dense_loss, parameters, retain_graph=protect)
        extra_grad = flat_grad(additional, parameters)
        if mode in ("total_loss_projection", PRIMARY):
            reference = flat_grad(false_positive_mass(logits, mask, target), parameters) if protect else dense_grad
            displacement, violation = projected(-extra_grad, reference)
            corrections.append(violation)
            protected.append(displacement.dot(reference))
            extra_grad = -displacement
        step_grad = dense_grad + extra_grad
        opt.zero_grad(set_to_none=True)
        for variable, gradient in zip(parameters, step_grad.split([p.numel() for p in parameters]), strict=True):
            variable.grad = gradient.reshape_as(variable)
        nn.utils.clip_grad_norm_(parameters, 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    result = dict(method=mode, seed=seed, steps=n, params=sum(p.numel() for p in parameters),
                  constrained_updates=int((torch.stack(corrections) > 1e-9).sum()) if corrections else 0,
                  residual_increases=int((torch.stack(protected) > 1e-7).sum()) if protected else 0,
                  loss=float((dense_loss + additional).detach()))
    if not smoke:
        torch_save(checkpoint, {name: value.detach().cpu() for name, value in model.state_dict().items()})
        result.update(weights_SHA256=digest(checkpoint), protocol_SHA256=plan)
        atomic_write(receipt, result)
    print(result, flush=True)
    return result


def train(smoke):
    configure_cuda()
    cfg, path = registration(), ROOT / "protocol.json"
    if not smoke:
        if path.is_file() and read(path) != cfg:
            raise ValueError("The projected update scope changed")
        atomic_write(path, cfg)
    raw = raw_models()
    initial = sparse_weights()
    engine = DualScaleFeatures()
    dense, weight, rare = packets(engine, smoke)
    sparse, positions, _ = make_sparse_pool(engine, smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    rows = [fit_one(m, s, raw["raw_detail", s], initial["fusion_dense", s], (dense, rare), (sparse, positions), weight, smoke)
            for s in seeds for m in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=rows, protocol_SHA256=digest(path), gpu=torch.cuda.get_device_name(),
                     total_updates=sum(r["steps"] for r in rows), torch=str(torch.__version__), utc=now()))


def load():
    models, ledger = {}, read(ROOT / "completed.json")
    path = ROOT / "protocol.json"
    if read(path) != registration() or digest(path) != ledger["protocol_SHA256"]:
        raise ValueError("Registered constraint does not match the frozen models")
    for row in ledger["runs"]:
        name = row["method"], row["seed"]
        path = ROOT / f"{name[0]}-{name[1]}.pth"
        if digest(path) != row["weights_SHA256"] or row["steps"] != STEPS:
            raise ValueError(f"Changed constraint checkpoint {path.name}")
        net = DetailHead().cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        models[name] = net
    if set(models) != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("Incomplete paired projection")
    return models


def focus(rows):
    collected = defaultdict(lambda: np.zeros((19, 19), dtype=np.int64))
    for r in rows:
        collected[r["dataset"], r["method"], r["seed"]] += r["counts"]
    per_class = defaultdict(list)
    for (ds, m, _), cm in collected.items():
        for k in TRACKED:
            tp, positive, predicted = int(cm[k, k]), int(cm[k].sum()), int(cm[:, k].sum())
            per_class[f"{ds}/{m}/{k}"].append(dict(recall=100. * tp / max(1, positive),
                precision=100. * tp / max(1, predicted), iou=100. * tp / max(1, predicted + positive - tp),
                false_positive_per_million=1e6 * (predicted - tp) / max(1, int(cm.sum()) - positive),
                known_positive_pixels=positive))
    return {k: {name: mean(seed[name] for seed in detail) for name in detail[0]} for k, detail in per_class.items()}


@torch.inference_mode()
def score(part):
    configure_cuda()
    net, raw, prior = load(), raw_models(), sparse_weights()
    engine = DualScaleFeatures()
    rows, counts = [], Counter()
    for dataset, key, group, _, rgb, mask in images(part):
        field, _, _ = engine(rgb)
        for seed in SEEDS:
            initial = raw_anchor(raw["raw_detail", seed], field)
            heads = {a: net[a, seed] for a in ARMS}
            heads["before_adaptation"] = prior["fusion_dense", seed]
            for name, model in heads.items():
                z = initial + edit(model, field, "fusion_mixed")
                label = resize(z, mask.shape).argmax(1)[0]
                rows.append(dict(dataset=dataset, id=key, group=group, seed=seed, method=name,
                                 counts=confusion_matrix(label, mask, 19).tolist()))
        counts[dataset] += 1
        if sum(counts.values()) % 40 == 0:
            print(dict(projected=part, evaluated=sum(counts.values())), flush=True)
    if dict(counts) != read(V86 / f"{part}-test.json")["counts"]:
        raise ValueError("Projection used a mismatched evaluation population")
    full = add_global_cohort(rows)
    pairs = [(PRIMARY, m) for m in (*ARMS, "before_adaptation") if m != PRIMARY]
    result = dict(positive_recall=group_recall(rows)) if part == "weather" else dict(
        metrics=aggregate(full), comparisons=paired_iou_intervals(full, pairs, "method", "counts"), focused=focus(full))
    ledger = ROOT / f"{part}-individual.json.gz"
    compress_scores(ledger, rows)
    atomic_write(ROOT / f"{part}-test.json", dict(**result, counts=dict(counts), rows_SHA256=digest(ledger),
                 model_SHA256=digest(ROOT / "completed.json"), utc=now()))
    if part == "weather":
        print({m: round(v["mean_recall"], 4) for m, v in result["positive_recall"]["all"].items()}, flush=True)
    else:
        print({m: round(v["miou"], 4) for m, v in result["metrics"].items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("smoke", "fit", "held", "external", "weather"))
    args = cli.parse_args()
    if args.part in ("smoke", "fit"):
        train(args.part == "smoke")
    else:
        score(args.part)

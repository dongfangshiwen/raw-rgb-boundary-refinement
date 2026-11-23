"""Compare hard and confuser-matched pixel ranking for partially labelled fog."""
import argparse

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.utils import set_seed
from scripts.contrastive_response_v85 import raw_anchor
from scripts.false_positive_constraint_v87 import ROOT as V87, STEPS, TRACKED, loss_terms
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.resolution_interaction_v84 import DetailHead, DualScaleFeatures, SEEDS, batch, packets
from scripts.sparse_recovery_v86 import DATA, ROOT as V86, edit, load as sparse_weights, make_sparse_pool, raw_models
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/ranked-negatives-v89"
ARMS = ("rank_hard", "rank_margin", "rank_permuted", "rank_competitor", "rank_local")
PRIMARY = "rank_competitor"
PAIR_COUNT = 32
NEAR_POOL = 32
HARD_POOL = 128
RANK_WEIGHT = .2


def protocol():
    return dict(source_SHA256=digest(__file__), sparse_data_SHA256=digest(DATA / "fit-population.json"),
                parent_SHA256=digest(V86 / "completed.json"), projection_SHA256=digest(V87 / "completed.json"),
                arms=list(ARMS), seeds=list(SEEDS), steps=STEPS, classes=list(TRACKED), positives_per_class=PAIR_COUNT,
                near_pool=NEAR_POOL, hard_pool=HARD_POOL, rank_weight=RANK_WEIGHT,
                design="Use the V87 dense and sparse crop draws, common V86 dense-trained heads, SGD schedule and "
                       "base losses. Add the same softplus pair-ranking term to each arm. Freeze the initial model "
                       "to select pairs, using verified target positives and pixels known to belong to other classes. "
                       "Compare hard negatives, the nearest anchor score, nearest score with a permuted competitor, "
                       "and the nearest qualifying pixel sharing the original strongest alternative class. "
                       "A secondary control prefers qualifying pixels in the same sparse crop. Ignored targets never enter pairs.",
                primary="Traffic-light false positives per million at 85 percent recall on the existing Zurich "
                        "and complete Driving cohorts, compared with hard, margin and permutation controls. "
                        "Additional recall points, fine-only subset, traffic-sign/truck rankings, full mIoU, "
                        "and the original-image baseline show the scope. Previous cohorts are reused.",
                dual_output_renderer_fixed=True)


def flat_logits(values):
    return values.movedim(1, -1).reshape(-1, values.shape[1])


def class_margin(values, cls):
    best, index = values.topk(2, dim=1)
    predicted = index[:, 0] == cls
    return values[:, cls] - torch.where(predicted, best[:, 1], best[:, 0]), torch.where(
        predicted, index[:, 1], index[:, 0])


def select_pairs(anchor_margin, alternative, target, cls, sparse_start, positive_generator, control_generator, arm):
    positive = ((target == cls) & (torch.arange(target.numel(), device=target.device) >= sparse_start)).nonzero().flatten()
    negative = ((target < 19) & (target >= 0) & (target != cls)).nonzero().flatten()
    if not len(positive) or not len(negative):
        return positive[:0], negative[:0]
    positive = positive[torch.randperm(len(positive), generator=positive_generator, device=target.device)[:PAIR_COUNT]]
    if arm == "rank_hard":
        hard = anchor_margin[negative].topk(min(len(negative), HARD_POOL)).indices
        index = torch.randint(len(hard), (len(positive),), generator=control_generator, device=target.device)
        return positive, negative[hard[index]]
    alternative_id = alternative[negative]
    if arm == "rank_permuted":
        alternative_id = alternative_id[torch.randperm(len(negative), generator=control_generator, device=target.device)]
    gap = (anchor_margin[positive, None] - anchor_margin[negative]).abs()
    closest = gap.topk(min(len(negative), NEAR_POOL), largest=False).indices
    if arm == "rank_margin":
        return positive, negative[closest[:, 0]]
    matches = alternative_id[closest] == alternative[positive, None]
    priority = matches.long()
    if arm == "rank_local":
        priority = priority + (matches & (negative[closest] >= sparse_start)).long()
    chosen = closest.gather(1, priority.argmax(dim=1, keepdim=True)).squeeze(1)
    return positive, negative[chosen]


def ranking_loss(current, initial, dense, partial, positive_generator, control_generator, arm):
    labels = torch.cat((dense.flatten(), partial.flatten()))
    sparse_start = dense.numel()
    loss, diagnostics = [], []
    for cls in TRACKED:
        with torch.no_grad():
            margin, alternative = class_margin(initial, cls)
            positive, negative = select_pairs(margin, alternative, labels, cls, sparse_start, positive_generator,
                                              control_generator, arm)
        if not len(positive):
            diagnostics.append(current.new_zeros(4))
            continue
        margin_now, _ = class_margin(current, cls)
        loss.append(F.softplus(margin_now[negative] - margin_now[positive]).mean())
        with torch.no_grad():
            diagnostics.append(torch.stack((margin.new_tensor(len(positive)),
                                 (alternative[positive] == alternative[negative]).sum().float(),
                                 (negative >= sparse_start).sum().float(),
                                 (margin[positive] - margin[negative]).abs().sum())))
    return torch.stack(loss).mean() if loss else current.sum() * 0., torch.stack(diagnostics)


def train_one(arm, seed, raw, initial, dense, sparse, weights, smoke):
    path = ROOT / f"{arm}-{seed}.pth"
    receipt = path.with_suffix(".json")
    lock = digest(ROOT / "protocol.json") if not smoke else None
    if not smoke and receipt.is_file():
        saved = read(receipt)
        if digest(path) != saved["weights_SHA256"] or lock != saved["protocol_SHA256"]:
            raise ValueError(f"Changed ranking checkpoint {path.name}")
        return saved
    set_seed(seed)
    model = DetailHead().cuda()
    model.load_state_dict(initial.state_dict(), strict=True)
    opt = torch.optim.SGD(model.parameters(), lr=.025)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.003)
    dense_draw = torch.Generator().manual_seed(87001 + seed)
    sparse_draw = torch.Generator().manual_seed(87019 + seed)
    positive_draw = torch.Generator(device="cuda").manual_seed(89001 + seed)
    negative_draw = torch.Generator(device="cuda").manual_seed(89023 + seed)
    joined = torch.zeros((len(TRACKED), 4), device="cuda")
    n = 4 if smoke else STEPS
    for _ in range(n):
        views, mask = batch(*dense, dense_draw, "dense")
        extra, target = batch(*sparse, sparse_draw, "extra")
        extra, target = extra.cut(slice(1, 2)), target[1:2]
        dense_loss, logits = loss_terms(model, raw, views, mask, weights)
        extra_anchor = raw_anchor(raw, extra)
        sparse_logits = extra_anchor + edit(model, extra, "fusion_mixed")
        extra_loss = .5 * F.cross_entropy(sparse_logits, target, weight=weights, ignore_index=255)
        with torch.no_grad():
            start = raw_anchor(raw, views) + edit(initial, views, "fusion_mixed")
            start_extra = extra_anchor + edit(initial, extra, "fusion_mixed")
            fixed = torch.cat((flat_logits(start), flat_logits(start_extra)))
        values = torch.cat((flat_logits(logits), flat_logits(sparse_logits)))
        rank, stats = ranking_loss(values, fixed, mask, target, positive_draw, negative_draw, arm)
        joined += stats
        loss = dense_loss + extra_loss + RANK_WEIGHT * rank
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    info = {str(cls): dict(pairs=int(pairs), same_competitor=int(same), same_crop=int(local),
                           mean_anchor_gap=gap / max(1., pairs))
            for cls, (pairs, same, local, gap) in zip(TRACKED, joined.tolist(), strict=True)}
    result = dict(method=arm, seed=seed, steps=n, params=sum(p.numel() for p in model.parameters()),
                  pairs=info, loss=float(loss.detach()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        result.update(weights_SHA256=digest(path), protocol_SHA256=lock)
        atomic_write(receipt, result)
    print(result, flush=True)
    return result


def train(smoke):
    configure_cuda()
    path, scope = ROOT / "protocol.json", protocol()
    if not smoke:
        if path.is_file() and read(path) != scope:
            raise ValueError("Ranking settings changed after registration")
        atomic_write(path, scope)
    base, raw, engine = sparse_weights(), raw_models(), DualScaleFeatures()
    dense, weights, rare = packets(engine, smoke)
    sparse, positions, _ = make_sparse_pool(engine, smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    rows = [train_one(arm, seed, raw["raw_detail", seed], base["fusion_dense", seed], (dense, rare),
                      (sparse, positions), weights, smoke) for seed in seeds for arm in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=rows, total_updates=sum(r["steps"] for r in rows),
                     protocol_SHA256=digest(path), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def load():
    path, receipt = ROOT / "protocol.json", read(ROOT / "completed.json")
    if read(path) != protocol() or digest(path) != receipt["protocol_SHA256"]:
        raise ValueError("Ranking protocol or parents changed")
    models = {}
    for row in receipt["runs"]:
        key = row["method"], row["seed"]
        file = ROOT / f"{key[0]}-{key[1]}.pth"
        net = DetailHead().cuda().eval().requires_grad_(False)
        if row["steps"] != STEPS or row["weights_SHA256"] != digest(file) or row["params"] != sum(
                p.numel() for p in net.parameters()):
            raise ValueError(f"Ranking weight/shape mismatch: {file.name}")
        net.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"), strict=True)
        models[key] = net
    if set(models) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("Ranking result set is incomplete")
    return models


def check_selection():
    m = torch.tensor([1.215, 1.2, 1.3, 1.21, 1.21])
    labels = torch.tensor([7, 6, 2, 255, 6])
    alternative = torch.tensor([3, 0, 5, 5, 5])
    rng = torch.Generator().manual_seed(89)
    p, n = select_pairs(m, alternative, labels, 6, 4, rng, rng, PRIMARY)
    assert p.tolist() == [4] and n.tolist() == [2]
    p, n = select_pairs(m, alternative, labels, 6, 4, rng, rng, "rank_margin")
    assert p.tolist() == [4] and n.tolist() == [0]
    print("Positive origin, competitor priority and ignore-mask exclusion passed.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("check", "smoke", "fit"))
    args = cli.parse_args()
    if args.part == "check":
        check_selection()
    else:
        train(args.part == "smoke")

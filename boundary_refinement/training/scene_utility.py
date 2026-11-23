"""Train a pre-call PSNR-gain gate with controlled same-scene ranking supervision."""
import argparse
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.training.conditional_fusion import CALL_SHARE, bedde_split, city_manifest, features, load_state
from boundary_refinement.training.conditional_fusion import paired_geometry, relative_psnr, training_rows
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.training.preview_routing import fit_ridge, group_quantile, group_weights
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.staging.stage_bedde import RECEIPT as BEDDE
from boundary_refinement.data.staging.stage_lmhaze import ROOT, TRAIN_CARD, samples
from boundary_refinement.data.staging.stage_preview_training import STAGED
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ALPHA = .5
SEEDS = (31003, 31013, 31033)
STEPS = 240
WIDTH = 48
POINTWISE = "utility_mlp"
CROSS = "cross_scene_rank"
MAIN = "same_scene_rank"
MLP_ARMS = (POINTWISE, CROSS, MAIN)
RIDGE_ARMS = ("prior_ridge", "joint_ridge")
ARMS = (*RIDGE_ARMS, *MLP_ARMS)
PACKAGE = ROOT / "weights.pth"
PROTOCOL = ROOT / "protocol.json"
FIT = ROOT / "fit.json"


class UtilityHead(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(n_features, WIDTH), nn.SiLU(),
                                     nn.Linear(WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, 1))
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return (self.project(x) + self.linear(x))[:, 0]


@torch.no_grad()
def cache_rows(rows, forward, label):
    x, utility, identity = [], [], []
    for dataset, scene, group, hazy, clear, roi, role in rows:
        fog, target = as_float(hazy), as_float(clear)
        anchor, challenger = (net(fog).clamp(0,1) for net in forward)
        mask = None if roi is None else roi[None, None].cuda()
        x.append(features(fog, anchor)["anchor"][0])
        stats = paired_geometry(anchor, challenger, target, mask)
        utility.append(relative_psnr(stats, stats.new_tensor(ALPHA)))
        identity.append((dataset, scene, group, role))
        if len(identity) % 70 == 0:
            print(dict(cache=label, pairs=len(identity)), flush=True)
    x, utility = torch.stack(x), torch.stack(utility)
    if not x.isfinite().all() or not utility.isfinite().all():
        raise ValueError("Nonfinite training descriptor or utility")
    if len(set(identity)) != len(identity):
        raise ValueError("Duplicate gate training row")
    return dict(x=x.cpu(), utility=utility.cpu(), rows=identity)


def cached(label, rows, forward, sources):
    package, receipt = ROOT / (label+"_features.pth"), ROOT / (label+"_features.json")
    fingerprint = dict(sources=sources, code_SHA256=digest(__file__), alpha=ALPHA)
    if receipt.exists():
        card = read(receipt)
        if card["fingerprint"] != fingerprint or card["SHA256"] != digest(package):
            raise ValueError("Stored features changed or use a different protocol")
    else:
        data = cache_rows(rows, forward, label)
        torch_save(package, data)
        card = dict(fingerprint=fingerprint, SHA256=digest(package),
                    roles=dict(Counter(r[3] for r in data["rows"])), count=len(data["rows"]))
        atomic_write(receipt, card)
    return torch.load(package, map_location="cuda", weights_only=True), card


def prior_rows():
    cities = city_manifest()
    yield from training_rows(read(STAGED), cities, bedde_split(cities))


def cache_seed(forward, authors):
    card, fit = cached("prior_training", prior_rows(), forward, dict(staged=digest(STAGED), bedde=digest(BEDDE), authors=authors))
    if fit["roles"] != {"fit": 247, "calibrate": 70}:
        raise ValueError("Prior training groups changed")
    return card, fit


def identifiers(rows, field, device):
    codes = {name: i for i, name in enumerate(sorted({r[field] for r in rows}))}
    return torch.tensor([codes[r[field]] for r in rows], device=device)


def balanced_weights(rows, device):
    groups = identifiers(rows, 2, device)
    domains = identifiers(rows, 0, device)
    weight = group_weights(groups, 1.)
    mass = weight.new_zeros(int(domains.max())+1).scatter_add_(0, domains, weight)
    weight = weight / mass[domains]
    return weight / weight.sum()


def paired_index(rows, target, seed):
    """Match the count and true-gain-gap distribution of cross-scene comparisons."""
    unique = identifiers(rows, 2, target.device)
    lmhaze = torch.tensor([i for i,r in enumerate(rows) if r[0] == "lmhaze"], device=target.device)
    left, right = lmhaze[torch.triu_indices(len(lmhaze), len(lmhaze), 1, device=target.device)]
    delta = target[left] - target[right]
    gaps = delta.abs()
    bins = torch.bucketize(gaps, gaps.new_tensor([.3, .6, 1.2, 2.4, 4.8]))
    same = unique[left] == unique[right]
    valid = gaps > .15
    generator = torch.Generator(device=target.device).manual_seed(seed)
    choices = ([], [], [])
    for level in range(6):
        inside = (valid & same & (bins == level)).nonzero(as_tuple=True)[0]
        outside = (valid & ~same & (bins == level)).nonzero(as_tuple=True)[0]
        n = len(inside)
        if n and not len(outside):
            raise ValueError("No cross-scene control for a gain difference observed within a scene")
        if n:
            choices[0].append(inside)
            for subset in choices[1:]:
                drawn = torch.randperm(len(outside), generator=generator, device=target.device)[:n]
                if len(drawn) < n:
                    drawn = torch.randint(len(outside), (n,), generator=generator, device=target.device)
                subset.append(outside[drawn])
    inside, outer_a, outer_b = (torch.cat(index) for index in choices)
    if not len(inside):
        raise ValueError("No valid matched ranking pairs")
    return {key: (left[index], right[index], (delta[index]/1.5).sigmoid())
            for key,index in (("within",inside),("across_a",outer_a),("across_b",outer_b))}


def ranking_loss(estimate, row):
    left, right, target = row
    return F.binary_cross_entropy_with_logits((estimate[left]-estimate[right])*2, target)


def fit_heads(x, target, rows, weights):
    center = weights @ x
    scale = (weights @ (x-center).square()).sqrt().clamp_min(.003)
    data = ((x-center)/scale).clamp(-10,10)
    # Predictions and rank gaps are optimized in normalized dB units.
    utility = target.clamp(-12,9)/3
    stored = {method: [] for method in MLP_ARMS}
    ledger = []
    for seed in SEEDS:
        pairs = paired_index(rows, target, seed)
        for method in MLP_ARMS:
            set_seed(seed)
            model = UtilityHead(data.shape[1]).to(data.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.04)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.0003)
            for _ in range(STEPS):
                optimizer.zero_grad(set_to_none=True)
                pred = model(data)
                loss = (weights * F.smooth_l1_loss(pred, utility, reduction="none", beta=.5)).sum()
                if method != POINTWISE:
                    second = "within" if method == MAIN else "across_b"
                    loss = loss + .25*(ranking_loss(pred, pairs["across_a"])+ranking_loss(pred, pairs[second]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4, error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
            stored[method].append({key: value.detach().cpu() for key,value in model.state_dict().items()})
            row = dict(seed=seed, arm=method, steps=STEPS, final_loss=loss.item(),
                       params=sum(v.numel() for v in model.parameters()),
                       comparisons=[len(v[0]) for v in pairs.values()] if method != POINTWISE else [])
            print(row, flush=True)
            ledger.append(row)
    return dict(center=center.cpu(), scale=scale.cpu(), weights=stored), ledger


def models(blob, device="cuda"):
    center = blob["heads"]["center"].to(device)
    scale = blob["heads"]["scale"].to(device)
    nets = {}
    for name, states in blob["heads"]["weights"].items():
        heads = []
        for weights in states:
            model = UtilityHead(len(center)).to(device)
            model.load_state_dict(weights, strict=True)
            heads.append(model.eval().requires_grad_(False))
        nets[name] = heads
    return dict(ridge={n: {k: v.to(device) for k,v in p.items()} for n,p in blob["ridge"].items()},
                center=center, scale=scale, nets=nets, thresholds=blob.get("thresholds", {}))


@torch.inference_mode()
def predict(state, x):
    answer = {name: (((x-p["center"])/p["scale"])@p["coefficients"]+p["intercept"])[:,0]
              for name,p in state["ridge"].items()}
    data = ((x-state["center"])/state["scale"]).clamp(-10,10)
    answer.update({name: 3*torch.stack([net(data) for net in nets]).mean(0) for name,nets in state["nets"].items()})
    return answer


@torch.inference_mode()
def restore(image, forward, state, method=MAIN):
    first = forward[0](image).clamp(0,1)
    score = predict(state, features(image, first)["anchor"])[method].item()
    called = score > state["thresholds"][method]
    if called:
        restored = torch.lerp(first, forward[1](image).clamp(0,1), ALPHA)
        return restored, dict(pgl=True, alpha=ALPHA, predicted_improvement=score)
    return first, dict(pgl=False, alpha=0., predicted_improvement=score)


def load():
    receipt = read(FIT)
    if digest(PACKAGE) != receipt["weights_SHA256"] or digest(PROTOCOL) != receipt["protocol_SHA256"]:
        raise ValueError("Scene utility fit no longer matches")
    blob = torch.load(PACKAGE, map_location="cpu", weights_only=True)
    return models(blob), receipt


def train(cache_only):
    configure_cuda()
    ROOT.mkdir(parents=True, exist_ok=True)
    forward, authors = experts()
    prior, prior_card = cache_seed(forward, authors)
    if cache_only:
        print(prior_card["roles"], flush=True)
        return
    new, new_card = cached("lmhaze", samples({"fit","calibrate"}), forward, dict(manifest=digest(TRAIN_CARD), authors=authors))
    _, previous = load_state()
    design = dict(source_SHA256=digest(__file__), alpha=ALPHA, arms=list(ARMS), main=MAIN,
                  sources={"old_features":prior_card,"lmhaze_features":new_card,"previous_gate":previous["weights_SHA256"]},
                  authors=authors, seeds=list(SEEDS), steps=STEPS, sharpness_margin_db=.15, call_share=CALL_SHARE,
                  target="True RGB PSNR improvement of a fixed 50/50 mixture of MCT and PGL-Net-T relative to MCT.",
                  comparison="Regression-only MLP or equal architecture and updates with pairs from LMHaze. Both ranking variants have the same number and bins of target-gap comparisons. The within-scene variant replaces half of cross-scene comparisons with pairs sharing an exact clear image. NH, RW2AH and BeDDE contribute pointwise gains. At inference only foggy RGB and the already computed MCT output are used.",
                  split="Use previous V29 fit/calibration groups, plus LMHaze scene-disjoint fit/calibration. Author LMHaze Test scenes evaluate frozen models. V30 and older metrics are diagnostic.",
                  evaluation="Primary uses the same post-call 50/50 rule and exactly 30% calls. Fixed threshold is an independent online comparison. Report same-scene score concordance on all comparisons with true gap above 0.15 dB. Test labels never choose parameters.",
                  threshold="Maximum of zero, group-weighted 70th percentile of calibrated score, and group-weighted 80th percentile of score minus actual gain.")
    if PROTOCOL.exists() and read(PROTOCOL) != design:
        raise ValueError("Scene utility training protocol differs")
    if not PROTOCOL.exists():
        atomic_write(PROTOCOL, design)
    if FIT.exists():
        return load()
    rows = prior["rows"] + new["rows"]
    x = torch.cat((prior["x"], new["x"]))
    u = torch.cat((prior["utility"], new["utility"]))
    fit = torch.tensor([v[3] == "fit" for v in rows], device="cuda")
    fit_prior = torch.tensor([v[0] != "lmhaze" and v[3] == "fit" for v in rows], device="cuda")
    fit_rows = [r for r in rows if r[3] == "fit"]
    w = balanced_weights(fit_rows, "cuda")
    ridge_target = u.clamp(-12,9)[:,None]
    ridge = dict(prior_ridge=fit_ridge(x[fit_prior], ridge_target[fit_prior],
                                    balanced_weights([r for r in fit_rows if r[0] != "lmhaze"], "cuda")),
                  joint_ridge=fit_ridge(x[fit], ridge_target[fit], w))
    heads, ledger = fit_heads(x[fit], u[fit], fit_rows, w)
    blob = dict(ridge=ridge, heads=heads)
    cal_groups = identifiers([r for r in rows if r[3] == "calibrate"],2,"cuda")
    calibration = {}
    for n, values in predict(models(blob), x[~fit]).items():
        budget_cut = group_quantile(values, cal_groups, 1-CALL_SHARE).item()
        residual_cut = group_quantile(values-u[~fit], cal_groups, .8).item()
        calibration[n] = dict(budget_cut=budget_cut, residual_cut=residual_cut,
                              threshold=max(0., budget_cut, residual_cut))
    blob["thresholds"] = {name: card["threshold"] for name,card in calibration.items()}
    torch_save(PACKAGE, blob)
    receipt = dict(protocol_SHA256=digest(PROTOCOL),weights_SHA256=digest(PACKAGE),
                   fit_pairs=int(fit.sum()),calibration_pairs=int((~fit).sum()),
                   domain_fit_counts=dict(Counter(r[0] for r in fit_rows)), calibration=calibration,
                   updates=ledger, optimizer_updates=len(ledger)*STEPS, gpu=torch.cuda.get_device_name(),
                   torch=str(torch.__version__), finished_utc=now())
    atomic_write(FIT, receipt)
    print({k:v for k,v in receipt.items() if k != "updates"}, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-prior",action="store_true")
    train(parser.parse_args().cache_prior)


if __name__ == "__main__":
    main()

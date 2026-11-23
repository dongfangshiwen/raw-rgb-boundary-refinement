"""Estimate when class-weighted subpixel colors approximate a labelled class prototype."""
from boundary_refinement.data.source_integrity import matches_registered
from hashlib import sha256

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.models.operators import donor_colors as previous
from boundary_refinement.models.operators import subpixel_transport as base
from boundary_refinement.data.provenance import REPO, digest, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/prototype-risk-v97-owner"
RUNS = ("constant_proto", "local_proto", "local_ce")
FIXED = ("mean", "fixed", "soft", "rolled")
SEEDS, STEPS = base.SEEDS, base.STEPS


class PrototypeGate(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        if method == "constant_proto":
            self.bias = nn.Parameter(torch.tensor(np.log(4.), dtype=torch.float32))
        else:
            self.layers = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 1))
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.constant_(self.layers[-1].bias, np.log(4.))

    def forward(self, cues):
        if self.method == "constant_proto":
            return self.bias.sigmoid().expand(cues.shape[0], 1)
        return self.layers(cues).sigmoid()


def cell_data(z, guides):
    h, w = z.shape[-2:]
    fixed = previous.color_weights(z, "dual_core")
    soft = (fixed - .2 / 16.) / .8
    values = z[0].topk(2, dim=0).values
    # Match the donor renderer's tie convention, including quantized training logits.
    owner = z.argmax(1).reshape(-1, 1)
    gap = (values[0] - values[1]).reshape(-1, 1)
    nearest = F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False).argmax(1)
    agreement = (base.pack(nearest, h, w)[..., 0] == owner).float().mean(1, keepdim=True)
    # Each row stores raw RGB followed by restored RGB.
    child = base.pack(torch.cat((guides[0], guides[1])), h, w).float() / 255.
    mean = child.mean(1)
    diff = (soft[..., None] * (child - mean[:, None])).sum(1)
    spread = (child - mean[:, None]).square().mean(1).reshape(-1, 2, 3).mean(-1).sqrt()
    displacement = diff.square().reshape(-1, 2, 3).mean(-1).sqrt()
    cues = torch.cat((gap.clamp_max(12.) / 6., agreement, soft.amax(-1, keepdim=True),
                      1. / (16. * soft.square().sum(-1, keepdim=True)), 6. * spread, 12. * displacement), -1)
    return dict(owner=owner, soft=soft, fixed=fixed, child=child, mean=mean, diff=diff, cues=cues)


def weights(method, data, net=None):
    fixed = data["fixed"]
    if method == "fixed":
        return fixed
    if method == "mean":
        return torch.full_like(fixed, 1./16.)
    if method == "soft":
        return data["soft"]
    if method == "rolled":
        return fixed.reshape(-1, 4, 4).roll((2, 2), dims=(1, 2)).reshape(-1, 16)
    a = net(data["cues"])
    return (1. - a) / 16. + a * data["soft"]


def membership(data, target, h, w):
    labels = base.pack(target[None], h, w)[..., 0]
    owns = (labels == data["owner"]).float()
    count = owns.sum(-1)
    valid = (labels < 19).all(-1)
    proto = (data["child"] * owns[..., None]).sum(1) / count.clamp_min(1.)[:, None]
    return owns, count, valid, proto


def prototype_loss(a, data, target, h, w):
    _, count, valid, proto = membership(data, target, h, w)
    inside = torch.zeros((h, w), device=target.device, dtype=torch.bool)
    inside[2:-2, 2:-2] = True
    valid = valid & inside.flatten()
    # Exclude absent owners. Pure cells provide a known zero-shift target.
    mixed = valid & (count > 0) & (count < 16)
    pure = valid & (count == 16)
    difference = data["mean"] + a * data["diff"] - proto
    pixel = difference.square().mean(-1)
    return 1000. * (.75 * (pixel*mixed).sum() / mixed.sum().clamp_min(1)
                    + .25 * (pixel*pure).sum() / pure.sum().clamp_min(1))


def seg_loss(logits, truth):
    inside = (..., slice(8, -8), slice(8, -8))
    target = truth[inside]
    weight = (target < 19).float() * (1. + 3. * boundary_band(truth)[inside] +
                                      2. * ((target == 6) | (target == 7) | (target == 14)))
    error = F.cross_entropy(logits[inside], target[None], ignore_index=255, reduction="none")[0]
    return (weight * error).sum() / weight.sum().clamp_min(1.)


def loss(method, net, transport, sample):
    z, guides, truth = sample
    data = cell_data(z, guides)
    if method == "local_ce":
        stencil = previous.prepare(guides, z, "dual_core", weighting=weights(method, data, net))
        offset = base.prediction("dual_free", transport, z, stencil)
        return seg_loss(F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False) + offset, truth)
    return prototype_loss(net(data["cues"]), data, truth, *z.shape[-2:])


def register(*, frozen=False):
    lock = ROOT / "protocol.json"
    plan = dict(code=digest(__file__), frozen=digest(previous.ROOT / "completed.json"), split=digest(base.DATA / "fit.json"),
                held_selection=digest(held.SCENE_INDEX), learned=list(RUNS), interventions=list(FIXED),
                seeds=list(SEEDS), steps=STEPS, sampler="V92 exact two crops, group-uniform from 96 images in nine fog recordings.",
                architecture="Freeze each V94 dual_core transport. Shrink the softmax donor toward the plain mean using "
                             "a scalar per cell. Match local_proto and local_ce capacity and initialization; global gate "
                             "and fixed strengths 0, .8, 1 plus a spatial roll of the fixed weights are controls.",
                primary="direct truth-owner mass and RAW/RESTORED RGB RMS error to the empirical owner mean in fully "
                        "labelled mixed cells containing the predicted owner, plus native boundary mIoU. New MUSES "
                        "recordings first, prior Fine Driving, Zurich and VKITTI fog separately. Compare trained gate "
                        "to fixed and the equally sized CE-only gate. Publish whole-frame and thin-class effects. "
                        "The purity and prototype targets are evaluation-only outside the nine fit recordings.")
    if frozen:
        if not matches_registered(read(lock), plan):
            raise ValueError("Frozen model protocol changed")
        return digest(lock)
    if lock.exists() and read(lock) != plan:
        raise ValueError("Prototype protocol changed")
    if not lock.exists():
        atomic_write(lock, plan)
    return digest(lock)


def train_one(method, seed, pool, transport, lock):
    file = ROOT / f"{method}-{seed}.pth"
    receipt = file.with_suffix(".json")
    if receipt.exists():
        r = read(receipt)
        if r["weights"] != digest(file) or r["protocol"] != lock:
            raise ValueError("Prototype gate checkpoint mismatch")
        return r
    set_seed(seed)
    net = PrototypeGate(method).cuda()
    rng = np.random.default_rng(92001 + seed)
    opt = torch.optim.AdamW(net.parameters(), lr=.012, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.0012)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        objective = sum(loss(method, net, transport, base.train_crop(pool, rng, SEEDS.index(seed), focus, step))
                        for focus in (False, True)) / 2.
        objective.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        schedule.step()
    torch_save(file, {k: v.detach().cpu() for k, v in net.state_dict().items()})
    row = dict(method=method, seed=seed, weights=digest(file), protocol=lock, loss=float(objective.detach()),
               steps=STEPS, params=sum(p.numel() for p in net.parameters()),
               sampler=sha256(str(rng.bit_generator.state).encode()).hexdigest())
    atomic_write(receipt, row)
    print(method, seed, row["loss"], flush=True)
    return row


def fitted():
    card = read(ROOT / "completed.json")
    if register(frozen=True) != card["protocol"] or len(card["runs"]) != len(RUNS)*len(SEEDS):
        raise ValueError("Incomplete prototype gate training")
    models = {}
    for r in card["runs"]:
        method, seed = r["method"], r["seed"]
        path = ROOT / f"{method}-{seed}.pth"
        if r["weights"] != digest(path) or r["protocol"] != card["protocol"]:
            raise ValueError("Changed prototype weights")
        net = PrototypeGate(method).cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path, weights_only=True))
        models[method, seed] = net
    return models


if __name__ == "__main__":
    configure_cuda()
    lock = register()
    pool = base.train_frames(base.DualScaleFeatures(), base.anchors(), False)
    other = previous.load()
    rows = [train_one(m, s, pool, other["dual_core", s], lock) for s in SEEDS for m in RUNS]
    if any(len({r["sampler"] for r in rows if r["seed"] == s}) != 1 for s in SEEDS):
        raise ValueError("Gate crops differ")
    atomic_write(ROOT / "completed.json", dict(runs=rows, protocol=lock, steps=STEPS*len(rows),
                 torch=str(torch.__version__), gpu=torch.cuda.get_device_name()))

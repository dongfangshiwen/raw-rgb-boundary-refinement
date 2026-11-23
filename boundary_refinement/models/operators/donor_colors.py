"""Estimate donor colors using the coarse semantic class's within-cell confidence."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from hashlib import sha256

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.data.staging import stage_vkitti_fog as virtual
from boundary_refinement.models.operators import subpixel_transport as parent
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/semantic-subpixel-v94"
ARMS = ("raw_core", "dual_core", "tone_core", "roll_core", "center_core")
PRIMARY = "dual_core"
SEEDS, STEPS = parent.SEEDS, parent.STEPS


def protocol():
    return dict(code_SHA256=digest(__file__), v92_SHA256=digest(parent.ROOT / "completed.json"),
                vkitti_selection_SHA256=digest(virtual.LOCK), arms=list(ARMS), seeds=list(SEEDS), steps=STEPS,
                training="Reuse V92 initial parameters, crops, loss, learning rate, updated parameter count, capacity "
                "and three frozen V84 anchors. No transport column constraint. Change only the donor color statistic.",
                intervention="Calculate the local dominant class margin against its strongest competitor at each "
                "of the 16 children using the frozen bilinear 19-way logits. Subtract the local mean, divide by "
                "local standard deviation floored at 0.5 logits and apply softmax with coefficient 1.5. Estimate "
                "RGB representative colors with 80% class-weighted plus 20% uniform weights. Compare whole-cell mean, "
                "a spatial phase roll of the same weights, and geometric center weights. Compare raw and tone-only guides.",
                primary_endpoints="Boundary-band mIoU comparing dual_core with dual_free, roll_core, center_core, "
                "raw_core, JBU, and the actual trained DySample operator. Main old fog cohort is Driving Fine. "
                "Fresh supplemental fog test is Virtual KITTI 2, restricted to the ten mapped classes in five simulated "
                "drives. An independent restoration effect requires an advantage over raw_core and tone_core. "
                "Report full mIoU, traffic-light IoU and same-recall FPR as secondary real-set checks.")


def color_weights(logits, mode):
    h, w = logits.shape[-2:]
    if mode == "center_core":
        _, _, distance = parent.geometry(logits.device)
        return (.2 / 16. + .8 * (-8. * distance[:, 4]).softmax(0)).expand(h * w, -1)
    dense = F.interpolate(logits, scale_factor=4., mode="bilinear", align_corners=False)
    owner = F.interpolate(logits.argmax(1, keepdim=True).float(), scale_factor=4., mode="nearest").long()
    values, classes = dense.topk(2, dim=1)
    alternative = torch.where(classes[:, :1] == owner, values[:, 1:], values[:, :1])
    margin = parent.pack((dense.gather(1, owner) - alternative)[0], h, w)[..., 0]
    norm = (margin - margin.mean(-1, keepdim=True)) / margin.std(-1, keepdim=True, unbiased=False).clamp_min(.5)
    weight = .2 / 16. + .8 * (1.5 * norm).softmax(-1)
    if mode == "roll_core":
        weight = weight.reshape(-1, 4, 4).roll((2, 2), dims=(1, 2)).reshape(-1, 16)
    return weight


def prepare(guides, logits, mode, template=None, weighting=None):
    grid = logits.shape[-2:]
    if template is None:
        template = parent.guide_stencil(guides[:0], logits.argmax(1)[0])
    idx, neighbor, base, distance, _ = template
    weight = color_weights(logits, mode) if weighting is None else weighting
    second = {"raw_core": 0, "tone_core": 2}.get(mode, 1)
    first_two = []
    for source in dict.fromkeys((0, second)):
        child = parent.pack(guides[source], *grid).float() / 255.
        donor = (child * weight[..., None]).sum(1)
        error = (child[idx, :, None] - donor[neighbor][:, None]).square().mean(-1, keepdim=True)
        scale = torch.tensor(parent.RANGES, device=logits.device)
        first_two.append(1. - (-.5 * error / scale.square()).exp())
    if second == 0:
        first_two *= 2
    return idx, neighbor, base, distance, first_two


def loss(model, mode, example):
    z, guides, truth = example
    stencil = prepare(guides, z, mode)
    offset = parent.prediction("dual_free", model, z, stencil)
    y = F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False) + offset
    inside = (..., slice(8, -8), slice(8, -8))
    target = truth[inside]
    weight = (target < 19).float() * (1. + 3. * boundary_band(truth)[inside] +
                                      2. * ((target == 6) | (target == 7) | (target == 14)))
    loss_map = F.cross_entropy(y[inside], target[None], ignore_index=255, reduction="none")[0]
    return (loss_map * weight).sum() / weight.sum().clamp_min(1.)


def fit_one(mode, seed, frames):
    lock = digest(ROOT / "protocol.json")
    path = ROOT / f"{mode}-{seed}.pth"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        row = read(receipt)
        if lock != row["protocol_SHA256"] or digest(path) != row["weights_SHA256"]:
            raise ValueError("Prototype checkpoint changed")
        return row
    set_seed(seed)
    net = parent.ColorTransport().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.035, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.0035)
    rng = np.random.default_rng(92001 + seed)
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        objective = sum(loss(net, mode, parent.train_crop(frames, rng, SEEDS.index(seed), f, step))
                        for f in (False, True)) / 2.
        objective.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
    torch_save(path, {k: v.cpu() for k, v in net.state_dict().items()})
    row = dict(method=mode, seed=seed, params=sum(p.numel() for p in net.parameters()), steps=STEPS,
               loss=float(objective.detach()), sampler_end=sha256(str(rng.bit_generator.state).encode()).hexdigest(),
               weights_SHA256=digest(path), protocol_SHA256=lock)
    atomic_write(receipt, row)
    print(row, flush=True)
    return row


def fit():
    configure_cuda()
    lock = ROOT / "protocol.json"
    cfg = protocol()
    if lock.exists() and read(lock) != cfg:
        raise ValueError("Prototype follow-up protocol changed")
    atomic_write(lock, cfg)
    pool = parent.train_frames(parent.DualScaleFeatures(), parent.anchors(), False)
    rows = [fit_one(m, s, pool) for s in SEEDS for m in ARMS]
    atomic_write(ROOT / "completed.json", dict(runs=rows, total_updates=len(rows) * STEPS, protocol_SHA256=digest(lock),
                 gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), utc=now()))


def load():
    card = read(ROOT / "completed.json")
    plan = ROOT / "protocol.json"
    if not matches_registered(read(plan), protocol()) or digest(plan) != card["protocol_SHA256"]:
        raise ValueError("Follow-up protocol mismatch")
    nets = {}
    for r in card["runs"]:
        a, s = r["method"], r["seed"]
        path = ROOT / f"{a}-{s}.pth"
        if digest(path) != r["weights_SHA256"] or r["steps"] != STEPS:
            raise ValueError("Bad prototype checkpoint")
        net = parent.ColorTransport().cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        nets[a, s] = net
    if set(nets) != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Incomplete prototype training")
    return nets


def check():
    configure_cuda()
    torch.manual_seed(941)
    z = torch.randn(1, 19, 5, 8, device="cuda")
    guides = torch.randint(256, (4, 3, 20, 32), dtype=torch.uint8, device="cuda")
    direct, rolled = (color_weights(z, a) for a in (PRIMARY, "roll_core"))
    assert torch.equal(direct.sort().values, rolled.sort().values)
    assert (direct.sum(-1) - 1.).abs().max() < 5e-7
    assert torch.allclose(color_weights(torch.ones_like(z), PRIMARY), torch.full_like(direct, 1 / 16))
    stencil = parent.guide_stencil(guides, z.argmax(1)[0])
    uniform = prepare(guides, z, PRIMARY, stencil, torch.full_like(direct, 1 / 16))
    for i in range(2):
        assert torch.allclose(stencil[-1][i], uniform[-1][i], atol=1e-6)
    print("Uniform donor, spatial permutation and constant semantic margin checks passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "fit"))
    args = parser.parse_args()
    check()
    if args.mode == "fit":
        fit()

"""Fit a shared optical field for parallel RGB and semantic refinement."""
import argparse
from itertools import islice
import time

import torch
from torch import nn
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.author_pgl_v25 import pgl_rgb
from scripts.coarse_semantics_v39 import head_features, rgb_checkpoint
from scripts.crop_context_v32 import originals
from scripts.evaluate_real_semantics_v22 import load_segmenter
from scripts.haze_response_v35 import linear_rgb
from scripts.kitti_fog_v40 import MANIFEST, ROOT, load, render
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.source_replay_v32 import PROTOCOL as REAL_SOURCE
from scripts.stage_lmhaze_v31 import TRAIN_CARD
from scripts.structure_transfer_v32 import SEEDS, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


# The two main factors are the inverse-scattering RGB proposal and the
# same-pixel visibility field in the semantic branch.
ARMS = ("direct", "inverse_only", "visibility_only", "joint", "shifted")
MAIN = "joint"
STEPS = 384
WIDTH = 32


class SharedHazeField(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(9, WIDTH, 3, padding=1), nn.GroupNorm(4, WIDTH), nn.SiLU(),
                                     nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.GroupNorm(4, WIDTH), nn.SiLU())
        self.optics = nn.Conv2d(WIDTH, 4, 1)
        self.photo = nn.Conv2d(WIDTH, 7, 1)
        self.project = nn.Conv2d(256, 16, 1)
        self.semantic = nn.Sequential(nn.Conv2d(WIDTH + 16, WIDTH, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(WIDTH, 20, 1))
        with torch.no_grad():
            self.optics.weight.zero_()
            self.optics.bias.zero_()
            self.optics.bias[0] = 2.4
            self.photo.weight.zero_()
            self.photo.bias.zero_()
            self.photo.bias[6] = -1.
            self.semantic[-1].weight.zero_()
            self.semantic[-1].bias.zero_()

    def forward(self, image, anchor, feats, logits, mode):
        size = logits.shape[-2:]
        low = F.adaptive_avg_pool2d(torch.cat((image, anchor), dim=1), size)
        raw, base = low.chunk(2, dim=1)
        h = self.encoder(torch.cat((2 * raw - 1, 2 * base - 1, 3 * (base - raw)), dim=1))
        optical = self.optics(h).float()
        t = .12 + .88 * optical[:, :1].sigmoid()
        air = .50 + .48 * optical[:, 1:].mean((-2, -1), keepdim=True).sigmoid()
        proposal = self.photo(h).float()
        photo = F.interpolate(proposal, image.shape[-2:], mode="bilinear", align_corners=False)
        left, right, blend = photo.split((3, 3, 1), dim=1)
        left = .28 * left.tanh()
        right = .28 * right.tanh()
        if mode in ("inverse_only", MAIN, "shifted"):
            transmission = F.interpolate(t, image.shape[-2:], mode="bilinear", align_corners=False)
            linear = ((linear_rgb(image) - (1. - transmission) * air) / transmission).clamp(0., 1.)
            curve = 1.055 * linear.clamp_min(.0031308).pow(1. / 2.4) - .055
            inverse = torch.where(linear <= .0031308, 12.92 * linear, curve)
            right = right + (inverse - anchor).clamp(-.45, .45)
        rgb = (anchor + left.lerp(right, blend.sigmoid())).clamp(0., 1.)
        semantics = self.semantic(torch.cat((h, self.project(feats)), dim=1)).float()
        delta, gate = semantics.split((19, 1), dim=1)
        visibility = (.25 + .75 * (1. - t) / .56).clamp(max=1.)
        if mode == "shifted":
            visibility = visibility.roll(tuple(v // 2 for v in size), dims=(-2, -1))
        elif mode in ("direct", "inverse_only"):
            visibility = visibility.mean((-2, -1), keepdim=True)
        return rgb, logits + 3. * delta.tanh() * gate.sigmoid() * visibility, t, air


@torch.no_grad()
def encode(image, restorer, segmenter, mean, std):
    anchor = pgl_rgb(restorer, image).clamp(0., 1.)
    feats, logits = head_features(segmenter, image, mean, std)
    return image.float(), anchor.float(), feats, logits


def to_cache(packet):
    return tuple(x.half() for x in packet)


def restore_real(image, target):
    scale = 512 / max(image.shape[-2:])
    dims = tuple(round(v * scale) for v in image.shape[-2:])
    return F.interpolate(torch.stack((image, target)).float() / 255., dims, mode="bilinear",
                         align_corners=False, antialias=True)


@torch.no_grad()
def caches(restorer, model, mean, std, smoke):
    rows = [r for r in load() if r["role"] == "fit"]
    pairs = []
    for i, row in enumerate(rows[:4] if smoke else rows, 1):
        clear = row["rgb"][None].cuda().float() / 255.
        distance = row["distance"][None].cuda().float()
        view, t, air = render(clear, distance, "fit", row["id"])
        x = to_cache(encode(view, restorer, model, mean, std))
        truth = row["label"][None, None].cuda().float()
        native = F.interpolate(truth, x[-1].shape[-2:], mode="nearest-exact")[0, 0].long()
        pairs.append((x, clear, native, t, air))
        if i % 29 == 0:
            print(dict(cached_kitti=i), flush=True)
    lm = training_scenes()
    rgbs = ((r["frames"][index], r["clear"]) for r in lm for index in (1, 6))
    real = [[], []]
    for i, source in enumerate((rgbs, (tuple(v) for v in originals()))):
        for fog, clear in islice(source, 2 if smoke else None):
            x, y = restore_real(fog, clear)
            item = to_cache(encode(x[None], restorer, model, mean, std))
            real[i].append((item, y[None].half()))
        print(dict(real_cache=i, items=len(real[i])), flush=True)
    return pairs, real


def prediction(model, inputs, mode):
    image, anchor, feats, logits = inputs
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(image.float(), anchor.float(), feats, logits.float(), mode)


def rgb_loss(rgb, clear):
    d = rgb - clear
    return d.abs().mean() + .15 * (d.diff(dim=-1).abs().mean() + d.diff(dim=-2).abs().mean())


def synthetic_loss(pred, clear, label, t, air):
    rgb, scores, transmission, light = pred
    target = label[None].expand(len(rgb), -1, -1)
    semantic = F.cross_entropy(scores, target, ignore_index=255, reduction="sum") / (target != 255).sum().clamp_min(1)
    optics = (transmission - F.adaptive_avg_pool2d(t, transmission.shape[-2:])).abs().mean()
    optics = optics + .40 * (light[1:] - air[1:]).abs().mean()
    return rgb_loss(rgb, clear) + .32 * semantic + .45 * optics


def fit(smoke=False):
    configure_cuda()
    restorer, restoration = rgb_checkpoint()
    segmenter, mean, std, card = load_segmenter()
    capacity = sum(x.numel() for x in SharedHazeField().parameters())
    plan = dict(
        aim="Test a shared dense optical field, with one independent inverse-scattering factor for RGB and one aligned visibility modulation factor for the segmentation residual.",
        arms=list(ARMS), primary=MAIN, parameters=capacity, steps=STEPS, seeds=list(SEEDS),
        anchors=dict(segformer=card, rgb_SHA256=restoration),
        dependency={str(p.relative_to(REPO)): digest(p) for p in (MANIFEST, TRAIN_CARD, REAL_SOURCE,
                    REPO / "scripts/kitti_fog_v40.py", REPO / "scripts/shared_haze_field_v40.py")},
        training="Each update uses the same three synthetic views of a KITTI scene with paired RGB, 19-class labels, transmission and global airlight. Add one aligned real RGB pair, alternating LMHaze and RW2AH. Use a small raw-logit preservation loss on these images without semantic labels. Both anchor networks are frozen. Equal capacity, same optimizer and identical examples for all methods.",
        factors="All arms use optical supervision and a common architecture. Direct uses two unconstrained RGB proposals and image-mean optical weighting for semantics. Inverse-only tests the added inverse-scattering proposal. Visibility-only tests spatial rather than image-average optical weighting. Joint uses both. Shifted globally rolls the optical weighting to control for the strength distribution.",
        deploy="Each raw view goes through each pretrained anchor once at the shared evaluation resolution. Fusion uses raw segmentation features and a spatial optical field derived from original and restored RGB. The output contract is restored RGB and integer semantic labels.",
        scope="Use only the registered train groups from KITTI, LMHaze and RW2AH. KITTI fog is synthetic. Evaluate actual two real RGB conditions and real fog semantics independently; no semantic labels from Foggy Zurich or Foggy Driving Fine enter training. No post-test checkpoint selection.",
    )
    protocol = ROOT / "protocol.json"
    if not smoke:
        if protocol.exists() and read(protocol) != plan:
            raise ValueError("The shared field protocol changed")
        if not protocol.exists():
            atomic_write(protocol, plan)
    synthetic, real = caches(restorer, segmenter, mean, std, smoke)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in ARMS:
            path = ROOT / f"{mode}-{seed}.pth"
            receipt = path.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if digest(path) != row["checkpoint_SHA256"] or digest(protocol) != row["protocol_SHA256"]:
                    raise ValueError("A shared field checkpoint changed")
                ledger.append(row)
                continue
            set_seed(seed)
            model = SharedHazeField().cuda().train()
            opt = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00008)
            rng = torch.Generator().manual_seed(seed + 40043)
            start = time.perf_counter()
            for step in range(1, (2 if smoke else STEPS) + 1):
                opt.zero_grad(set_to_none=True)
                a, clear, labels, t, air = synthetic[int(torch.randint(len(synthetic), (), generator=rng))]
                bank = real[step % 2]
                b, clean = bank[int(torch.randint(len(bank), (), generator=rng))]
                src = prediction(model, a, mode)
                rgb, logits, _, _ = prediction(model, b, mode)
                preserve = F.kl_div(logits.log_softmax(1), b[-1].float().softmax(1),
                                    reduction="none").sum(1).mean()
                objective = synthetic_loss(src, clear, labels, t, air) + .75 * rgb_loss(rgb, clean) + .10 * preserve
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                opt.step()
                sched.step()
            result = dict(arm=mode, seed=seed, updates=step, parameters=capacity, loss=objective.item(),
                          seconds=round(time.perf_counter() - start, 2))
            print(result, flush=True)
            if not smoke:
                torch_save(path, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                row = dict(result, file=str(path), checkpoint_SHA256=digest(path),
                           protocol_SHA256=digest(protocol), finished_utc=now())
                atomic_write(receipt, row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(
            runs=ledger, updates=sum(r["updates"] for r in ledger), gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__), protocol_SHA256=digest(protocol), finished_utc=now(),
        ))


def load_models():
    source = read(ROOT / "completed.json")
    if source["protocol_SHA256"] != digest(ROOT / "protocol.json") or {
        (r["arm"], r["seed"]) for r in source["runs"]
    } != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("Registered shared field training is incomplete")
    result = {}
    for row in source["runs"]:
        if row["checkpoint_SHA256"] != digest(row["file"]):
            raise ValueError("A shared field weight changed")
        net = SharedHazeField().cuda()
        net.load_state_dict(torch.load(row["file"], map_location="cpu", weights_only=True), strict=True)
        result[row["arm"], row["seed"]] = net.eval().requires_grad_(False)
    return result


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--smoke", action="store_true")
    fit(smoke=arg.parse_args().smoke)

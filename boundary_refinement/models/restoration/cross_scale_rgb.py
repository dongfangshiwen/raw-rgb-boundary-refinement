"""Learn cross-scale haze correction with a controlled native detail path."""
from boundary_refinement.paths import SOURCE_ROOT, record_path
import argparse
import time

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.training.crop_context import originals
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.source_replay import PROTOCOL as RW_FIT
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD
from boundary_refinement.data.staging.stage_postfusion_holdout import RW_CARD
from boundary_refinement.data.staging.stage_revide_holdout import ROOT
from boundary_refinement.training.structure_transfer import SEEDS, training_scenes
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


MODES = ("tied", "bands", "copy_detail", "local_affine", "shifted_affine")
DETAIL = "local_affine"
STEPS, SIDE = 384, 160
ALPHAS = (.25, .50, .75, 1.)
RW_LIMIT = -.25
RGB_MARGIN = .10


def same_box(x, width):
    return F.avg_pool2d(x, width, stride=1, padding=width // 2, count_include_pad=False)


def coarse_gain(image, restored):
    mean = same_box(torch.cat((image, restored), dim=1), 9)
    mx, my = mean.chunk(2, dim=1)
    variance = (same_box(image.square(), 9) - mx.square()).clamp_min(0.)
    covariance = same_box(image * restored, 9) - mx * my
    # The ridge shrinks texture gains toward identity in nearly flat regions.
    return ((covariance + .0015) / (variance + .0015)).clamp(.40, 3.)


def resize(image, shape, antialias=False):
    return F.interpolate(image, shape, mode="bilinear", align_corners=False, antialias=antialias)


@torch.no_grad()
def packet(restorer, x):
    """Return full-grid prepared channels. Target RGB never enters this step."""
    h, w = x.shape[-2:]
    b = pgl_rgb(restorer, x).clamp(0., 1.)
    small = resize(x, (max(32, round(h * .5)), max(32, round(w * .5))), antialias=True)
    c = pgl_rgb(restorer, small).clamp(0., 1.)
    coarse = resize(c, (h, w))
    scale = coarse_gain(small, c)
    gain = resize(scale, (h, w))
    rotated = resize(scale.roll((scale.shape[-2] // 2, scale.shape[-1] // 2), dims=(-2, -1)), (h, w))
    difference = coarse - b
    local = same_box(difference, 13)
    x_detail = x - same_box(x, 7)
    b_detail = b - same_box(b, 7)
    guides = torch.cat((2. * x - 1., 2. * b - 1., 2. * coarse - 1., 3. * local, 6. * x_detail.abs()), dim=1)
    return (b, coarse, guides, local, difference - local, x_detail, b_detail, gain, rotated)


class ScaleAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(15, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 5, 1))
        nn.init.zeros_(self.branch[-1].weight)
        with torch.no_grad():
            self.branch[-1].bias.zero_()
            self.branch[-1].bias[3:] = -1.25

    def forward(self, prepared, mode):
        base, _, guides, local, frequency, raw, existing, gain, shuffled = prepared
        shape = base.shape[-2:]
        context = F.adaptive_avg_pool2d(guides, (max(4, shape[0] // 4), max(4, shape[1] // 4)))
        heads = resize(self.branch(context).float(), shape)
        color, amounts = heads.split((3, 2), dim=1)
        if mode == "tied":
            low = high = amounts.mean(1, keepdim=True).sigmoid()
        else:
            low, high = amounts.sigmoid().split(1, 1)
        if mode == "copy_detail":
            frequency = raw - existing
        elif mode in (DETAIL, "shifted_affine"):
            strength = gain if mode == DETAIL else shuffled
            frequency = strength * raw - existing
        residual = .22 * color.tanh() + low * local + high * frequency
        return residual


@torch.no_grad()
def prepare_train(restorer, smoke):
    lm = training_scenes()
    chosen = (1, 3, 5, 7)
    lm_images = ((r["frames"][j], r["clear"]) for r in (lm[:2] if smoke else lm) for j in chosen)
    rw = originals()
    source = (tuple(x) for x in (rw[:3] if smoke else rw))
    banks, counts = [], []
    for group, stream in enumerate((lm_images, source)):
        bank = []
        for pair in stream:
            views = torch.stack(pair).float().cuda() / 255.
            h, w = views.shape[-2:]
            longest = max(h, w)
            shape = tuple(max(SIDE, round(a * min(1., 768 / longest))) for a in (h, w))
            x, y = resize(views, shape, antialias=True)
            features = packet(restorer, x[None])
            bank.append((tuple(v.half() for v in features), y[None].half()))
        banks.append(bank)
        counts.append(len(bank))
        print(dict(training_bank=group, frames=len(bank)), flush=True)
    return banks, counts


def example(bank, rng):
    prepared, y = bank[int(torch.randint(len(bank), (), generator=rng))]
    h, w = y.shape[-2:]
    top, left = int(torch.randint(h - SIDE + 1, (), generator=rng)), int(torch.randint(w - SIDE + 1, (), generator=rng))
    region = (..., slice(top, top + SIDE), slice(left, left + SIDE))
    return tuple(v[region].float() for v in prepared), y[region].float()


def rgb_loss(output, truth):
    d = output - truth
    return (d.square() + 1e-6).sqrt().mean() + .18 * (d.diff(dim=-1).abs().mean() + d.diff(dim=-2).abs().mean())


def fit(smoke=False):
    configure_cuda()
    restorer, checkpoint = rgb_checkpoint()
    parameters = sum(p.numel() for p in ScaleAdapter().parameters())
    protocol = ROOT / "protocol.json"
    registered = dict(
        methods=list(MODES), evaluation_candidate=DETAIL, trainable=parameters, steps=STEPS,
        seeds=list(SEEDS), crop=SIDE, anchor_SHA256=checkpoint,
        inputs={record_path(path): digest(path) for path in (TRAIN_CARD, RW_FIT, RW_CARD,
                 SOURCE_ROOT / "boundary_refinement/models/restoration/cross_scale_rgb.py")},
        hypothesis="Separate broad haze correction from native texture, and estimate a local contrast transfer from the same coarse input/output pixels. Compare affine detail to unit contrast and equally distributed but spatially shifted contrast.",
        controls="Every learned arm has an identical convolutional decoder, the same frozen native and half-scale passes, shared additive RGB correction and equal compute schedule. Tied uses one blend amount for coarse-minus-native low and high bands. Bands learns separate amounts for the same bands. Copy detail replaces the coarse high-band proposal with observed detail. Local affine estimates per-channel ridge gain, anchored at unity in low-texture neighborhoods. Shifted affine relocates the same gains and preserves the native texture.",
        fit_data="Use four fixed intensity positions per 32 registered LMHaze train scenes and the 86 existing RW2AH training captures. Sample one native crop per domain, optimize the same paired loss, and penalize large changes to the anchor on source captures. No REVIDE pairs participate in fitting.",
        selection="Select one common amplitude per arm and seed matrix using previous LMHaze calibrate scenes and the RW reserve at a -0.25 dB source budget. Evaluate 130 LMHaze views and 46 RW2AH captures already used in prior studies, and new REVIDE cuts. Compare with native anchor, second-scale anchor and unlearned equal RGB merge.",
        gain_margin_db=RGB_MARGIN, scales=[0., *ALPHAS], source_floor_db=RW_LIMIT,
    )
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if protocol.exists() and read(protocol) != registered:
            raise ValueError("The cross-scale architecture protocol changed")
        if not protocol.exists():
            atomic_write(protocol, registered)
    banks, counts = prepare_train(restorer, smoke)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            storage = ROOT / f"{mode}-{seed}.pth"
            receipt = storage.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["checkpoint_SHA256"] != digest(storage) or row["protocol_SHA256"] != digest(protocol):
                    raise ValueError("A trained cross-scale checkpoint changed")
                ledger.append(row)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 44041)
            model = ScaleAdapter().cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00008)
            start = time.perf_counter()
            for step in range(1, (2 if smoke else STEPS) + 1):
                optimizer.zero_grad(set_to_none=True)
                a, ya = example(banks[0], rng)
                b, yb = example(banks[1], rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    da = model(a, mode)
                    db = model(b, mode)
                total = rgb_loss((a[0] + da).clamp(0., 1.), ya)
                total += .75 * rgb_loss((b[0] + db).clamp(0., 1.), yb) + .55 * db.abs().mean()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
            data = dict(arm=mode, seed=seed, updates=step, trainable=parameters, loss=total.item(),
                        seconds=round(time.perf_counter() - start, 2))
            print(data, flush=True)
            if not smoke:
                torch_save(storage, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                row = dict(data, file=str(storage), checkpoint_SHA256=digest(storage), protocol_SHA256=digest(protocol),
                           finished_utc=now())
                atomic_write(receipt, row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, updates=sum(r["updates"] for r in ledger),
            protocol_SHA256=digest(protocol), lm_train_views=counts[0], rw_train_views=counts[1],
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now()))


def fitted(blend):
    prior = read(ROOT / "completed.json")
    if prior["protocol_SHA256"] != digest(ROOT / "protocol.json") or {
        (r["arm"], r["seed"]) for r in prior["runs"]
    } != {(method, seed) for method in MODES for seed in SEEDS}:
        raise ValueError("The cross-scale matched study is incomplete")
    choices = {}
    for row in prior["runs"]:
        if row["checkpoint_SHA256"] != digest(row["file"]):
            raise ValueError("A registered cross-scale model changed")
        if blend is not None and row["arm"] not in blend:
            continue
        net = ScaleAdapter().cuda()
        net.load_state_dict(torch.load(row["file"], map_location="cpu", weights_only=True), strict=True)
        choices[row["arm"], row["seed"]] = net.eval().requires_grad_(False)
    return choices


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--smoke", action="store_true")
    fit(cli.parse_args().smoke)

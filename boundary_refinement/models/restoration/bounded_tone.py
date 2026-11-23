"""Matched RGB curve experiments on a frozen MCAF restoration, with a new sealed tile set."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.data.staging import hrsd_test as hrsd
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.models.restoration.complementary_rgb import ColorHead, MAX_RMS
from boundary_refinement.training.decision_overlap import MANIFEST, rr_records
from boundary_refinement.evaluation.photographic.evaluate_complementary_rgb import fresh_pairs
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.training.overlap_quadratic import SEEDS, references
from boundary_refinement.metrics.color import ciede2000_lab, srgb_to_lab
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.strong_anchor_color import ANCHOR, STEPS, anchor, context, split
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/bounded-tone-v96"
ARMS = ("affine", "luma", "rolled", "channel_tangent", "channel_flow")
BINS = (0., .15, .35, .65, .85, 1.)


def luminance(x):
    return (x[:, :1] + 2. * x[:, 1:2] + x[:, 2:]) * .25


def window(rgb, lum, arm):
    if arm == "affine":
        return torch.ones_like(lum)
    ref = rgb if arm.startswith("channel") else lum
    mask = 4. * ref * (1. - ref)
    if arm == "rolled":
        return mask.roll(tuple(n // 2 for n in ref.shape[-2:]), dims=(-2, -1))
    return mask


def basis(reference, lum):
    one = torch.ones_like(lum)
    axes = reference.new_tensor(((1., 1., 1.), (1., 1., 1.), (1., -1., 0.), (-.5, -.5, 1.)))
    return torch.stack((one, 2. * lum - 1., one, one), dim=1) * axes[None, :, :, None, None]


def flow(x, direction):
    return x / (x + (1. - x) * (-4. * direction).exp())


@torch.no_grad()
def packet(rgb, gt, rec, frozen):
    state, _ = context(as_float(rgb), rec, frozen)
    x, p = state.anchor, state.partition
    y, lum = as_float(gt), luminance(x)
    axes = basis(x, lum)
    rms = {}
    for arm in ARMS:
        fields = axes * window(x, lum, arm)[:, None]
        rms[arm] = p.average(fields.square().mean(2)).clamp_min(0.).sqrt().clamp_min(.055)
    return x, y, state.features, rms


def render(net, arm, x, cues, scale):
    amplitude = MAX_RMS * net(cues).tanh()
    code = F.interpolate(amplitude / scale, x.shape[-2:], mode="bilinear", align_corners=False)
    lum = luminance(x)
    delta = (basis(x, lum) * code[:, :, None]).sum(1)
    pred = flow(x, delta) if arm == "channel_flow" else x + delta * window(x, lum, arm)
    return pred, amplitude


def protocol():
    return dict(code=digest(__file__), anchor=digest(ANCHOR / "completed.json"), split=digest(MANIFEST),
                held_tiles=digest(held.TILES), hrsd=digest(hrsd.MANIFEST), seeds=list(SEEDS), steps=STEPS,
                arms=list(ARMS), strength=1., bins=list(BINS), cases=split(),
                fit="Same 48 paired RRSHID images, frozen features, RMS-normalized four directions, batch 3, "
                    "normalized delivered RGB MSE plus 0.002 times mean squared normalized amplitude, "
                    "320 AdamW updates; full strength fixed for all arms. Identical draws per seed.",
                primary="Fresh RRSHID tiles, luma minus affine and rolled on CIEDE2000, also report PSNR; "
                        "channel_flow minus channel_tangent tests the finite channelwise constraint. Evaluate "
                        "two previously seen external HRSD domains and all brightness bins without retuning. "
                        "Native PSNR; color error and Lab-plane error on fixed hashed 4x4 sampling phase.")


def register():
    config = protocol()
    file = ROOT / "protocol.json"
    if file.exists() and read(file) != config:
        raise ValueError("Locked tone experiment changed")
    if not file.exists():
        atomic_write(file, config)
    return digest(file)


def fit(data, lock):
    x, y, features, scales = zip(*data)
    x, y, features = (torch.cat(v) for v in (x, y, features))
    scale = {a: torch.cat([v[a] for v in scales]) for a in ARMS}
    variance = (x-y).square().mean((1, 2, 3)).clamp_min(.0025)
    models, receipts = {}, []
    for seed in SEEDS:
        for arm in ARMS:
            set_seed(seed)
            net = ColorHead().cuda()
            path = ROOT / f"{arm}-{seed}.pth"
            note = path.with_suffix(".json")
            if note.exists():
                row = read(note)
                if row["weights"] != digest(path) or row["protocol"] != lock:
                    raise ValueError("Tone checkpoint changed")
                net.load_state_dict(torch.load(path, weights_only=True))
            else:
                rng = torch.Generator().manual_seed(seed + 62225)
                opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00007)
                for _ in range(STEPS):
                    idx = torch.randint(len(data), (3,), generator=rng).cuda()
                    pred, amp = render(net, arm, x[idx], features[idx], scale[arm][idx])
                    mse = (pred.clamp(0., 1.) - y[idx]).square().mean((1, 2, 3))
                    loss = (mse / variance[idx]).mean() + .002 * (amp / MAX_RMS).square().mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
                    opt.step()
                    schedule.step()
                torch_save(path, {k: v.detach().cpu() for k, v in net.state_dict().items()})
                row = dict(arm=arm, seed=seed, steps=STEPS, weights=digest(path), protocol=lock,
                           params=sum(p.numel() for p in net.parameters()), loss=float(loss.detach()),
                           sampler=sha256(rng.get_state().numpy().tobytes()).hexdigest())
                atomic_write(note, row)
            models[arm, seed] = net.eval().requires_grad_(False)
            receipts.append(row)
            print(arm, seed, flush=True)
    for s in SEEDS:
        if len({r["sampler"] for r in receipts if r["seed"] == s}) != 1:
            raise ValueError("Unmatched tone draws")
    atomic_write(ROOT / "completed.json", dict(records=receipts, protocol=lock, torch=str(torch.__version__),
                 device=torch.cuda.get_device_name(), steps=STEPS * len(receipts)))
    return models


def stats(pred, target, phase, bins, ref_lab):
    clipped = ((pred < 0.) | (pred > 1.)).float().mean()
    delivered = pred.clamp(0., 1.)
    psnr = -10. * (delivered - target).square().mean().clamp_min(1e-12).log10()
    lab = srgb_to_lab(delivered[..., phase[0]::4, phase[1]::4])
    de = ciede2000_lab(lab, ref_lab).flatten()
    chroma = torch.linalg.vector_norm(lab[..., 1:] - ref_lab[..., 1:], dim=-1).flatten()
    light = (lab[..., 0] - ref_lab[..., 0]).abs().flatten()
    parts = []
    for i in range(len(BINS) - 1):
        select = bins == i
        parts.append(torch.stack([select.sum(), (de*select).sum(), (chroma*select).sum()]))
    vals = torch.stack([psnr, de.mean(), chroma.mean(), light.mean(), clipped]).tolist()
    return dict(zip(("psnr", "deltaE", "ab_error", "L_error", "out_of_range"), vals),
                bins=torch.stack(parts).tolist())


def summarize(rows):
    result = {}
    pairs = (("luma", "affine"), ("luma", "rolled"), ("luma", "anchor"), ("channel_flow", "luma"),
             ("channel_tangent", "luma"), ("channel_flow", "channel_tangent"))
    for ds in sorted({r["ds"] for r in rows}):
        cases = [r for r in rows if r["ds"] == ds]
        by_group = defaultdict(lambda: defaultdict(list))
        for r in cases:
            by_group[r["group"]][r["arm"]].append(r)
        scores, effects = {}, {}
        for metric in ("psnr", "deltaE", "ab_error", "L_error", "out_of_range", *[f"bin{i}" for i in range(5)]):
            bank = defaultdict(dict)
            for g, arms in by_group.items():
                for arm, records in arms.items():
                    if metric.startswith("bin"):
                        j = int(metric[-1])
                        samples = [r["bins"][j][1] / r["bins"][j][0] for r in records if r["bins"][j][0] >= 16]
                    else:
                        samples = [r[metric] for r in records]
                    if samples:
                        bank[arm][g] = mean(samples)
            scores[metric] = {arm: mean(v.values()) for arm, v in bank.items()}
            effects[metric] = {}
            for a, b in pairs:
                common = bank[a].keys() & bank[b].keys()
                if common:
                    effects[metric][a+"-"+b] = bootstrap({g: bank[a][g]-bank[b][g] for g in sorted(common)})
        result[ds] = dict(images=len({r["key"] for r in cases}), groups=len(by_group), scores=scores, effects=effects)
    return result


@torch.inference_mode()
def evaluate(rec, frozen, models, lock):
    rows, seen = [], Counter()
    for ds, key, group, rgb, gt, _, condition in chain(fresh_pairs(read(held.TILES)), hrsd.pairs(hrsd.stage())):
        ds = "rrshid_new" if ds == "rrshid_fresh" else ds
        x, y, features, scales = packet(rgb, gt, rec, frozen)
        phase = [b % 4 for b in sha256(key.encode()).digest()[:2]]
        ref_lab = srgb_to_lab(y[..., phase[0]::4, phase[1]::4])
        l = luminance(x)[..., phase[0]::4, phase[1]::4].flatten()
        bins = torch.bucketize(l, l.new_tensor(BINS[1:-1]), right=True)
        for arm, seed in [("anchor", 0), *models]:
            pred = x if arm == "anchor" else render(models[arm, seed], arm, x, features, scales[arm])[0]
            rows.append(dict(ds=ds, key=key, group=group, condition=condition, arm=arm, seed=seed,
                             **stats(pred, y, phase, bins, ref_lab)))
        seen[ds] += 1
        if sum(seen.values()) % 30 == 0:
            print("scored", dict(seen), flush=True)
    if seen != dict(rrshid_new=90, hrsd_dhid=40, hrsd_lhid=40):
        raise ValueError(f"Incomplete color evaluation {seen}")
    path = ROOT / "individual.json"
    atomic_write(path, rows)
    atomic_write(ROOT / "test.json", dict(protocol=lock, ledger=digest(path), sets=summarize(rows), utc=now()))
    print("tone complete", seen, flush=True)


if __name__ == "__main__":
    configure_cuda()
    held.tiles()
    lock = register()
    rec, frozen = references(), anchor()
    keys = set(split()["fit"])
    data = [packet(x, y, rec, frozen) for _, k, _, x, y in rr_records("val") if k in keys]
    if len(data) != len(keys) or len(keys) != 48:
        raise ValueError("Invalid RGB fit cohort")
    models = fit(data, lock)
    evaluate(rec, frozen, models, lock)

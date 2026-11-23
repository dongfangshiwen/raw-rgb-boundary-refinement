"""Isolate regional RGB correction from whole-image bias and finer residuals."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from statistics import mean

import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.models.restoration.cross_scale_rgb import resize, same_box
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_region_restore import calibrate_data, render_rgb
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.region_restore import FLOOR_DB, LowField, SCALES, SEEDS, view_size
from boundary_refinement.models.restoration.single_anchor_student import ROOT as PARENT, area_psnr, paired_test_data
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.paired.veil_response import training_pairs


ROOT = REPO / "results/controlled-rgb-band-v51"
MODES = ("global", "coarse", "dense", "detail")
STEPS = 512
TEST_COUNTS = dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=16, nhhaze2=5, bedde=208)
CONTRASTS = (("coarse", "global"), ("coarse", "dense"), ("coarse", "detail"), ("coarse", "native"), ("dense", "native"))


def signals(x, native):
    shape = tuple(max(n // 2, 8) * 2 for n in view_size(x, 96, 16))
    raw, base = (F.adaptive_avg_pool2d(t, shape) for t in (x, native))
    var = (F.adaptive_avg_pool2d(x.square(), shape) - raw.square()).clamp_min(0.)
    inputs = torch.cat((2. * raw - 1., 2. * base - 1., 2. * same_box(base, 9) - 1., 4. * var.sqrt()), dim=1)
    return inputs, base, None, None, None


def band(delta, mode):
    if mode == "global":
        return delta.mean((-1, -2), keepdim=True)
    if mode == "dense":
        return delta
    low = F.avg_pool2d(delta, 2, 2)
    if mode == "coarse":
        return resize(low, delta.shape[-2:])
    return delta - F.interpolate(low, scale_factor=2, mode="nearest")


@torch.no_grad()
def fit_data(base, smoke):
    banks = defaultdict(list)
    for domain, _, _, cloudy, clear in training_pairs():
        if smoke and len(banks[domain]) >= 3:
            continue
        image = torch.stack((cloudy, clear)).cuda().float() / 255.
        raw, truth = resize(image, view_size(image, 768), antialias=True).split(1)
        context = signals(raw, pgl_rgb(base, raw).clamp(0, 1))
        target = F.adaptive_avg_pool2d(truth, context[1].shape[-2:])
        banks[domain].append((context, target))
    if {k: len(v) for k, v in banks.items()} != (dict(lm=3, rw=3, nh=3) if smoke else dict(lm=128, rw=86, nh=36)):
        raise ValueError("The registered field-bias training population changed")
    return [banks[k] for k in ("lm", "rw", "nh")]


def fit(smoke):
    configure_cuda()
    anchor, signature = rgb_checkpoint()
    registry = dict(modes=list(MODES), seeds=list(SEEDS), steps=STEPS, code_SHA256=digest(__file__),
        native_SHA256=signature, parent_protocol_SHA256=digest(PARENT / "protocol.json"),
        parameters=sum(v.numel() for v in LowField().parameters()),
        design="All heads use the same RGB features and predict the same 96-pixel-grid RGB residual. Project out spatial variation in global, retain smooth spatial variation by 2:1 pooling and interpolation in coarse, leave the grid unchanged in dense, and subtract the block mean in detail. All stages share the same RGB targets and training draws. No MCT or SegFormer is required by the RGB head.",
        question="Does regional correction outperform a global RGB bias and the finer-grid control? Require at least 0.10 dB on an untouched-by-fit target and a loss of at most 0.25 dB on the RW2AH reference.",
        test_counts=TEST_COUNTS)
    path = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if path.exists() and read(path) != registry:
            raise ValueError("The fixed frequency-control registry changed")
        if not path.exists():
            atomic_write(path, registry)
    banks = fit_data(anchor, smoke)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            storage = ROOT / f"{mode}-{seed}.pth"
            receipt = storage.with_suffix(".json")
            if not smoke and receipt.exists():
                card = read(receipt)
                if card["checkpoint_SHA256"] != digest(storage) or card["protocol_SHA256"] != digest(path):
                    raise ValueError("The frozen frequency-control checkpoint changed")
                ledger.append(card)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 51205)
            net = LowField().cuda()
            optimizer = torch.optim.AdamW(net.parameters(), lr=.0017, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.0001)
            for step in range(2 if smoke else STEPS):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for bank in banks:
                    inp, target = bank[int(torch.randint(len(bank), (), generator=rng))]
                    native = inp[1]
                    rgb = (native + band(net(inp, "direct"), mode)).clamp(0, 1)
                    normalized = (rgb - target).square().mean() / (native - target).square().mean().clamp_min(.0025)
                    losses.append((normalized + .25).log())
                total = losses[0] + .75 * sum(losses[1:])
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
            result = dict(mode=mode, seed=seed, updates=step + 1, loss=total.item())
            print(result, flush=True)
            if not smoke:
                torch_save(storage, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()})
                result.update(checkpoint_SHA256=digest(storage), protocol_SHA256=digest(path))
                atomic_write(receipt, result)
                ledger.append(result)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, updates=sum(r["updates"] for r in ledger),
            protocol_SHA256=digest(path), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def models():
    path = ROOT / "protocol.json"
    plan, receipt = read(path), read(ROOT / "completed.json")
    if not matches_registered(plan['code_SHA256'], digest(__file__)) or receipt["protocol_SHA256"] != digest(path):
        raise ValueError("The frequency-control study was modified")
    if {(r["mode"], r["seed"]) for r in receipt["runs"]} != {(m, seed) for m in MODES for seed in SEEDS}:
        raise ValueError("The frequency-control matrix is incomplete")
    output = {}
    for row in receipt["runs"]:
        key = row["mode"], row["seed"]
        path = ROOT / f"{key[0]}-{key[1]}.pth"
        if digest(path) != row["checkpoint_SHA256"]:
            raise ValueError("A matched frequency-control model changed")
        network = LowField().cuda()
        network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        output[key] = network.eval().requires_grad_(False)
    return output


@torch.inference_mode()
def calibrate(anchor, heads):
    path = ROOT / "calibration.json"
    stamp = digest(ROOT / "completed.json")
    if path.exists():
        known = read(path)
        if known["fit_SHA256"] != stamp:
            raise ValueError("The fixed calibration changed")
        return known
    totals, counts = defaultdict(list), Counter()
    for split, items in calibrate_data():
        for _, _, _, foggy, clear in items:
            x, y = as_float(foggy), as_float(clear)
            b = pgl_rgb(anchor, x).clamp(0, 1)
            context = signals(x, b)
            p = area_psnr(b, y, None)
            for (mode, _), net in heads.items():
                change = band(net(context, "direct"), mode)
                full = resize(change, b.shape[-2:])
                for a in SCALES:
                    diff = area_psnr((b + a * full).clamp(0, 1), y, None) - p if a else 0.
                    totals[split, mode, a].append(diff)
            counts[split] += 1
    if counts != dict(lm=64, rw=47, nh=9):
        raise ValueError("Unregistered frequency-control calibration pairs")
    table = {k: mean(v) for k, v in totals.items()}
    amount = {m: max((a for a in SCALES if table["rw", m, a] >= FLOOR_DB),
                     key=lambda a: (mean(table[s, m, a] for s in ("lm", "nh")), -a)) for m in MODES}
    receipt = dict(amplitudes=amount, counts=dict(counts), fit_SHA256=stamp,
        means={"/".join(map(str, k)): v for k, v in table.items()}, finished_utc=now())
    atomic_write(path, receipt)
    print(dict(calibrated=amount), flush=True)
    return receipt


@torch.inference_mode()
def test():
    configure_cuda()
    anchor, _ = rgb_checkpoint()
    heads = models()
    selected = calibrate(anchor, heads)
    rows, counts, means, paired, panels = [], Counter(), defaultdict(list), defaultdict(lambda: defaultdict(list)), []
    for ds, name, group, foggy, clear, roi in paired_test_data():
        x, y = as_float(foggy), as_float(clear)
        b = pgl_rgb(anchor, x).clamp(0., 1.)
        context = signals(x, b)
        mask = roi.cuda() if roi is not None else None
        within = defaultdict(list)
        predictions = {("native", 0): b}
        predictions.update({(m, s): render_rgb(b, band(net(context, "direct"), m), selected["amplitudes"][m])
                            for (m, s), net in heads.items()})
        for (mode, seed), pred in predictions.items():
            value = area_psnr(pred, y, mask)
            rows.append(dict(dataset=ds, group=group, id=name, method=mode, seed=seed, psnr=value))
            within[mode].append(value)
        for m, value in within.items():
            means[ds, m].append(mean(value))
        for a, c in CONTRASTS:
            paired[ds, a, c][group].append(mean(within[a]) - mean(within[c]))
        counts[ds] += 1
        if counts[ds] == 1:
            photo = (x, b, *(predictions[m, SEEDS[0]] for m in ("global", "coarse", "dense")), y)
            panels.append((ds + " / " + name, [to_rgb(t) for t in photo],
                           ("Hazy input", "Native PGL", "Global RGB bias", "Coarse regional", "Dense regional", "Clear reference")))
        if counts[ds] % 64 == 0:
            print(dict(dataset=ds, scored=counts[ds]), flush=True)
    if counts != TEST_COUNTS or len(rows) != sum(TEST_COUNTS.values()) * (1 + len(MODES) * len(SEEDS)):
        raise ValueError(f"RGB band test populations changed {counts}")
    summary = {"/".join(k): mean(v) for k, v in means.items()}
    intervals = {"/".join(k): bootstrap({g: mean(v) for g, v in items.items()}) for k, items in paired.items()}
    atomic_write(ROOT / "individual.json", rows)
    draw(ROOT / "predictions.png", panels)
    atomic_write(ROOT / "test.json", dict(counts=dict(counts), rows=len(rows), psnr=summary, paired=intervals,
        amplitudes=selected["amplitudes"], calibration_SHA256=digest(ROOT / "calibration.json"),
        table_SHA256=digest(ROOT / "individual.json"), panel_SHA256=digest(ROOT / "predictions.png"), finished_utc=now()))
    for ds in counts:
        print(ds, {m: round(summary[f"{ds}/{m}"], 4) for m in ("native", *MODES)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "fit", "test"))
    args = parser.parse_args()
    if args.phase == "test":
        test()
    else:
        fit(args.phase == "smoke")

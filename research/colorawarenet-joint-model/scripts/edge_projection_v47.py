"""Train a local gradient correction and compare matched edge and pixel ablations."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.coarse_semantics_v39 import blend, native_labels, paint, rgb_checkpoint
from scripts.cross_scale_rgb_v44 import DETAIL, ROOT as V44, SEEDS, example, fitted, packet, prepare_train, resize, rgb_loss
from scripts.dual_view_inference_v43 import two_views
from scripts.evaluate_cross_scale_rgb_v44 import calibrate as parent_calibrate
from scripts.evaluate_cross_scale_rgb_v44 import metrics, outputs, test_pairs, visible_edges
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_paired_basis_v34 import validation_source
from scripts.evaluate_real_semantics_v22 import load_segmenter, resize_image
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.evaluate_structure_transfer_v32 import data_split
from scripts.expert_fusion_v26 import experts
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.stage_nhhaze2_v32 import CARD as NH2
from scripts.stage_revide_holdout_v44 import MANIFEST as REVIDE_FIRST
from scripts.stage_revide_remaining_v45 import MANIFEST as REVIDE_FINAL, pairs as additional_revide
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/edge-projection-v47"
ARMS = ("uniform", "unit", "shifted", "pixel", "aligned")
CANDIDATE = "aligned"
STEPS = 384
AXES = (-1, -2)
ALPHAS = (0., .25, .5, .75, 1., 1.5, 2.)
SOURCE_FLOOR_DB = -.03
METRICS = ("psnr", "ssim", "visible_edge_mae", "edge_mae", "rgb_mean_shift", "rgb_change")
TEST_COUNTS = dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=16, nhhaze2=5)


def endpoints(value, axis):
    if axis == -1:
        return .5 * (value[..., 1:] + value[..., :-1])
    return .5 * (value[..., 1:, :] + value[..., :-1, :])


def adjacent(value, axis):
    if axis == -1:
        return F.pad(value, (1, 0, 0, 0)), F.pad(value, (0, 1, 0, 0))
    return F.pad(value, (0, 0, 1, 0)), F.pad(value, (0, 0, 0, 1))


def adjoint(fluxes):
    hx, hy = (adjacent(value, axis) for value, axis in zip(fluxes, AXES, strict=True))
    return (hx[0] - hx[1]) + (hy[0] - hy[1])


def project(target, weights, initial):
    """Take fixed, stable steps on ||delta||^2 + sum w |D delta - target|^2."""
    direction = adjoint(tuple(w * t for w, t in zip(weights, target, strict=True)))
    change = initial
    for _ in range(4):
        field = tuple(w * change.diff(dim=axis) for w, axis in zip(weights, AXES, strict=True))
        change = change + .11 * (direction - adjoint(field) - change)
    return change


def raw_edges(raw):
    difference = tuple(raw.diff(dim=axis) for axis in AXES)
    strength = tuple(((v.abs().mean(dim=1, keepdim=True) - .012) / .078).clamp(0., 1.) for v in difference)
    return difference, strength


class EdgeAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(15, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 2, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context, method):
        base, _, guides, _, _, texture, existing, gain, shifted = context
        h, w = base.shape[-2:]
        low = F.adaptive_avg_pool2d(guides, (max(4, h // 4), max(4, w // 4)))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=low.is_cuda):
            prediction = self.network(low)
        controls = resize(prediction.float(), (h, w)).sigmoid().split(1, dim=1)
        observed, strength = raw_edges(guides[:, :3].mul(.5).add(.5))
        if method == "uniform":
            strength = tuple(g.mean(dim=(-2, -1), keepdim=True) for g in strength)
        weights = tuple(s * endpoints(c, axis) for s, c, axis in zip(strength, controls, AXES, strict=True))
        if method == "pixel":
            # Same gains and directions, applied to high-pass RGB directly.
            incident = .25 * sum(sum(adjacent(v, axis)) for v, axis in zip(weights, AXES, strict=True))
            change = incident * (gain * texture - existing).clamp(-.20, .20)
            return change - change.mean(dim=(-2, -1), keepdim=True)
        contrast = shifted if method == "shifted" else gain
        if method == "unit":
            contrast = torch.ones_like(gain)
        target = tuple((endpoints(contrast, axis) * v - base.diff(dim=axis)).clamp(-.20, .20)
                       for v, axis in zip(observed, AXES, strict=True))
        return project(target, weights, torch.zeros_like(base))


def visible_error(rgb, truth, support):
    residual = rgb - truth
    terms = [((residual.diff(dim=axis).square() + 1e-6).sqrt() * g).sum()
             for axis, g in zip(AXES, support, strict=True)]
    return sum(terms) / (3 * sum(g.sum() for g in support)).clamp_min(1)


def fit(smoke=False):
    configure_cuda()
    anchor, signature = rgb_checkpoint()
    parameters = sum(p.numel() for p in EdgeAdapter().parameters())
    registered = dict(
        code_SHA256=digest(__file__), anchor_SHA256=signature, parent_SHA256=digest(V44 / "test.json"),
        parent_fit_SHA256=digest(V44 / "protocol.json"), seeds=list(SEEDS), modes=list(ARMS), steps=STEPS, trainable=parameters,
        hypothesis="Transfer an aligned gain to finite differences and recover a zero-mean contrast residual with four local screened gradient steps. Preserve the unmodified native RGB estimate as the anchor.",
        controls="All arms share native and half-resolution PGL, the convolutional guidance head, training pairs and sampled crops. Uniform removes local edge confidence, unit removes the coarse contrast gain, shifted keeps its distribution while relocating it, and pixel applies the same cues to high-pass RGB without gradient integration.",
        supervision="Use 128 real LMHaze training views and 86 RW2AH training pairs. One crop from each dataset per update. The same loss in every arm combines RGB and visible edge fidelity. No external pairs enter fitting.",
        calibration=dict(amplitudes=list(ALPHAS), constraint_db=SOURCE_FLOOR_DB,
                         objective="Minimize visible-edge MAE on 64 LMHaze and 47 RW2AH calibration pairs with equal domain weight while respecting the PSNR floor in both."),
        evaluation=dict(expected=TEST_COUNTS, primary="Compare aligned to native and the four controls for visible edges, PSNR and SSIM. REVIDE was previously measured by other experiments. NH-HAZE2 is a V47-only supplemental transfer slice within the V44-V47 sequence and has five matched images.",
                        data_SHA256={p.name: digest(p) for p in (REVIDE_FIRST, REVIDE_FINAL, NH2)}),
    )
    lock = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if lock.exists() and read(lock) != registered:
            raise ValueError("The fixed gradient experiment changed")
        if not lock.exists():
            atomic_write(lock, registered)
    banks, counts = prepare_train(anchor, smoke)
    completed = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in ARMS:
            storage = ROOT / f"{mode}-{seed}.pth"
            receipt = storage.with_suffix(".json")
            if not smoke and receipt.exists():
                info = read(receipt)
                if info["checkpoint_SHA256"] != digest(storage) or info["protocol_SHA256"] != digest(lock):
                    raise ValueError("A trained edge adapter changed")
                completed.append(info)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 47017)
            head = EdgeAdapter().cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=.0012, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00008)
            start = perf_counter()
            for _ in range(2 if smoke else STEPS):
                optimizer.zero_grad(set_to_none=True)
                loss = []
                for bank in banks:
                    context, truth = example(bank, rng)
                    raw = context[2][:, :3].mul(.5).add(.5)
                    support = visible_edges(raw, truth)
                    correction = head(context, mode)
                    prediction = (context[0] + correction).clamp(0., 1.)
                    loss.append(rgb_loss(prediction, truth) + .85 * visible_error(prediction, truth, support))
                total = loss[0] + .75 * loss[1]
                total.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
            stats = dict(mode=mode, seed=seed, steps=2 if smoke else STEPS, loss=total.item(),
                         seconds=round(perf_counter() - start, 2))
            print(stats, flush=True)
            if not smoke:
                torch_save(storage, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                info = dict(stats, checkpoint_SHA256=digest(storage), protocol_SHA256=digest(lock))
                atomic_write(receipt, info)
                completed.append(info)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=completed, updates=sum(r["steps"] for r in completed),
            fit_counts=counts, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
            protocol_SHA256=digest(lock), finished_utc=now()))


def load_heads(methods=ARMS, seeds=SEEDS):
    path = ROOT / "protocol.json"
    card = read(ROOT / "completed.json")
    protocol = digest(path)
    if card["protocol_SHA256"] != protocol or read(path)["code_SHA256"] != digest(__file__):
        raise ValueError("Gradient architecture receipt changed")
    if {(q["mode"], q["seed"]) for q in card["runs"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Incomplete gradient fit matrix")
    nets = {}
    for row in card["runs"]:
        mode, seed = row["mode"], row["seed"]
        if mode in methods and seed in seeds:
            storage = ROOT / f"{mode}-{seed}.pth"
            if digest(storage) != row["checkpoint_SHA256"] or row["protocol_SHA256"] != protocol:
                raise ValueError("A registered edge adapter changed")
            head = EdgeAdapter().cuda()
            head.load_state_dict(torch.load(storage, map_location="cpu", weights_only=True), strict=True)
            nets[mode, seed] = head.eval().requires_grad_(False)
    return nets


@torch.inference_mode()
def calibrate():
    configure_cuda()
    target = ROOT / "calibration.json"
    stamp = digest(ROOT / "completed.json")
    if target.exists():
        receipt = read(target)
        if receipt["fit_SHA256"] != stamp:
            raise ValueError("Gradient calibration no longer matches the fits")
        return receipt
    anchor, _ = rgb_checkpoint()
    models = load_heads()
    ledger, counts = defaultdict(list), defaultdict(int)
    for split, rows in (("lm", data_split("calibrate")), ("rw", validation_source())):
        for _, _, _, fog, clear in rows:
            raw, truth = as_float(fog), as_float(clear)
            context = packet(anchor, raw)
            support = visible_edges(raw, truth)
            base = context[0]
            base_score = metrics(base, truth, support)
            for (mode, _), head in models.items():
                delta = head(context, mode)
                for a in ALPHAS:
                    result = metrics((base + a * delta).clamp(0., 1.), truth, support) if a else base_score
                    edge = result["visible_edge_mae"]
                    ledger[split, mode, a, "psnr"].append(result["psnr"] - base_score["psnr"])
                    if edge is not None:
                        ledger[split, mode, a, "edge"].append(edge - base_score["visible_edge_mae"])
            counts[split] += 1
            if counts[split] % 25 == 0:
                print(dict(calibration=split, paired=counts[split]), flush=True)
    if dict(counts) != dict(lm=64, rw=47):
        raise ValueError("Incorrect calibration population")
    average = {k: mean(v) for k, v in ledger.items()}
    selected = {}
    for mode in ARMS:
        valid = [a for a in ALPHAS if all(average[split, mode, a, "psnr"] >= SOURCE_FLOOR_DB for split in counts)]
        selected[mode] = min(valid, key=lambda a: (mean(average[s, mode, a, "edge"] for s in counts), a))
    receipt = dict(amplitudes=selected, counts=dict(counts), fit_SHA256=stamp, metrics=["PSNR", "visible_edge_MAE"],
                   relative_to_native={"/".join(map(str, k)): v for k, v in average.items()}, finished_utc=now())
    atomic_write(target, receipt)
    print(dict(amplitudes=selected), flush=True)
    return receipt


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count, out=np.full(len(METRICS), np.nan), where=count > 0)


def aggregate(rows):
    by_image, by_method = defaultdict(list), defaultdict(list)
    for item in rows:
        vector = [np.nan if item[m] is None else item[m] for m in METRICS]
        by_image[item["dataset"], item["group"], item["scene"], item["method"]].append(vector)
    aligned = defaultdict(dict)
    for (ds, group, scene, method), values in by_image.items():
        record = finite_mean(values)
        aligned[ds, group, scene][method] = record
        by_method[ds, method].append(record)
    summary = {"/".join(k): dict(zip(METRICS, [float(x) if np.isfinite(x) else None for x in finite_mean(v)], strict=True))
               for k, v in by_method.items()}
    grouped = defaultdict(lambda: defaultdict(list))
    for (ds, group, _), methods in aligned.items():
        for control in ("native", "v44_detail", *[a for a in ARMS if a != CANDIDATE]):
            grouped[ds, control][group].append(methods[CANDIDATE] - methods[control])
    paired = {}
    for key, blocks in grouped.items():
        means = {g: finite_mean(v) for g, v in blocks.items()}
        statistics = {}
        for i, metric in enumerate(METRICS):
            available = {g: float(value[i]) for g, value in means.items() if np.isfinite(value[i])}
            if available:
                statistics[metric] = bootstrap(available)
        paired["/".join(key)] = statistics
    return summary, paired


def nhhaze2_pairs():
    archive = read(NH2)
    if digest(archive["file"]) != archive["SHA256"]:
        raise ValueError("NH-HAZE2 pixels differ from the registration")
    items = torch.load(archive["file"], map_location="cpu", weights_only=True)
    if len(items) != archive["count"]:
        raise ValueError("NH-HAZE2 population changed")
    for row in items:
        yield "nhhaze2", row["id"], row["id"], row["hazy"], row["clean"]


def draw(file, rows):
    columns = len(rows[0][2])
    w, h = (250, 214) if columns > 3 else (450, 298)
    canvas = Image.new("RGB", (w * columns, h * len(rows)), "#f2f5f8")
    pen = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=16)
    for i, (caption, images, labels) in enumerate(rows):
        top = i * h
        if caption:
            pen.text((8, top + 5), caption, fill="#273747", font=title)
        for j, (tag, value) in enumerate(zip(labels, images, strict=True)):
            left = j * w
            panel = Image.fromarray(value).convert("RGB")
            panel.thumbnail((w - 14, h - 55), Image.Resampling.LANCZOS)
            pen.text((left + 8, top + 27), tag, fill="#273747", font=title)
            canvas.paste(panel, (left + (w - panel.width) // 2, top + 49 + (h - 55 - panel.height) // 2))
    canvas.save(file)


@torch.inference_mode()
def test():
    configure_cuda()
    cal = calibrate()
    parent = parent_calibrate()["scales"][DETAIL]
    previous = fitted({DETAIL})
    candidates = load_heads()
    anchor, _ = rgb_checkpoint()
    published, provenance = experts()
    table, panels, counts = [], [], defaultdict(int)
    streams = (test_pairs(), additional_revide(), nhhaze2_pairs())
    for stream in streams:
        for ds, name, group, fog, clear in stream:
            raw, truth = as_float(fog), as_float(clear)
            context = packet(anchor, raw)
            base, support = context[0], visible_edges(raw, truth)
            predicted = {("native", 0): base, ("author_pgl", 0): published[1](raw).clamp(0., 1.),
                         ("author_mct", 0): published[0](raw).clamp(0., 1.)}
            predicted.update({("v44_detail", seed): (base + parent * value).clamp(0., 1.)
                              for (_, seed), value in outputs(context, previous).items()})
            predicted.update({(m, seed): (base + cal["amplitudes"][m] * head(context, m)).clamp(0., 1.)
                              for (m, seed), head in candidates.items()})
            for (m, seed), value in predicted.items():
                delta = value - base
                table.append(dict(dataset=ds, scene=name, group=group, method=m, seed=seed,
                    **metrics(value, truth, support), ssim=gaussian_ssim(value, truth).item(),
                    rgb_mean_shift=delta.mean((-2, -1)).abs().mean().item() * 255.,
                    rgb_change=delta.abs().mean().item() * 255.))
            counts[ds] += 1
            if counts[ds] == 1:
                pictures = (raw, base, predicted["v44_detail", SEEDS[0]], predicted[CANDIDATE, SEEDS[0]],
                            predicted["author_mct", 0], truth)
                panels.append((f"{ds} / {name}", [to_rgb(value) for value in pictures],
                    ("Hazy input", "Native anchor", "V44 detail", "V47 projected", "DehazeFormer-MCT", "Clear RGB")))
            if counts[ds] % 30 == 0:
                print(dict(dataset=ds, pairs=counts[ds]), flush=True)
    if dict(counts) != TEST_COUNTS:
        raise ValueError("The fixed evaluation population changed")
    summary, paired = aggregate(table)
    atomic_write(ROOT / "individual.json", table)
    image = ROOT / "predictions.png"
    draw(image, panels)
    report = dict(counts=dict(counts), rows=len(table), means=summary, paired=paired, amplitudes=cal["amplitudes"],
                  calibration_SHA256=digest(ROOT / "calibration.json"), individual_SHA256=digest(ROOT / "individual.json"),
                  image_SHA256=digest(image), authors=provenance, finished_utc=now())
    atomic_write(ROOT / "test.json", report)
    for d in counts:
        print(d, {k: tuple(round(summary[f"{d}/{k}"][m], 5) for m in METRICS[:3])
                  for k in ("native", "v44_detail", *ARMS, "author_pgl", "author_mct")}, flush=True)


@torch.inference_mode()
def infer(image, out):
    configure_cuda()
    calibrated = calibrate()
    anchor, signature = rgb_checkpoint()
    head = load_heads((CANDIDATE,), (SEEDS[0],))[CANDIDATE, SEEDS[0]]
    seg, mean_rgb, std_rgb, author = load_segmenter()
    with Image.open(image) as opened:
        pixels = np.array(ImageOps.exif_transpose(opened).convert("RGB"))
    x = torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda().float() / 255.
    context = packet(anchor, x)
    original = context[0]
    rgb = (original + calibrated["amplitudes"][CANDIDATE] * head(context, CANDIDATE)).clamp(0., 1.)
    _, _, scores = two_views(seg, mean_rgb, std_rgb, resize_image(pixels)[0])
    semantic = native_labels(scores, pixels.shape[:2]).byte().cpu().numpy()
    if rgb.shape[-2:] != semantic.shape or not np.isin(semantic, range(19)).all():
        raise ValueError("RGB and native class IDs do not align")
    out.mkdir(parents=True, exist_ok=True)
    saved = dict(input=pixels, anchor=to_rgb(original), restored=to_rgb(rgb), mask=semantic,
                 color_mask=paint(semantic), overlay=blend(pixels, semantic))
    for name, value in saved.items():
        Image.fromarray(value).save(out / (name + ".png"))
    draw(out / "comparison.png", (
        (image.name, [saved[n] for n in ("input", "anchor", "restored")],
         ("Hazy input", "Native RGB anchor", "Projected dehazed RGB")),
        ("Original-resolution 19-class segmentation", [saved[n] for n in ("input", "color_mask", "overlay")],
         ("Segmentation input", "Predicted classes", "Predicted overlay")),
    ))
    atomic_write(out / "metadata.json", dict(input=str(image.resolve()), input_SHA256=digest(image),
        size=list(semantic.shape[::-1]), backbone_SHA256=signature, amplitude=calibrated["amplitudes"][CANDIDATE],
        segmentation=author["model"], output_SHA256={name: digest(out / name) for name in
        (*(n + ".png" for n in saved), "comparison.png")}, finished_utc=now()))
    print(dict(native=list(semantic.shape[::-1]), saved=list(saved)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    for stage in ("smoke", "fit", "calibrate", "test"):
        commands.add_parser(stage)
    native = commands.add_parser("infer")
    native.add_argument("--input", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.stage == "infer":
        if args.input.resolve() in {(args.output / (name + ".png")).resolve() for name in
                ("input", "anchor", "restored", "mask", "color_mask", "overlay", "comparison")}:
            cli.error("Output must differ from source")
        infer(args.input, args.output)
    elif args.stage in ("smoke", "fit"):
        fit(args.stage == "smoke")
    elif args.stage == "calibrate":
        calibrate()
    else:
        test()

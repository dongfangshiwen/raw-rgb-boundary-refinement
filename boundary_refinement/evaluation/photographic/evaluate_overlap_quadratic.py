"""Measure covariance and overlap ablations on the locked real-haze benchmarks."""
import argparse
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageOps
import torch

from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import blend, native_labels, paint
from boundary_refinement.inference.dual_view_inference import two_views
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_orthogonal_risk import COUNTS as OLDER, evaluation_stream, reference_models
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter, resize_image
from boundary_refinement.evaluation.paired.evaluate_real_transfer import gaussian_ssim
from boundary_refinement.evaluation.photographic.evaluate_region_restore import calibrate_data
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.cohorts.fresh_real import CARD, COUNTS as FRESH, pairs as new_pairs, stage
from boundary_refinement.training.overlap_quadratic import ROOT, SEEDS, decoded, direct, fitted, geometry, label, preference, references, solve, upper_bound
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.staging.realhyper import CARD as HYPER
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


NAMES = ("direct", "full", "diag", "hard_overlap")
SELECT = (.0, .25, .5, .75, 1.)
FLOOR = -.25
POPULATION = dict(OLDER, **FRESH)


def directions(geom, bank):
    pairs = {}
    for seed in SEEDS:
        pairs["direct", seed] = direct(bank["direct", seed](geom.features))
        energy = preference(bank["alignment", seed], geom)
        pairs["full", seed] = solve(geom.gram, energy)
        pairs["diag", seed] = solve(geom.gram, energy, diagonal=True)
    return pairs


@torch.inference_mode()
def calibrate():
    configure_cuda()
    path = ROOT / "calibration.json"
    lock = digest(ROOT / "completed.json")
    code = digest(__file__)
    if path.exists():
        result = read(path)
        if (result["fit_SHA256"], result["code_SHA256"]) != (lock, code):
            raise ValueError("The locked local fit differs")
        return result
    models = fitted()
    refs = references()
    stats, counts = defaultdict(list), Counter()
    for name, rows in calibrate_data():
        for _, _, _, fog, clear in rows:
            x, y = as_float(fog), as_float(clear)
            geom = geometry(x, refs)
            reference = area_psnr(geom.base, y, None)
            for (mode, _), a in directions(geom, models).items():
                for step in SELECT:
                    delta = 0. if step == 0. else area_psnr(decoded(geom, a, step), y, None) - reference
                    stats[name, mode, step].append(delta)
            counts[name] += 1
    if counts != dict(lm=64, rw=47, nh=9):
        raise ValueError("Unexpected photometric calibration set")
    summary = {k: mean(v) for k, v in stats.items()}
    scales = {m: max((a for a in SELECT if summary["rw", m, a] >= FLOOR),
                     key=lambda a: (mean(summary[d, m, a] for d in ("lm", "nh")), -a)) for m in NAMES[:3]}
    result = dict(scales=scales, counts=dict(counts), fit_SHA256=lock, code_SHA256=code,
        estimates={"/".join(map(str, k)): v for k, v in summary.items()}, finished_utc=now())
    atomic_write(path, result)
    print(dict(calibrated=scales), flush=True)
    return result


def samples():
    for d, k, g, a, b, roi, close, _ in evaluation_stream():
        yield d, k, g, a, b, roi, "closest" if close else "all"
    for row, a, b in new_pairs():
        yield row["dataset"], row["id"], row["group"], a, b, None, row["level"]


def seams(geom, roi):
    inside = torch.ones_like(geom.base[:, :1], dtype=torch.bool) if roi is None else roi[None, None].bool()
    h, w = geom.base.shape[-2:]
    gy, gx = geom.partition.grid
    rows = torch.arange(h, device=inside.device) * gy // h
    cols = torch.arange(w, device=inside.device) * gx // w
    horiz = (cols[1:] != cols[:-1]) & (inside[..., 1:] & inside[..., :-1])
    vert = (rows[1:] != rows[:-1])[:, None] & (inside[..., 1:, :] & inside[..., :-1, :])
    return horiz, vert


def edge_score(error, stripe):
    differences = [(error.diff(dim=axis).abs() * mask).sum().item()
                   for axis, mask in zip((-1, -2), stripe, strict=True)]
    n = sum(mask.sum().item() for mask in stripe)
    return 255. * sum(differences) / (3. * n) if n else None


def stats(rows):
    sets, by_image = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        scope = [r["dataset"]]
        if r["condition"] not in ("all", "night"):
            scope.append(r["dataset"] + "_" + r["condition"])
        for d in scope:
            sets[d, r["method"]].append(r)
            by_image[d, r["group"], r["image"]][r["method"]].append(r)
    metrics = ("psnr", "ssim", "seam_gradient_mae", "harm", "edited")
    means = {"/".join(k): {m: mean(r[m] for r in v if r.get(m) is not None) for m in metrics if any(r.get(m) is not None for r in v)}
             for k, v in sets.items()}
    paired = defaultdict(lambda: defaultdict(list))
    for (d, g, _), p in by_image.items():
        for control in ("anchor", "direct", "diag_match", "diag", "hard_overlap", "equal_pair"):
            for m in ("psnr", "harm", "seam_gradient_mae"):
                left = [r[m] for r in p["full"] if r[m] is not None]
                right = [r[m] for r in p[control] if r[m] is not None]
                if left and right:
                    paired[d, control, m][g].append(mean(left) - mean(right))
    return means, {"/".join(k): bootstrap({g: mean(v) for g, v in vals.items()}) for k, vals in paired.items()}


def prediction(geom, models, scales):
    knots = directions(geom, models)
    values = {(key, seed): decoded(geom, a, scales[key]) for (key, seed), a in knots.items()}
    for seed in SEEDS:
        values["diag_match", seed] = decoded(geom, knots["diag", seed], scales["full"])
        values["hard_overlap", seed] = decoded(geom, knots["full", seed], scales["full"], hard=True)
    return values, knots


@torch.inference_mode()
def test():
    configure_cuda()
    external = stage()
    config = calibrate()
    receipt = ROOT / "evaluation_protocol.json"
    setup = dict(code_SHA256=digest(__file__), calibration_SHA256=digest(ROOT / "calibration.json"),
        data={p.name: digest(p) for p in (CARD, HYPER)}, counts=POPULATION, test_scope=external["note"],
        contrasts="Full Gram versus diagonal at the same scale and from the same learned alignment. Compare with independently calibrated diagonal and same-parameter trained direct gate. Use identical knot weights for hard rendering and bilinear rendering of the full solution. Monitor seam gradient error exactly on the predeclared nearest boundaries.",
        useful="Require +0.10 dB over both learned controls in a fresh paired domain and no more than 0.25 dB loss versus anchor in the longstanding RW2AH reference. Bootstrap by available scene group or documented near-duplicate clusters, not by an implied capture ID.")
    if receipt.exists() and read(receipt) != setup:
        raise ValueError("The locked local evaluation changed")
    if not receipt.exists():
        atomic_write(receipt, setup)
    anchors = references()
    pgl, rs, author = reference_models()
    models = fitted()
    rows, count, panels, inequality_gaps, oracle_gains = [], Counter(), [], defaultdict(list), defaultdict(list)
    for source, name, group, a, b, roi, condition in samples():
        raw, truth = as_float(a), as_float(b)
        geom = geometry(raw, anchors)
        region = roi.cuda() if roi is not None else None
        stripe = seams(geom, region)
        values, knots = prediction(geom, models, config["scales"])
        local, mct, _ = (geom.base + shift[None] for shift in geom.deltas)
        outputs = {("raw", 0): raw, ("anchor", 0): geom.base, ("v51", 0): local, ("mct", 0): mct,
                   ("equal_pair", 0): .5 * (geom.base + mct), ("pgl_author", 0): pgl_rgb(pgl, raw).clamp(0, 1)}
        outputs.update(values)
        if source.startswith("realhyper_") or source == "rrshid":
            outputs["dformer_rs", 0] = author_rgb(rs, raw).clamp(0, 1)
        ref = area_psnr(geom.base, truth, region)
        oracle, _ = label(geom, truth)
        base_err = (geom.base-truth).square().mean()
        oracle_solution = solve(geom.gram, oracle)
        oracle_gains[source].append(area_psnr(decoded(geom, oracle_solution), truth, region) - ref)
        for seed in SEEDS:
            upper = float(upper_bound(geom, config["scales"]["full"] * knots["full", seed], oracle))
            actual = float((values["full", seed] - truth).square().mean() - base_err)
            if actual > upper + 2e-6:
                raise ArithmeticError("The exact overlap bound failed on the measured reference")
            inequality_gaps[source].append(upper - actual)
        for (model, seed), restored in outputs.items():
            psnr = area_psnr(restored, truth, region)
            rows.append(dict(dataset=source, condition=condition, image=name, group=group, method=model, seed=seed,
                psnr=psnr, ssim=float(gaussian_ssim(restored, truth)[0]), seam_gradient_mae=edge_score(restored - truth, stripe),
                harm=psnr <= ref-.10, edited=float((restored-geom.base).square().mean().sqrt()) >= 1/255))
        count[source] += 1
        if source in ("lmhaze", "revide_final", "rrshid", "realhyper_csnc", "nt_haze") and count[source] == 1:
            views = (raw, geom.base, outputs["direct", SEEDS[0]], outputs["diag_match", SEEDS[0]], outputs["full", SEEDS[0]], truth)
            panels.append((source + " / " + name, [to_rgb(pic) for pic in views],
                ("Hazy input", "Native anchor", "Learned direct gate", "Diagonal approximation", "Full covariance", "Clear reference")))
        if count[source] % 100 == 0:
            print(dict(dataset=source, measured=count[source]), flush=True)
    if count != POPULATION:
        raise ValueError(f"Unexpected tested pairs {count}")
    summary, intervals = stats(rows)
    table = ROOT / "individual.json"
    atomic_write(table, rows)
    pic = ROOT / "predictions.png"
    draw(pic, panels)
    audit = {d: dict(checks=len(v), worst_gap=min(v), mean_gap=mean(v)) for d, v in inequality_gaps.items()}
    atomic_write(ROOT / "test.json", dict(counts=dict(count), rows=len(rows), means=summary, paired=intervals, scales=config["scales"],
        bounds=audit, label_informed_oracle_only={k: mean(v) for k, v in oracle_gains.items()}, calibration_SHA256=digest(ROOT / "calibration.json"),
        table_SHA256=digest(table), figure_SHA256=digest(pic), protocol_SHA256=digest(receipt), author=author, finished_utc=now()))
    for d in count:
        print(d, {m: tuple(round(summary[f"{d}/{m}"][k], 4) for k in ("psnr", "seam_gradient_mae"))
                  for m in ("raw", "anchor", "v51", "mct", "equal_pair", "direct", "diag_match", "diag", "hard_overlap", "full")}, flush=True)


@torch.inference_mode()
def infer(path, output):
    configure_cuda()
    cfg = calibrate()
    if path.resolve() in {(output / (name + ".png")).resolve() for name in ("raw", "restored", "anchor", "mask", "color_mask", "overlay", "comparison")}:
        raise ValueError("Output must differ from input")
    with Image.open(path) as handle:
        pixels = np.array(ImageOps.exif_transpose(handle).convert("RGB"))
    image = as_float(torch.from_numpy(pixels.copy()).permute(2, 0, 1))
    geom = geometry(image, references())
    outputs, _ = prediction(geom, fitted(), cfg["scales"])
    restored = to_rgb(outputs["full", SEEDS[0]])
    _, _, probabilities = two_views(*load_segmenter()[:3], resize_image(pixels)[0])
    mask = native_labels(probabilities, pixels.shape[:2]).byte().cpu().numpy()
    if mask.shape != restored.shape[:2] or not np.isin(mask, range(19)).all():
        raise ValueError("Native RGB and semantic labels differ in size or scope")
    files = dict(raw=pixels, anchor=to_rgb(geom.base), restored=restored, mask=mask, color_mask=paint(mask), overlay=blend(pixels, mask))
    output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        Image.fromarray(value).save(output / (name + ".png"))
    draw(output / "comparison.png", ((path.name, [files[k] for k in ("raw", "anchor", "restored")], ("Hazy image", "Native anchor", "Full covariance RGB")),
        ("Cityscapes19 on the raw photograph", [files[k] for k in ("raw", "color_mask", "overlay")], ("Segmentation input", "Predicted mask", "Predicted overlay"))))
    atomic_write(output / "metadata.json", dict(input=str(path.resolve()), input_SHA256=digest(path),
        native=list(mask.shape[::-1]), rgb_scale=cfg["scales"]["full"], seed=SEEDS[0],
        files={n+".png": digest(output/(n+".png")) for n in (*files, "comparison")}, finished_utc=now()))
    print(dict(size=list(mask.shape[::-1])), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="step", required=True)
    sub.add_parser("test")
    sub.add_parser("data-check")
    demo = sub.add_parser("infer")
    demo.add_argument("--input", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.step == "test":
        test()
    elif args.step == "data-check":
        result = Counter(r[0] for r in samples())
        if result != POPULATION:
            raise ValueError(f"Population mismatch {result}")
        print(dict(result), flush=True)
    else:
        infer(args.input, args.output)

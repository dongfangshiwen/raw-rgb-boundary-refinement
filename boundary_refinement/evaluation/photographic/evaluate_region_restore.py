"""Calibrate and evaluate region restoration, preserving native mask coordinates."""
import argparse
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import blend, native_labels, paint
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.inference.dual_view_inference import two_views
from boundary_refinement.models.restoration.edge_projection import TEST_COUNTS, draw, nhhaze2_pairs
from boundary_refinement.evaluation.photographic.evaluate_cross_scale_rgb import metrics, psnr, test_pairs, visible_edges
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_paired_basis import validation_source
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import resize_image
from boundary_refinement.evaluation.paired.evaluate_real_transfer import gaussian_ssim
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.models.restoration.region_restore import FLOOR_DB, MIN_GAIN, MODES, ROOT, SCALES, SEEDS, load_anchors, load_heads, prepare
from boundary_refinement.data.staging.stage_nhhaze2 import CARD as NH2
from boundary_refinement.data.staging.stage_revide_holdout import MANIFEST as REVIDE_A
from boundary_refinement.data.staging.stage_revide_remaining import MANIFEST as REVIDE_B, pairs as additional_revide
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda
from boundary_refinement.evaluation.paired.veil_response import nh


CRITERIA = ("psnr", "ssim", "coarse_rmse", "edge_mae", "visible_edge_mae")
CONTRASTS = (("scalar", "global"), ("split", "direct"), ("split", "scalar"), ("split", "rotated"),
    ("semantic", "split"), ("semantic", "color"), ("semantic", "shifted"),
    ("paired", "semantic"), ("paired", "cross_scene"), ("paired", "native"),
    ("paired", "equal_low"), ("paired", "author_mct"))


def calibrate_data():
    yield "lm", data_split("calibrate")
    yield "rw", validation_source()
    yield "nh", nh("calibration")


def test_data():
    yield from chain(test_pairs(), additional_revide(), nhhaze2_pairs())


def encode(image, anchors):
    restorer, mct, semantic, _ = anchors
    base = pgl_rgb(restorer, image).clamp(0, 1)
    second = author_rgb(mct, image).clamp(0, 1)
    return base, second, prepare(image, base, second, semantic)


def render_rgb(base, delta, amount):
    return (base + amount * resize(delta, base.shape[-2:])).clamp(0., 1.)


@torch.inference_mode()
def calibrate():
    configure_cuda()
    path = ROOT / "calibration.json"
    fit_stamp, code = digest(ROOT / "completed.json"), digest(__file__)
    if path.exists():
        card = read(path)
        if (card["fit_SHA256"], card["evaluator_SHA256"]) != (fit_stamp, code):
            raise ValueError("The fixed coarse RGB calibration changed")
        return card
    models = load_heads()
    anchors = load_anchors()
    totals, counts = defaultdict(list), Counter()
    for split, samples in calibrate_data():
        for _, _, _, fog, clear in samples:
            raw, truth = as_float(fog), as_float(clear)
            base, _, features = encode(raw, anchors)
            reference = psnr(base, truth)
            for (mode, _), network in models.items():
                change = resize(network(features, mode), base.shape[-2:])
                for a in SCALES:
                    result = psnr((base + a * change).clamp(0, 1), truth) - reference if a else 0.
                    totals[split, mode, a].append(result)
            counts[split] += 1
        print(dict(calibration=split, pairs=counts[split]), flush=True)
    if counts != dict(lm=64, rw=47, nh=9):
        raise ValueError(f"Incorrect RGB calibration population {counts}")
    table = {k: mean(v) for k, v in totals.items()}
    amplitude, utility = {}, {}
    for mode in MODES:
        eligible = [a for a in SCALES if table["rw", mode, a] >= FLOOR_DB]
        score = {a: .5 * (table["lm", mode, a] + table["nh", mode, a]) for a in eligible}
        amount = max(score, key=lambda a: (score[a], -a))
        amplitude[mode], utility[mode] = amount, score[amount]
    record = dict(fit_SHA256=fit_stamp, evaluator_SHA256=code, amplitudes=amplitude, target_db=utility,
        select_for_demo=max(MODES, key=utility.get), counts=dict(counts),
        calibration_means={"/".join(map(str, k)): v for k, v in table.items()}, finished_utc=now())
    atomic_write(path, record)
    print(dict(amplitudes=amplitude, calibration_utility=utility, visualization=record["select_for_demo"]), flush=True)
    return record


def exact_stats(prediction, target, support, coarse_shape):
    error = prediction - target
    return dict(**metrics(prediction, target, support), ssim=gaussian_ssim(prediction, target).item(),
        coarse_rmse=F.adaptive_avg_pool2d(error, coarse_shape).square().mean().sqrt().item() * 255.)


def aggregate(rows):
    image_rows, summaries = defaultdict(list), defaultdict(list)
    for row in rows:
        image_rows[row["dataset"], row["group"], row["id"], row["method"]].append(row)
    by_image = defaultdict(dict)
    for (data, group, name, method), items in image_rows.items():
        score = {k: mean(r[k] for r in items if r[k] is not None) for k in CRITERIA if any(r[k] is not None for r in items)}
        by_image[data, group, name][method] = score
        summaries[data, method].append(score)
    means = {"/".join(k): {m: mean(r[m] for r in v if m in r) for m in CRITERIA if any(m in r for r in v)}
             for k, v in summaries.items()}
    compare = defaultdict(lambda: defaultdict(list))
    for (ds, group, _), results in by_image.items():
        for lhs, rhs in CONTRASTS:
            for metric in results[lhs].keys() & results[rhs].keys():
                compare[ds, lhs, rhs, metric][group].append(results[lhs][metric] - results[rhs][metric])
    paired = {"/".join(k): bootstrap({g: mean(value) for g, value in blocks.items()}) for k, blocks in compare.items()}
    return means, paired


@torch.inference_mode()
def test():
    configure_cuda()
    cal = calibrate()
    cal_sha = digest(ROOT / "calibration.json")
    scope = dict(calibration_SHA256=cal_sha, evaluator_SHA256=digest(__file__),
                 sources={p.name: digest(p) for p in (REVIDE_A, REVIDE_B, NH2)}, counts=TEST_COUNTS,
                 comparisons=CONTRASTS, margin=MIN_GAIN, source_loss_floor_db=FLOOR_DB)
    scope["comparisons"] = [list(pair) for pair in CONTRASTS]
    plan = ROOT / "evaluation_protocol.json"
    if plan.exists() and read(plan) != scope:
        raise ValueError("The registered coarse RGB evaluation changed")
    if not plan.exists():
        atomic_write(plan, scope)
    anchors = load_anchors()
    official_pgl, authors = load_pgl()
    trained = load_heads()
    rows, pictures, counts = [], [], Counter()
    fixed_demo = cal["select_for_demo"]
    for ds, name, group, fog, clear in test_data():
        raw, truth = as_float(fog), as_float(clear)
        base, second, features = encode(raw, anchors)
        diff = features[2] - features[1]
        actual = {("native", 0): base, ("author_mct", 0): second,
                  ("author_pgl", 0): pgl_rgb(official_pgl, raw).clamp(0., 1.),
                  ("equal_low", 0): render_rgb(base, diff, .5), ("equal_full", 0): .5 * (base + second)}
        actual.update({(m, seed): render_rgb(base, evaluator(features, m), cal["amplitudes"][m])
                       for (m, seed), evaluator in trained.items()})
        support = visible_edges(raw, truth)
        for (method, seed), value in actual.items():
            rows.append(dict(dataset=ds, id=name, group=group, method=method, seed=seed,
                **exact_stats(value, truth, support, diff.shape[-2:])))
        counts[ds] += 1
        if counts[ds] == 1:
            visible = (raw, base, actual["equal_low", 0], actual[fixed_demo, SEEDS[0]], second, truth)
            pictures.append((f"{ds} / {name}", [to_rgb(t) for t in visible],
                ("Hazy input", "Native PGL", "Equal low-band merge", f"Selected {fixed_demo}", "Author MCT", "Clear reference")))
        if counts[ds] % 32 == 0:
            print(dict(domain=ds, evaluated=counts[ds]), flush=True)
    if counts != TEST_COUNTS or len(rows) != sum(TEST_COUNTS.values()) * (5 + len(MODES) * len(SEEDS)):
        raise ValueError(f"Physical evaluation rows differ from the registration {counts}")
    means, paired = aggregate(rows)
    atomic_write(ROOT / "individual.json", rows)
    photo = ROOT / "predictions.png"
    draw(photo, pictures)
    report = dict(counts=dict(counts), means=means, paired=paired, amplitudes=cal["amplitudes"], rows=len(rows),
        selection=fixed_demo, test_plan_SHA256=digest(plan), individual_SHA256=digest(ROOT / "individual.json"),
        visual_SHA256=digest(photo), author_pgl=authors, pgl_mct_segformer=anchors[-1], finished_utc=now())
    atomic_write(ROOT / "test.json", report)
    for ds in counts:
        print(ds, {key: round(means[f"{ds}/{key}"]["psnr"], 3) for key in ("native", "equal_low", "equal_full", *MODES,
                      "author_pgl", "author_mct")}, flush=True)


@torch.inference_mode()
def infer(file, output):
    configure_cuda()
    cal = calibrate()
    mode = cal["select_for_demo"]
    model = load_heads((mode,), (SEEDS[0],))[mode, SEEDS[0]]
    anchors = load_anchors()
    with Image.open(file) as handle:
        pixels = np.array(ImageOps.exif_transpose(handle).convert("RGB"))
    native = torch.from_numpy(pixels).permute(2, 0, 1)[None].cuda().float() / 255.
    base, second, features = encode(native, anchors)
    restored = to_rgb(render_rgb(base, model(features, mode), cal["amplitudes"][mode]))
    calibrated_input, _ = resize_image(pixels)
    _, _, combined = two_views(*anchors[2], calibrated_input)
    mask = native_labels(combined, pixels.shape[:2]).byte().cpu().numpy()
    if mask.shape != restored.shape[:2] or not np.isin(mask, range(19)).all():
        raise ValueError("Expected native RGB and Cityscapes class IDs")
    saves = dict(input=pixels, anchor=to_rgb(base), challenger=to_rgb(second), restored=restored,
                 mask=mask, color_mask=paint(mask), overlay=blend(pixels, mask))
    output.mkdir(parents=True, exist_ok=True)
    for name, value in saves.items():
        Image.fromarray(value).save(output / (name + ".png"))
    panel = output / "comparison.png"
    draw(panel, ((file.name, [saves[k] for k in ("input", "anchor", "restored")],
                  ("Hazy input", "Native PGL", f"Coarse correction {mode}")),
                 ("Cityscapes19 segmentation on the raw image", [saves[k] for k in ("input", "color_mask", "overlay")],
                  ("Segmentation input", "Semantic classes", "Semantic overlay"))))
    atomic_write(output / "metadata.json", dict(input=str(file.resolve()), input_SHA256=digest(file),
        size=list(mask.shape[::-1]), model_mode=mode, pgl_SHA256=anchors[-1]["pgl_SHA256"], seed=SEEDS[0],
        amplitude=cal["amplitudes"][mode], calibration_SHA256=digest(ROOT / "calibration.json"),
        output_SHA256={name + ".png": digest(output / (name + ".png")) for name in (*saves, "comparison")},
        finished_utc=now()))
    print(dict(native=mask.shape[::-1], mode=mode, saved=list(saves)), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="step", required=True)
    for stage in ("calibrate", "test", "data-check"):
        sub.add_parser(stage)
    command = sub.add_parser("infer")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.step == "calibrate":
        calibrate()
    elif args.step == "test":
        test()
    elif args.step == "data-check":
        test_count = Counter(row[0] for row in test_data())
        cal_count = {name: sum(1 for _ in stream) for name, stream in calibrate_data()}
        if test_count != TEST_COUNTS or cal_count != dict(lm=64, rw=47, nh=9):
            raise ValueError(f"Evaluation pairing differs {test_count} {cal_count}")
        print(dict(test=dict(test_count), calibration=cal_count), flush=True)
    else:
        if args.input.resolve() in {(args.output / (name + ".png")).resolve() for name in
                ("input", "anchor", "challenger", "restored", "mask", "color_mask", "overlay", "comparison")}:
            cli.error("Output must differ from source")
        infer(args.input, args.output)

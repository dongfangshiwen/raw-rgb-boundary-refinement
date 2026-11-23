"""Calibrate, audit and evaluate the independent RGB correction rule."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageOps
import torch

from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import blend, native_labels, paint
from boundary_refinement.inference.dual_view_inference import two_views
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter, resize_image
from boundary_refinement.evaluation.paired.evaluate_real_transfer import gaussian_ssim
from boundary_refinement.evaluation.photographic.evaluate_region_restore import calibrate_data
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.orthogonal_risk import COVERAGE, MIN_EDIT, ROOT, SEEDS, anchors, coefficients, encode, fitted, nonconformity, pack_predictions, quantile, reconstruct, target
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.baselines.published_runtime import author_module
from boundary_refinement.data.staging.realhyper import CARD as HYPER, COUNTS as HYPER_COUNTS, SOURCE, pairs as hyper_pairs, stage
from boundary_refinement.models.restoration.single_anchor_student import area_psnr, paired_test_data
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


HARM_DB = .10
MODES = ("ordinary_gate", "signed_gate", "scalar_point", "scalar_guarded", "split_point", "split_guarded", "split_equal_energy")
COUNTS = dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=16, nhhaze2=5, bedde=208, **HYPER_COUNTS)
MAIN = "split_guarded"


def prepared(context, state, models):
    norm, cues = context[-2:]
    x = ((cues - state["center"]) / state["scale"]).clamp(-5., 5.)
    return {key: pack_predictions(net, x, norm, key[0]) for key, net in models.items()}


@torch.inference_mode()
def calibrate():
    configure_cuda()
    artifact = ROOT / "calibration.json"
    stamp = digest(ROOT / "completed.json")
    code = digest(__file__)
    if artifact.exists():
        locked = read(artifact)
        if (locked["training_SHA256"], locked["code_SHA256"]) != (stamp, code):
            raise ValueError("The calibrated RGB risk operator changed")
        return locked
    network = anchors()
    center, models = fitted()
    by_group, by_camera = defaultdict(dict), defaultdict(dict)
    examples, counts = defaultdict(list), Counter()
    for data, rows in calibrate_data():
        for _, name, g, fog, clean in rows:
            hazy, clear = as_float(fog), as_float(clean)
            context = encode(hazy, network)
            oracle, _ = target(context[0], clear, context[2])
            norm = context[3]
            capture = name.rsplit("-", 1)[0] if data == "rw" else g
            for (mode, seed), (z, sd) in prepared(context, center, models).items():
                if mode not in ("scalar", "split"):
                    continue
                key = f"{mode}/{seed}"
                s = float(nonconformity(z, sd, oracle, norm, mode))
                for table, key_group in ((by_group, capture), (by_camera, g)):
                    identifier = (data, key_group)
                    table[key][identifier] = max(table[key].get(identifier, 0.), s)
                examples[key].append((z, sd, norm))
            counts[data] += 1
    if counts != dict(lm=64, rw=47, nh=9):
        raise ValueError("The calibration population is incorrect")
    q = {k: quantile(v.values(), COVERAGE) for k, v in by_group.items()}
    q_camera = {k: quantile(v.values(), COVERAGE) for k, v in by_camera.items()}
    if any(value == float("inf") for value in (*q.values(), *q_camera.values())):
        raise ValueError("Insufficient calibration groups for the registered coverage")
    shrink, summary = {}, {}
    for k, v in examples.items():
        mode = k.split("/")[0]
        values, sigma, norms = map(torch.stack, zip(*v, strict=True))
        plain = coefficients(values, sigma, norms, mode)[0].norm(dim=-1)
        certified = coefficients(values, sigma, norms, mode, q[k])[0].norm(dim=-1)
        shrink[k] = float((certified.mean() / plain.mean().clamp_min(1e-8)).clamp(0, 1))
        summary[k] = dict(plain_rms=float(plain.mean()), guarded_rms=float(certified.mean()),
            edit_rate=float((certified >= MIN_EDIT).float().mean()))
    state = dict(training_SHA256=stamp, code_SHA256=code, radii=q, camera_grouped_radii=q_camera,
        output_shrink=shrink, counts=dict(counts), group_counts={k: len(v) for k, v in by_group.items()},
        camera_group_counts={k: len(v) for k, v in by_camera.items()}, calibration_summary=summary, finished_utc=now())
    atomic_write(artifact, state)
    print(dict(radii=q, camera_radii=q_camera, mean_strength=summary), flush=True)
    return state


def comparison(context, state, heads, cal, oracle=None):
    native, delta, basis, norm, _ = context
    results = {("anchor", 0): (native, None, None), ("v51", 0): ((native + delta).clamp(0, 1), None, None)}
    for (arm, seed), (value, scale) in prepared(context, state, heads).items():
        coeff, _ = coefficients(value, scale, norm, arm)
        method = arm + "_point" if arm in ("scalar", "split") else arm
        results[method, seed] = (reconstruct(native, coeff, basis), None, None)
        if arm not in ("scalar", "split"):
            continue
        key = f"{arm}/{seed}"
        limit = cal["radii"][key]
        restricted, bound = coefficients(value, scale, norm, arm, limit)
        coverage = None if oracle is None else bool(nonconformity(value, scale, oracle, norm, arm) <= limit)
        results[arm + "_guarded", seed] = (reconstruct(native, restricted, basis), float(bound), coverage)
        if arm == "split":
            matched = coeff * cal["output_shrink"][key]
            results["split_equal_energy", seed] = (reconstruct(native, matched, basis), None, None)
    return results


def reference_models():
    file = SOURCE / "dehazeformer-b.pth"
    release = read(SOURCE / "author_split.json")
    if digest(file) != release["model_SHA256"]:
        raise ValueError("The HyperHazeOff DehazeFormer-B checkpoint changed")
    model = author_module("author-dehazeformer/dehazeformer.py").dehazeformer_b()
    weights = torch.load(file, weights_only=True, map_location="cpu")["state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
    pgl, pgl_card = load_pgl()
    return pgl, model.cuda().eval(), dict(dehazeformer=release, pgl=pgl_card)


def evaluation_stream():
    for data, name, group, fog, clean, roi in paired_test_data():
        yield data, name, group, fog, clean, roi, False, name
    for row, fog, clean in hyper_pairs():
        yield row["dataset"], row["id"], row["site"], fog, clean, None, row["officially_closest"], row["crop"]


def aggregate(rows):
    groups = defaultdict(list)
    for raw in rows:
        domains = [raw["dataset"]]
        if raw["closest"]:
            domains.append(raw["dataset"] + "_closest")
        for data in domains:
            groups[data, raw["method"]].append(raw)
    values = {}
    for key, row in groups.items():
        hap = [r for r in row if r["covered"] is not None]
        by_site = defaultdict(list)
        for r in hap:
            by_site[r["group"], r["seed"]].append(r["covered"])
        values["/".join(key)] = dict(frames=len({r["id"] for r in row}), psnr=mean(r["psnr"] for r in row),
            deltapsnr=mean(r["delta_db"] for r in row), ssim=mean(r["ssim"] for r in row),
            harmful=mean(r["harmful"] for r in row), edit_fraction=mean(r["edited"] for r in row),
            edit_rms=mean(r["edit_rms"] for r in row),
            edited_delta=mean(r["delta_db"] for r in row if r["edited"]) if any(r["edited"] for r in row) else None,
            group_all_covered=mean(all(v) for v in by_site.values()) if hap else None,
            empirical_frame_coverage=mean(r["covered"] for r in hap) if hap else None)
    controls = ("anchor", "v51", "ordinary_gate", "signed_gate", "scalar_guarded", "split_point", "split_equal_energy")
    by_camera = defaultdict(lambda: defaultdict(list))
    for (data, method), subset in groups.items():
        if method not in (*controls, MAIN):
            continue
        for r in subset:
            by_camera[data, r["group"], r["id"]][method].append((r["psnr"], float(r["harmful"])))
    by_place = defaultdict(lambda: defaultdict(list))
    for (d, place, _), methods in by_camera.items():
        for method in controls:
            for column, name in enumerate(("psnr", "harm")):
                diff = mean(a[column] for a in methods[MAIN]) - mean(a[column] for a in methods[method])
                by_place[d, method, name][place].append(diff)
    return values, {"/".join(k): bootstrap({g: mean(v) for g, v in d.items()}) for k, d in by_place.items()}


@torch.inference_mode()
def test():
    configure_cuda()
    manifest = stage()
    cal = calibrate()
    lock = ROOT / "evaluation_protocol.json"
    reg = dict(code_SHA256=digest(__file__), cal_SHA256=digest(ROOT / "calibration.json"), external_SHA256=digest(HYPER),
        expected=COUNTS, evaluation_threshold_db=HARM_DB, proposal="three-seed mean V51 coarse head with its locked amplitude",
        criteria="Prefer meaningful editing rather than identical fallback. Assess guarded editing against equally shrunk point regression, scalar projection, and ordinary/signed gates on harm and PSNR. Reports include full-frame mathematical coverage, explicit ROI PSNR where needed, official closest pairs, and all external pairs grouped by site.",
        useful_edit_criterion="At least 20% of independently evaluated frames edited by one intensity level of RGB RMS, with reduced harm and at most 0.10 dB decline in mean PSNR relative to the anchor.")
    if lock.exists() and read(lock) != reg:
        raise ValueError("The external evaluation was updated after registration")
    if not lock.exists():
        atomic_write(lock, reg)
    source = anchors()
    norm, heads = fitted()
    pgl, remote, authors = reference_models()
    counts, examples, rows = Counter(), [], []
    for data, name, group, fog, clean, roi, closest, crop in evaluation_stream():
        raw, clear = as_float(fog), as_float(clean)
        context = encode(raw, source)
        anchor, delta, basis, _, _ = context
        oracle, base_error = target(anchor, clear, basis)
        outputs = comparison(context, norm, heads, cal, oracle)
        outputs["raw", 0] = (raw, None, None)
        outputs["author_pgl", 0] = (pgl_rgb(pgl, raw).clamp(0., 1.), None, None)
        if data in HYPER_COUNTS:
            outputs["author_dehazeformer_b_rs", 0] = (author_rgb(remote, raw).clamp(0., 1.), None, None)
        region = roi.cuda() if roi is not None else None
        base_db = area_psnr(anchor, clear, region)
        for (method, seed), (image, lower, covered) in outputs.items():
            score = area_psnr(image, clear, region)
            edit = float((image - anchor).square().mean().sqrt())
            true_gain = float(base_error - (image - clear).square().mean())
            if covered and true_gain + 2e-6 < lower:
                raise ArithmeticError("The improvement violates the covered geometric bound")
            rows.append(dict(dataset=data, id=name, crop=crop, group=group, closest=closest, method=method, seed=seed,
                psnr=score, delta_db=score - base_db, ssim=float(gaussian_ssim(image, clear)[0]),
                harmful=score <= base_db-HARM_DB, edited=edit >= MIN_EDIT, edit_rms=edit,
                covered=covered, lower_full_mse_gain=lower, true_full_mse_gain=true_gain))
        counts[data] += 1
        if data in ("lmhaze", "revide_external", "realhyper_csnc", "realhyper_csso") and counts[data] in (1, 40):
            photos = [raw, anchor, outputs["v51", 0][0], outputs["split_point", SEEDS[0]][0], outputs[MAIN, SEEDS[0]][0], clear]
            examples.append((data + " / " + name, [to_rgb(v) for v in photos], ("Hazy input", "Native anchor", "Frozen regional edit", "Estimated signed edit", "Risk-guarded edit", "Clear reference")))
        if counts[data] % 60 == 0:
            print(dict(domain=data, pairs=counts[data]), flush=True)
    if counts != COUNTS:
        raise ValueError(f"Evaluation pairing changed {counts}")
    stat, intervals = aggregate(rows)
    file = ROOT / "individual.json"
    atomic_write(file, rows)
    draw(ROOT / "predictions.png", examples)
    atomic_write(ROOT / "test.json", dict(counts=dict(counts), closest=manifest["closest"], external_sites=manifest["sites"],
        methods=stat, contrast_intervals=intervals, rows=len(rows), authors=authors, input_SHA256=digest(lock),
        table_SHA256=digest(file), visual_SHA256=digest(ROOT / "predictions.png"), finished_utc=now()))
    for d in (*counts, "realhyper_csnc_closest", "realhyper_csso_closest"):
        print(d, {m: tuple(round(stat[f"{d}/{m}"][k], 4) for k in ("psnr", "harmful", "edit_fraction"))
                  for m in ("raw", "anchor", "v51", *MODES, "author_pgl", "author_dehazeformer_b_rs") if f"{d}/{m}" in stat}, flush=True)


@torch.inference_mode()
def infer(path, destination):
    configure_cuda()
    cal = calibrate()
    if path.resolve() in {(destination / f"{k}.png").resolve() for k in ("input", "restored", "mask", "color_mask", "overlay", "anchor", "regional", "comparison")}:
        raise ValueError("Cannot replace the input with an output")
    with Image.open(path) as handle:
        pixels = np.array(ImageOps.exif_transpose(handle).convert("RGB"))
    raw = as_float(torch.from_numpy(pixels.copy()).permute(2, 0, 1))
    context = encode(raw, anchors())
    normalization, models = fitted()
    outputs = comparison(context, normalization, models, cal)
    restored, gain, _ = outputs[MAIN, SEEDS[0]]
    faces = load_segmenter()[:3]
    _, _, probabilities = two_views(*faces, resize_image(pixels)[0])
    mask = native_labels(probabilities, pixels.shape[:2]).byte().cpu().numpy()
    if mask.shape != restored.shape[-2:] or not np.isin(mask, range(19)).all():
        raise ValueError("The RGB and semantic outputs are not on the same native grid")
    files = dict(input=pixels, anchor=to_rgb(context[0]), regional=to_rgb(outputs["v51", 0][0]),
                 restored=to_rgb(restored), mask=mask, color_mask=paint(mask), overlay=blend(pixels, mask))
    destination.mkdir(parents=True, exist_ok=True)
    for name, image in files.items():
        Image.fromarray(image).save(destination / (name + ".png"))
    panel = destination / "comparison.png"
    draw(panel, ((path.name, [files[k] for k in ("input", "anchor", "restored")], ("Hazy input", "Anchor RGB", "Risk-guarded RGB")),
           ("Cityscapes19 predictions", [files[k] for k in ("input", "color_mask", "overlay")], ("Raw segmentation input", "Predicted mask", "Class overlay"))))
    atomic_write(destination / "metadata.json", dict(input=str(path.resolve()), input_SHA256=digest(path),
        native=list(mask.shape[::-1]), seed=SEEDS[0], mse_improvement_lower_bound=gain,
        observed_edit_rms=float((restored - context[0]).square().mean().sqrt()),
        files={name + ".png": digest(destination / (name + ".png")) for name in (*files, "comparison")}, finished_utc=now()))
    print(dict(native=list(mask.shape[::-1]), lower_bound=gain), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("calibrate", "test", "data-check"):
        sub.add_parser(name)
    command = sub.add_parser("infer")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "calibrate":
        calibrate()
    elif args.stage == "test":
        test()
    elif args.stage == "data-check":
        actual = Counter(row[0] for row in evaluation_stream())
        if actual != COUNTS:
            raise ValueError(f"Registered test population is misstated {actual}")
        print(dict(actual), flush=True)
    else:
        infer(args.input, args.output)

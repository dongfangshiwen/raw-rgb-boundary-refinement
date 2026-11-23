"""Select an update scale on calibration pairs and test frozen checkpoints."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from boundary_refinement.utils import write_csv
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_rgb
from boundary_refinement.evaluation.photographic.evaluate_post_fusion import samples as previous_samples
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.photographic.evaluate_selective_adaptation import samples as v33_samples
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split, gaussian_ssim
from boundary_refinement.training.paired_basis import ALPHAS, ARMS, MAIN, ROOT, SEEDS, SOURCE_FLOOR, TARGET_MARGIN, completed
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


METRICS = ("psnr", "ssim", "edge_mae")
EVAL_BATCH = 3
LABELS = dict(hazy="Hazy input", pgl="PGL-Net-T (author)", local_lora="Local LoRA",
              local_matched="Local coordinates", source_metric="Random metric LoRA",
              marginal="Target marginal basis", shuffled="Shuffled pair basis",
              paired="Same-scene pair basis", clear="Clear reference")


def load_blend(initial, row, alpha):
    model = deepcopy(initial)
    state = torch.load(row["file"], map_location="cpu", weights_only=True)
    mix = model.state_dict()
    for name, original in mix.items():
        revised = state[name].to(original)
        if original.is_floating_point() and not torch.equal(original, revised):
            original.lerp_(revised, alpha)
    return model


def packs(records):
    buckets = defaultdict(list)
    for entry in records:
        key = entry[-2].shape
        chunk = buckets[key]
        chunk.append(entry)
        if len(chunk) == EVAL_BATCH:
            yield chunk
            buckets[key] = []
    yield from (chunk for chunk in buckets.values() if chunk)


def inputs(group):
    return [(torch.stack(column).to("cuda", dtype=torch.float32) / 255.) for column in zip(
        *((r[-2], r[-1]) for r in group), strict=True
    )]


def image_scores(output, truth, metrics):
    error = output - truth
    psnr = -10. * error.square().mean((1, 2, 3)).clamp_min(1e-12).log10()
    if metrics == ("psnr",):
        return psnr[:, None].cpu().tolist()
    ssim = gaussian_ssim(output, truth)
    horizontal = error.diff(dim=-1).abs().mean((1, 2, 3))
    vertical = error.diff(dim=-2).abs().mean((1, 2, 3))
    return torch.stack((psnr, ssim, 127.5 * (horizontal + vertical)), dim=1).cpu().tolist()


def run_scores(records, networks, metrics=("psnr",), demo_ids=()):
    rows, examples = [], []
    for number, group in enumerate(packs(records), 1):
        x, y = inputs(group)
        show = [i for i, r in enumerate(group) if (r[0], r[1]) in demo_ids]
        by_name = {i: {"hazy": as_rgb(x[i:i + 1]), "clear": as_rgb(y[i:i + 1])} for i in show}
        for (method, seed, alpha), model in networks.items():
            prediction = pgl_rgb(model, x).clamp(0., 1.)
            for record, score in zip(group, image_scores(prediction, y, metrics), strict=True):
                dataset, scene, cluster = record[:3]
                rows.append(dict(dataset=dataset, scene=scene, group=cluster, method=method,
                                 seed=seed, alpha=alpha, **dict(zip(metrics, score, strict=True))))
            if seed in (0, SEEDS[0]):
                for i in show:
                    by_name[i][method] = as_rgb(prediction[i:i + 1])
        examples.extend((group[i][:2], images) for i, images in by_name.items())
        if number % 12 == 0:
            print(dict(batches=number, rows=len(rows)), flush=True)
    return rows, examples


def validation_source():
    for dataset, scene, group, x, y, roi in previous_samples("rw_reserve"):
        if roi is not None:
            raise ValueError("This experiment compares full unmasked paired RGB data")
        yield dataset, scene, group, x, y


def means(rows):
    imp = defaultdict(list)
    for r in rows:
        imp[r["dataset"], r["method"], r["alpha"]].append(r["psnr"])
    return {key: float(np.mean(value)) for key, value in imp.items()}


@torch.inference_mode()
def calibrate():
    configure_cuda()
    stored = ROOT / "calibration.json"
    checked = completed()
    fit_key = digest(ROOT / "completed.json")
    if stored.exists():
        value = read(stored)
        if value["fit_SHA256"] != fit_key or value["csv_SHA256"] != digest(ROOT / "calibration.csv"):
            raise ValueError("Calibration and its all-option scores differ")
        if not matches_registered(value['code_SHA256'], digest(__file__)):
            raise ValueError("Evaluation code differs after calibration")
        return value
    initial = load_pgl()[0]
    author_key = ("pgl", 0, 0.)
    networks = {author_key: initial}
    for row in checked["runs"]:
        for alpha in ALPHAS:
            networks[row["arm"], row["seed"], alpha] = load_blend(initial, row, alpha)
    print("Calibrating source quality", flush=True)
    source, _ = run_scores(validation_source(), networks)
    reference = means(source)
    floor = reference["rw_reserve", "pgl", 0.] + SOURCE_FLOOR
    allowed = {mode: [a for a in ALPHAS if reference["rw_reserve", mode, a] >= floor] for mode in ARMS}
    eligible = {k: v for k, v in networks.items() if k == author_key or k[2] in allowed[k[0]]}
    print(dict(eligible=allowed, source_floor=floor), flush=True)
    print("Calibrating target quality", flush=True)
    target, _ = run_scores(data_split("calibrate"), eligible)
    summary = means(target)
    choice = {}
    for mode, candidates in allowed.items():
        options = {0.: summary["calibrate", "pgl", 0.]}
        options.update({a: summary["calibrate", mode, a] for a in candidates})
        choice[mode] = max(options, key=lambda a: (options[a], -a))
    csv_path = ROOT / "calibration.csv"
    write_csv(csv_path, source + target)
    result = dict(fit_SHA256=fit_key, code_SHA256=digest(__file__), csv_SHA256=digest(csv_path),
                  chosen=choice, allowed=allowed, rw_floor=floor,
                  source_means={" / ".join(map(str, k)): v for k, v in reference.items()},
                  target_means={" / ".join(map(str, k)): v for k, v in summary.items()},
                  source_images=len({r["scene"] for r in source}), target_images=len({r["scene"] for r in target}),
                  finished_utc=now())
    atomic_write(stored, result)
    print(result["chosen"], flush=True)
    return result


def summarize(rows):
    per_model = defaultdict(list)
    per_input = defaultdict(lambda: defaultdict(list))
    for r in rows:
        values = [r[k] for k in METRICS]
        per_model[r["dataset"], r["method"], r["seed"]].append(values)
        per_input[r["dataset"], r["group"], r["scene"]][r["method"]].append(values)
    model_summary = {"/".join(map(str, k)): dict(zip(METRICS, np.mean(v, axis=0).tolist(), strict=True))
                     for k, v in per_model.items()}
    by_dataset = defaultdict(list)
    differences = defaultdict(lambda: defaultdict(list))
    camera = defaultdict(list)
    for (dataset, group, _), models in per_input.items():
        values = {mode: np.mean(v, axis=0) for mode, v in models.items()}
        for mode, scores in values.items():
            by_dataset[dataset, mode].append(scores)
            camera[dataset, mode, group].append(scores[0])
            if mode != MAIN:
                differences[dataset, mode][group].append(values[MAIN] - scores)
    summary = {"/".join(k): dict(zip(METRICS, np.mean(v, axis=0).tolist(), strict=True))
               for k, v in by_dataset.items()}
    intervals = {"/".join(k): {metric: bootstrap({g: float(np.mean(v, axis=0)[i]) for g, v in groups.items()})
                              for i, metric in enumerate(METRICS)}
                 for k, groups in differences.items()}
    subgroups = {"/".join(k): float(np.mean(v)) for k, v in camera.items()}
    return summary, model_summary, intervals, subgroups


def draw_examples(path, examples, modes):
    cell_w, cell_h = 244, 208
    grid = Image.new("RGB", (cell_w * len(modes), cell_h * len(examples)), "#f4f5f7")
    text = ImageDraw.Draw(grid)
    for row, ((dataset, scene), images) in enumerate(examples):
        y = row * cell_h
        text.text((6, y + 5), f"{dataset} / {scene}", fill="#17202a")
        for col, mode in enumerate(modes):
            x = col * cell_w
            text.text((x + 6, y + 22), LABELS.get(mode, mode), fill="#17202a")
            image = images[mode].copy()
            image.thumbnail((cell_w - 8, cell_h - 41), Image.Resampling.LANCZOS)
            grid.paste(image, (x + (cell_w - image.width) // 2, y + 38))
    grid.save(path)


def evidence(summary, paired):
    ref = summary["rw_new/pgl"]["psnr"]
    source = summary[f"rw_new/{MAIN}"]["psnr"] - ref
    delta = {mode: summary[f"lmhaze/{MAIN}"]["psnr"] - summary[f"lmhaze/{mode}"]["psnr"]
             for mode in ARMS if mode != MAIN}
    return dict(source_delta=source, source_pass=source >= SOURCE_FLOOR,
                target_deltas=delta, target_pass={mode: v >= TARGET_MARGIN and
                             paired[f"lmhaze/{mode}"]["psnr"]["ci95"][0] > 0
                             for mode, v in delta.items()})


def test_input():
    yield from data_split("lmhaze")
    yield from v33_samples("fresh_rw")


@torch.inference_mode()
def test():
    configure_cuda()
    check = completed()
    cal_path = ROOT / "calibration.json"
    if not cal_path.exists():
        raise ValueError("Calibrate update scales before evaluating the test data")
    cal = calibrate()
    initial, author = load_pgl()
    networks = {("pgl", 0, 0.): initial}
    for row in check["runs"]:
        mode = row["arm"]
        alpha = cal["chosen"][mode]
        networks[mode, row["seed"], alpha] = load_blend(initial, row, alpha) if alpha else initial
    data = list(test_input())
    if len(data) != 176 or len({(r[0], r[1]) for r in data}) != 176:
        raise ValueError("Expected 130 test views of LMHaze and 46 RW test captures")
    demo = [(data[i][0], data[i][1]) for i in (0, 40, 85, 129, 130, 149, 175)]
    rows, examples = run_scores(data, networks, METRICS, demo)
    summary, per_seed, paired, cameras = summarize(rows)
    csv_path = ROOT / "test.csv"
    write_csv(csv_path, rows)
    picture = ROOT / "predictions.png"
    draw_examples(picture, examples, ("hazy", "pgl", "local_lora", "local_matched", "marginal",
                                     "shuffled", MAIN, "clear"))
    result = dict(rows=len(rows), fit_SHA256=digest(ROOT / "completed.json"),
                  calibration_SHA256=digest(cal_path), csv_SHA256=digest(csv_path), image_SHA256=digest(picture),
                  code_SHA256=digest(__file__), author=author, scales=cal["chosen"],
                  summary=summary, per_seed=per_seed, paired=paired, per_group=cameras,
                  evidence=evidence(summary, paired), finished_utc=now())
    atomic_write(ROOT / "test.json", result)
    print({k: round(v["psnr"], 5) for k, v in result["summary"].items()}, flush=True)
    print(result["evidence"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("calibrate", "test"))
    calibrate() if cli.parse_args().stage == "calibrate" else test()

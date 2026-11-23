"""Choose cross-scale correction on calibration and verify fixed predictions."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.models.restoration.cross_scale_rgb import ALPHAS, DETAIL, MODES, ROOT, RW_LIMIT, SEEDS, fitted, packet
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_paired_basis import validation_source
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.photographic.evaluate_selective_adaptation import ROOT as V33
from boundary_refinement.evaluation.photographic.evaluate_selective_adaptation import samples as v33_samples
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import quality, to_rgb
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.staging.stage_revide_holdout import FILE as REVIDE, MANIFEST
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


CONTROLS = ("native", "coarse", "equal")
PRIMARY = (("bands", "tied"), (DETAIL, "copy_detail"), (DETAIL, "shifted_affine"), (DETAIL, "bands"),
           (DETAIL, "native"))


def outputs(features, models):
    corrections = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for (arm, seed), net in models.items():
            corrections[arm, seed] = net(features, arm).float()
    return corrections


def fresh_revide():
    registered = read(MANIFEST)
    if digest(REVIDE) != registered["SHA256"]:
        raise ValueError("External video pixels differ from the registration")
    data = torch.load(REVIDE, map_location="cpu", weights_only=True)
    if len(data) != registered["count"]:
        raise ValueError("Incomplete external video benchmark")
    for row in data:
        yield "revide_external", row["id"], row["group"], row["hazy"], row["clean"]


def test_pairs():
    yield from data_split("lmhaze")
    yield from v33_samples("fresh_rw")
    yield from fresh_revide()


def psnr(a, b):
    return float(-10. * (a - b).square().mean().clamp_min(1e-12).log10())


@torch.inference_mode()
def calibrate():
    configure_cuda()
    root = ROOT / "calibration.json"
    code = digest(__file__)
    fit = digest(ROOT / "completed.json")
    if root.exists():
        current = read(root)
        if current["fit_SHA256"] != fit or current["code_SHA256"] != code:
            raise ValueError("Fixed cross-scale calibration changed")
        return current
    nets = fitted(None)
    anchor, anchor_hash = rgb_checkpoint()
    if anchor_hash != read(ROOT / "protocol.json")["anchor_SHA256"]:
        raise ValueError("Reference network differs")
    table, by_group, counts = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for split, pairs in (("rw", validation_source()), ("lm", data_split("calibrate"))):
        for _, scene, group, fog, target in pairs:
            x, y = as_float(fog), as_float(target)
            context = packet(anchor, x)
            native = context[0]
            base = psnr(native, y)
            table[split, "native", 0.].append(base)
            for (name, seed), delta in outputs(context, nets).items():
                for a in ALPHAS:
                    k = (split, name, a)
                    value = psnr((native + a * delta).clamp(0., 1.), y)
                    table[k].append(value)
                    by_group[k][group].append(value - base)
            counts[split] += 1
            if counts[split] % 20 == 0:
                print(dict(calibration=split, paired=counts[split]), flush=True)
    if dict(counts) != dict(rw=47, lm=64):
        raise ValueError("Calibration population changed")
    avg = {k: float(np.mean(v)) for k, v in table.items()}
    eligible = {(arm, a) for arm in MODES for a in ALPHAS if avg["rw", arm, a] >= avg["rw", "native", 0.] + RW_LIMIT}
    use = {}
    for mode in MODES:
        alternatives = {0.: avg["lm", "native", 0.]}
        alternatives.update({a: avg["lm", mode, a] for m, a in eligible if m == mode})
        use[mode] = max(alternatives, key=lambda a: (alternatives[a], -a))
    document = dict(scales=use, counts=dict(counts), fit_SHA256=fit, code_SHA256=code,
        strict_psnr_source_floor=RW_LIMIT, averages={"/".join(map(str, k)): v for k, v in avg.items()},
        mean_group_effect={"/".join(map(str, k)): {g: float(np.mean(v)) for g, v in groups.items()}
                           for k, groups in by_group.items()}, finished_utc=now())
    atomic_write(root, document)
    print(dict(scales=use, means={mode: [round(avg.get((d, mode, a), avg[d, "native", 0.]), 4)
         for d in ("lm", "rw")] for mode, a in use.items()}), flush=True)
    return document


def visible_edges(x, y):
    confidence = []
    for axis in (-1, -2):
        a = x.diff(dim=axis).abs().mean(1, keepdim=True)
        b = y.diff(dim=axis).abs().mean(1, keepdim=True)
        confidence.append((a >= .03) & (b >= .05))
    return confidence


def metrics(x, y, weights):
    record = quality(x, y)
    residual = x - y
    costs = [(residual.diff(dim=axis).abs() * signal).sum().item() for axis, signal in zip((-1, -2), weights, strict=True)]
    count = sum(int(v.sum()) for v in weights)
    record["visible_edge_mae"] = 255. * sum(costs) / (3 * count) if count else None
    record["visible_edge_pixels"] = count
    return record


def aggregate(rows):
    reported, details = defaultdict(list), defaultdict(list)
    for r in rows:
        values = (r["psnr"], r["edge_mae"], r["visible_edge_mae"] if r["visible_edge_pixels"] else np.nan)
        reported[r["dataset"], r["method"]].append(values)
        details[r["dataset"], r["group"], r["scene"], r["method"]].append(values)
    names = ("psnr", "edge_mae", "visible_edge_mae")
    means = {"/".join(k): {m: float(np.mean(u[np.isfinite(u)])) if np.isfinite(u).any() else None
             for m, u in zip(names, np.asarray(v).T, strict=True)} for k, v in reported.items()}
    examples = defaultdict(dict)
    for (ds, group, scene, method), values in details.items():
        examples[ds, group, scene][method] = np.asarray([
            np.mean(u[np.isfinite(u)]) if np.isfinite(u).any() else np.nan for u in np.asarray(values).T
        ])
    paired = defaultdict(lambda: defaultdict(list))
    for (ds, group, _), models in examples.items():
        for left, right in PRIMARY:
            paired[ds, left, right][group].append(models[left] - models[right])
    between = {}
    for key, groups in paired.items():
        stats = {}
        for index, name in enumerate(names):
            means_by_group = {}
            for group, values in groups.items():
                band = np.asarray(values)[:, index]
                if np.isfinite(band).any():
                    means_by_group[group] = float(np.mean(band[np.isfinite(band)]))
            if means_by_group:
                stats[name] = bootstrap(means_by_group)
        between["/".join(key)] = stats
    return means, between


def draw(path, images):
    labels = ("Hazy input", "Native PGL anchor", "Tied fusion", "Copied native detail", "Local contrast transfer", "Clear reference")
    w, h = 248, 214
    canvas = Image.new("RGB", (w * len(labels), h * len(images)), "#f3f5f8")
    pen = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for i, (title, members) in enumerate(images):
        y = i * h
        pen.text((8, y + 5), title, fill="#223448", font=font)
        for j, (name, rgb) in enumerate(zip(labels, members, strict=True)):
            x = j * w
            pen.text((x + 8, y + 23), name, fill="#223448", font=font)
            panel = Image.fromarray(rgb)
            panel.thumbnail((w - 12, h - 46), Image.Resampling.LANCZOS)
            canvas.paste(panel, (x + (w - panel.width) // 2, y + 42 + (h - 46 - panel.height) // 2))
    canvas.save(path)


@torch.inference_mode()
def test():
    configure_cuda()
    cal_path = ROOT / "calibration.json"
    if not cal_path.exists():
        raise ValueError("Lock the common validation amplitudes first")
    calibration = calibrate()
    scales = calibration["scales"]
    settings = ROOT / "evaluation_protocol.json"
    experiment = dict(calibration_SHA256=digest(cal_path), new_video_SHA256=digest(MANIFEST),
        rw_SHA256=digest(V33 / "rw_data.json"), code_SHA256=digest(__file__), lm_frames=130,
        source_frames=46, other_pairs=read(MANIFEST)["count"], primary_pairs=[list(r) for r in PRIMARY],
        criteria="Check +0.10 dB over copy and shifted controls and a source drop no larger than 0.25 dB; examine whole-image and visible-edge error. Test sets are frozen and coefficients are selected on calibration only.",
        visible_edge_definition="Evaluate gradients at fixed raw-reference edge magnitudes of at least 0.03 and 0.05 intensity respectively. A shared mask gates all restorers. This is a diagnostic on matched references, not an input to inference.")
    if settings.exists() and read(settings) != experiment:
        raise ValueError("The registered external evaluation changed")
    if not settings.exists():
        atomic_write(settings, experiment)
    model = fitted(None)
    anchor, _ = rgb_checkpoint()
    official, author = experts()
    rows, pictures, counts = [], [], defaultdict(int)
    for ds, name, group, fog, clear in test_pairs():
        x, y = as_float(fog), as_float(clear)
        prepared = packet(anchor, x)
        base, coarse = prepared[:2]
        # The MCT and original PGL checkpoints serve as external author references.
        restorations = {("native", 0): base, ("coarse", 0): coarse, ("equal", 0): .5 * (base + coarse),
                        ("author_mct", 0): official[0](x).clamp(0., 1.),
                        ("author_pgl", 0): official[1](x).clamp(0., 1.)}
        restorations.update({(arm, seed): (base + scales[arm] * d).clamp(0., 1.)
                             for (arm, seed), d in outputs(prepared, model).items()})
        weights = visible_edges(x, y)
        for (mode, seed), value in restorations.items():
            rows.append(dict(dataset=ds, group=group, scene=name, method=mode, seed=seed, **metrics(value, y, weights)))
        counts[ds] += 1
        if counts[ds] in (1, 12) and ds in ("lmhaze", "revide_external"):
            chosen = [restorations[m, SEEDS[0]] for m in ("tied", "copy_detail", DETAIL)]
            pictures.append((ds + " / " + name, [to_rgb(t) for t in (x, base, *chosen, y)]))
        if counts[ds] % 40 == 0:
            print(dict(measured=ds, frames=counts[ds]), flush=True)
    if dict(counts) != dict(lmhaze=130, rw_new=46, revide_external=experiment["other_pairs"]):
        raise ValueError("A registered external benchmark is incomplete")
    means, intervals = aggregate(rows)
    data = ROOT / "individual.json"
    atomic_write(data, rows)
    figure = ROOT / "predictions.png"
    draw(figure, pictures)
    atomic_write(ROOT / "test.json", dict(summary=means, paired_intervals=intervals, count=len(rows),
        counts=dict(counts), amplitudes=scales, individual_SHA256=digest(data), calibrate_SHA256=digest(cal_path),
        plan_SHA256=digest(settings), panel_SHA256=digest(figure), authors=author, finished_utc=now()))
    for ds in counts:
        print(ds, {k.split("/")[-1]: {m: round(v[m], 4) for m in ("psnr", "edge_mae")}
                   for k, v in means.items() if k.startswith(ds + "/")}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("calibrate", "test"))
    calibrate() if cli.parse_args().stage == "calibrate" else test()

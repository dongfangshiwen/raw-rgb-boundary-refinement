"""Test transfer with source response matching along a haze-strength change."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from collections import defaultdict
import time

import numpy as np
import torch

from boundary_refinement.utils import set_seed, write_csv
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.training.crop_context import originals
from boundary_refinement.evaluation.photographic.evaluate_paired_basis import (
    inputs, load_blend, packs, run_scores, test_input, validation_source,
)
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split
from boundary_refinement.training.paired_basis import ALPHAS, SOURCE_FLOOR, TARGET_MARGIN, state_copy, stream
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.selective_adaptation import adapt, layout, materialize, verify_changes
from boundary_refinement.training.source_replay import PROTOCOL as REPLAY, band_parts
from boundary_refinement.data.staging.stage_lmhaze import CARD as LMH, TRAIN_CARD
from boundary_refinement.data.staging.stage_postfusion_holdout import RW_CARD
from boundary_refinement.training.structure_transfer import PAIRS, SEEDS, STEPS, objective, training_scenes
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/haze-response-v35"
ARMS = ("supervised", "point", "shuffled_response", "paired_response")
MAIN = "paired_response"
PROBE = .60


def linear_rgb(x):
    return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).pow(2.4))


def srgb(x):
    return torch.where(x <= .0031308, 12.92 * x, 1.055 * x.clamp_min(0.).pow(1. / 2.4) - .055)


def attenuated(haze, clean, strength):
    return srgb(torch.lerp(linear_rgb(clean), linear_rgb(haze), strength))


def teacher_cost(error, strength, mode):
    a, b = error.split(PAIRS)
    point = .5 * (a.abs().mean() + b.abs().mean())
    if mode == "point":
        return 2. * point
    if mode == "shuffled_response":
        a = a.roll(1, 0)
    # The finite change uses the same haze/clear pair in the proposed arm.
    # The common point term anchors shifts shared by both haze strengths.
    response = ((b - a) * (.25 / (1. - strength))).abs().mean()
    return point + response


def source_loss(error):
    low, high = band_parts(error)
    return (low.square() + 1e-6).sqrt().mean() + .2 * high.abs().mean()


def fit(smoke=False):
    configure_cuda()
    teacher, author = load_pgl()
    student = load_pgl()[0]
    reference = state_copy(student)
    param_names = set(dict(student.named_parameters()))
    stages = layout(student)
    target, source = training_scenes(), originals()
    capacity = sum(s["budget"] for s in stages)
    steps = STEPS * 2
    code = [__file__, SOURCE_ROOT / "boundary_refinement/training/paired_basis.py", SOURCE_ROOT / "boundary_refinement/evaluation/photographic/evaluate_paired_basis.py"]
    registration = dict(
        hypotheses="Match the frozen source model response to a local reduction in haze strength alongside point distillation. Test lower held response drift and better target transfer at a fixed source PSNR budget.",
        arms=list(ARMS), seed=list(SEEDS), updates=steps, trainable=capacity, author=author,
        code={str(path): digest(path) for path in code},
        data={str(path.relative_to(REPO)): digest(path) for path in (REPLAY, LMH, TRAIN_CARD, RW_CARD)},
        training="All arms see the same two source views and both target views. Source supervision uses the real clear reference. The lighter view interpolates the matching clear and hazy image in linearized RGB. This proxy scales the per-pixel haze contribution assuming fixed lighting and a linear camera response. Frozen source teacher, frozen BN, matched local LoRA, AdamW LR 0.00012 to 0.00002 and BF16 CUDA.",
        controls="Supervised replay already uses the lighter views. Point distillation applies two units of direct matching. Response arms use one unit of point matching plus a pseudo-slope between weak and original views, aligned or scene-shuffled. All arms compute the frozen prediction on the same source views.",
        source_alpha=[0., *ALPHAS], source_floor_db=SOURCE_FLOOR, target_margin_db=TARGET_MARGIN,
        probe_strength=PROBE, assessment="Reuse predetermined development and test scene IDs from V34. The source response metric evaluates the actual teacher drift on test images after the registered held strength change. It is a diagnostic on an artificial perturbation. Real source PSNR and real target PSNR remain primary.",
    )
    protocol = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if protocol.exists() and read(protocol) != registration:
            raise ValueError("Registered haze response protocol changed")
        if not protocol.exists():
            atomic_write(protocol, registration)
    rows = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in ARMS:
            path = ROOT / f"{mode}-{seed}.pth"
            receipt = path.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["protocol_SHA256"] != digest(protocol) or row["checkpoint_SHA256"] != digest(path):
                    raise ValueError("Haze response model provenance changed")
                rows.append(row)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 35023)
            student.load_state_dict(reference, strict=True)
            selected = adapt(student, "local_lora", stages)
            params = [p for p in student.parameters() if p.requires_grad]
            if sum(p.numel() for p in params) != capacity:
                raise ValueError("Local LoRA budget changed")
            optimizer = torch.optim.AdamW(params, lr=.00012, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00002)
            started = time.perf_counter()
            for step, (haze, dest, src) in enumerate(stream(target, source, seed), 1):
                a = (.68 + .22 * torch.rand((PAIRS, 1, 1, 1), generator=rng)).to(src)
                weak = attenuated(haze[2 * PAIRS:], src, a)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = student(torch.cat((haze, weak))).float()
                    with torch.no_grad():
                        teacher_out = teacher(torch.cat((haze[2 * PAIRS:], weak))).float()
                y, src_y = out.split(2 * PAIRS)
                objective_y, _ = objective(y, dest, None, "point")
                loss = objective_y + source_loss(src_y - torch.cat((src, src)))
                if mode != "supervised":
                    loss = loss + teacher_cost(src_y - teacher_out, a, mode)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                if smoke and step == 2:
                    break
            if not smoke and step != steps:
                raise ValueError("Training updates differed from the protocol")
            state = materialize(student, selected)
            changed = verify_changes(reference, state, param_names, selected, "local_lora")
            train = dict(arm=mode, seed=seed, updates=step, trainable=capacity, changed=changed,
                         seconds=round(time.perf_counter() - started, 2), loss=loss.item())
            print(train, flush=True)
            if not smoke:
                torch_save(path, state)
                row = dict(train, file=str(path), checkpoint_SHA256=digest(path),
                           protocol_SHA256=digest(protocol), finished_utc=now())
                atomic_write(receipt, row)
                rows.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(
            runs=rows, updates=sum(row["updates"] for row in rows), protocol_SHA256=digest(protocol),
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now(),
        ))


def completed():
    info = read(ROOT / "completed.json")
    if info["protocol_SHA256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Haze response training protocol changed")
    if {(r["arm"], r["seed"]) for r in info["runs"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Haze response is missing a matched run")
    for run in info["runs"]:
        if run["checkpoint_SHA256"] != digest(run["file"]):
            raise ValueError("Haze response checkpoint changed")
    return info


def averages(scores):
    means = defaultdict(list)
    for r in scores:
        means[r["dataset"], r["method"], r["alpha"]].append(r["psnr"])
    return {k: float(np.mean(v)) for k, v in means.items()}


@torch.inference_mode()
def calibrate():
    configure_cuda()
    key = digest(ROOT / "completed.json")
    card = ROOT / "calibration.json"
    if card.exists():
        value = read(card)
        if value["fit_SHA256"] != key or value["score_SHA256"] != digest(ROOT / "calibration.csv"):
            raise ValueError("Haze-response calibration differs")
        return value
    author = load_pgl()[0]
    nets = {("pgl", 0, 0.): author}
    for row in completed()["runs"]:
        for a in ALPHAS:
            nets[row["arm"], row["seed"], a] = load_blend(author, row, a)
    src, _ = run_scores(validation_source(), nets)
    sr = averages(src)
    floor = sr["rw_reserve", "pgl", 0.] + SOURCE_FLOOR
    allowed = {(m, a) for m in ARMS for a in ALPHAS if sr["rw_reserve", m, a] >= floor}
    eligible = {k: v for k, v in nets.items() if k[0] == "pgl" or (k[0], k[2]) in allowed}
    target, _ = run_scores(data_split("calibrate"), eligible)
    tr = averages(target)
    reference = tr["calibrate", "pgl", 0.]
    selected = {}
    for method in ARMS:
        alpha = max((a for m, a in allowed if m == method), default=0.,
                    key=lambda a: (tr["calibrate", method, a], -a))
        if not alpha or tr["calibrate", method, alpha] <= reference:
            alpha = 0.
        selected[method] = alpha
    table = ROOT / "calibration.csv"
    write_csv(table, src + target)
    levels = dict(chosen=selected, fit_SHA256=key, source_floor=floor, score_SHA256=digest(table),
                  source={"/".join(map(str, k)): v for k, v in sr.items()},
                  target={"/".join(map(str, k)): v for k, v in tr.items()}, finished_utc=now())
    atomic_write(card, levels)
    print(dict(chosen=selected), flush=True)
    return levels


def response_scores(data, author, adapted):
    rows = []
    for group in packs(data):
        fog, clear = inputs(group)
        mix = attenuated(fog, clear, PROBE)
        reference = pgl_rgb(author, mix).clamp(0., 1.) - pgl_rgb(author, fog).clamp(0., 1.)
        for (arm, seed, alpha), model in adapted.items():
            if arm == "pgl":
                continue
            span = pgl_rgb(model, mix).clamp(0., 1.) - pgl_rgb(model, fog).clamp(0., 1.)
            rmse = (span - reference).square().mean((1, 2, 3)).sqrt().cpu().tolist()
            for record, value in zip(group, rmse, strict=True):
                dataset, scene, cluster = record[:3]
                rows.append(dict(dataset=dataset, scene=scene, group=cluster, method=arm,
                                 seed=seed, alpha=alpha, response_rmse=value))
    return rows


@torch.inference_mode()
def test():
    configure_cuda()
    receipt = ROOT / "calibration.json"
    if not receipt.exists():
        raise ValueError("Calibrate on the development data first")
    calibration = calibrate()
    author = load_pgl()[0]
    models = {("pgl", 0, 0.): author}
    for row in completed()["runs"]:
        method = row["arm"]
        a = calibration["chosen"][method]
        models[method, row["seed"], a] = load_blend(author, row, a) if a else author
    data = list(test_input())
    rgb, _ = run_scores(data, models)
    response = response_scores((r for r in data if r[0] == "rw_new"), author, models)
    table, probe = ROOT / "test.csv", ROOT / "response.csv"
    write_csv(table, rgb)
    write_csv(probe, response)
    quality = {f"{d}/{m}": v for (d, m, _), v in averages(rgb).items()}
    response_by_arm = {m: float(np.mean([r["response_rmse"] for r in response if r["method"] == m])) for m in ARMS}
    paired = defaultdict(lambda: defaultdict(list))
    for entry in rgb:
        paired[entry["dataset"], entry["group"], entry["scene"]][entry["method"]].append(entry["psnr"])
    interval = {}
    for method in ARMS[:-1]:
        groups = defaultdict(list)
        for (d, g, _), preds in paired.items():
            if d == "lmhaze":
                groups[g].append(np.mean(preds[MAIN]) - np.mean(preds[method]))
        interval[method] = bootstrap({g: np.mean(v) for g, v in groups.items()})
    source = quality[f"rw_new/{MAIN}"] - quality["rw_new/pgl"]
    delta = {m: quality[f"lmhaze/{MAIN}"] - quality[f"lmhaze/{m}"] for m in ARMS[:-1]}
    evidence = dict(source_delta=source, source_pass=source >= SOURCE_FLOOR, target_delta=delta,
                    target_pass={m: v >= TARGET_MARGIN and interval[m]["ci95"][0] > 0 for m, v in delta.items()})
    result = dict(rgb=quality, response=response_by_arm, intervals=interval, scales=calibration["chosen"],
                  source_probe=PROBE, evidence=evidence, rows=len(rgb), response_rows=len(response),
                  csv_SHA256=digest(table), response_SHA256=digest(probe), calibration_SHA256=digest(receipt),
                  fit_SHA256=digest(ROOT / "completed.json"), finished_utc=now())
    atomic_write(ROOT / "test.json", result)
    print(dict(rgb=quality, response=response_by_arm, evidence=evidence), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("smoke", "fit", "calibrate", "test"))
    stage = cli.parse_args().stage
    if stage in ("smoke", "fit"):
        fit(smoke=stage == "smoke")
    elif stage == "calibrate":
        calibrate()
    else:
        test()

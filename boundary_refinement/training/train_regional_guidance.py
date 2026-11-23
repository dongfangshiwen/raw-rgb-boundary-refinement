"""Train three video-disjoint folds with two seeds and matched local selectors."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
import gc
import hashlib
import json
import math
import time

import numpy as np
import torch

from boundary_refinement.utils import runtime_metadata, set_seed, write_csv
from boundary_refinement.data.cohorts.audit_camvid_groups import OUT as VIDEO_AUDIT
from boundary_refinement.data.camvid_data import DATA, make_batch, photo_loss, semantic_loss
from boundary_refinement.evaluation.semantic.evaluate_regional_guidance import evaluate, image_measures
from boundary_refinement.data.provenance import REPO, digest, now, read, write
from boundary_refinement.data.regional_data import HEIGHT, VIEWS, WIDTH, boundary_band, calibration_batch, prepare_grouped_data
from boundary_refinement.training.regional_guidance import (
    CONTROLS, SELECTORS, TILE, RegionalGuidance, TileSelector, apply_choice, choose_tiles,
    selector_features, selector_loss, tile_gain,
)
from boundary_refinement.models.joint.structure_joint import AUTHOR
from boundary_refinement.models.joint.task_compatible import PRETRAINED


ROOT = REPO / "results/regional-guidance-v18"
CORE_EPOCHS = 10
CORE_BATCH = 4
CALIBRATION_REPEATS = 3
SELECTOR_EPOCHS = 14
SELECTOR_BATCH = 24
COVERAGE = 0.5
SEEDS = (42, 43)


def atomic_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write(temp, value)
    temp.replace(path)


def torch_save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def configure_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment uses the installed CUDA PyTorch with BF16 support")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def register(folds):
    sources = [
        DATA, VIDEO_AUDIT / "audit.json", AUTHOR / "NAFNet_arch.py", AUTHOR / "arch_util.py",
        *sorted(path for path in PRETRAINED.iterdir() if path.is_file()),
        *(SOURCE_ROOT / name for name in (
            "boundary_refinement/training/regional_guidance.py", "boundary_refinement/data/regional_data.py",
            "boundary_refinement/training/train_regional_guidance.py", "boundary_refinement/evaluation/semantic/evaluate_regional_guidance.py",
            "boundary_refinement/data/camvid_data.py", "boundary_refinement/data/data_alignment.py",
            "boundary_refinement/models/joint/decision_joint.py", "boundary_refinement/models/joint/affine_split.py",
            "boundary_refinement/models/joint/dual_coordinate.py", "boundary_refinement/models/joint/structure_joint.py", "boundary_refinement/metrics/segmentation.py",
        )),
    ]
    plan = dict(
        study="Measure local semantic guidance by the RGB error change caused by using it",
        train_scope="CamVid 468 development frames only. Three whole held-out videos. Core and selector training use nonoverlapping frames from the other two videos. No independent test predictions.",
        folds=[dict(video=f["held_video"], ids=f["ids"]) for f in folds],
        seeds=list(SEEDS), core_epochs=CORE_EPOCHS, core_batch=CORE_BATCH,
        calibration_repeats=CALIBRATION_REPEATS, selector_epochs=SELECTOR_EPOCHS, selector_batch=SELECTOR_BATCH,
        source_grid=[WIDTH, HEIGHT], tile=TILE, selected_tile_fraction=COVERAGE, conditions=list(VIEWS),
        geometry="Bilinear RGB resize, nearest original label, class 255 void. Boundary metrics use all-valid class changes dilated by three pixels.",
        stage_core="Fresh NAF/affine base and ImageNet MiT-B0 per video and seed. Encoder frozen first epoch. Core, RGB-only and semantic-conditioned corrections receive common RGB loss. Correction losses detach base and semantic probabilities. Every core and selector uses the final epoch.",
        common_selector_inputs="Pooled RGB, variation, candidate difference, semantic probabilities, their local variation and entropy. No reference or transmission.",
        intervention="Alternatives share the RGB anchor and the same-size correction heads. Use the RGB-only correction at tile margins. At inference, select exactly 50% of tiles per image for learned selectors, entropy control and diagnostic oracle.",
        selector_targets={
            "direct": "Optimize selected RGB squared error with straight-through top half",
            "gain": "Regress measured tile MSE RGB minus semantic",
            "paired_gain": "For two independently synthesized spatial views of the same scene, learn their minimum gain. Camera and uniform views use their own gains.",
        },
        selector_calibration="Deterministic middle 20% of each training video, exclude one neighboring frame on both sides from core fit. Pair spatial views without shifting geometry. Replay the same samples and batches for every selector.",
        comparisons=list(CONTROLS) + ["oracle"],
        screen=dict(
            primary_views=["spatial", "spatial_alt"], equal_video_seed_view_weight=True,
            candidate="gain", required_boundary_psnr_delta=0.10, required_full_psnr_delta=0.0,
            comparisons=["rgb", "entropy", "direct"], report_all_folds_and_both_seeds=True,
            scope="Exploratory screen on three source videos. Paired gain is exploratory and summarized regardless of direction. No adaptive model choice or external testing.",
        ),
        files_SHA256={str(p): digest(p) for p in sources},
        pytorch=runtime_metadata(torch.device("cuda")),
    )
    sha = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = ROOT / "plan.json"
    if path.exists():
        saved = read(path)
        if saved["plan_SHA256"] != sha:
            raise ValueError("Current code, data, or runtime differs from the frozen regional study")
        return saved
    plan.update(plan_SHA256=sha, registered_utc=now())
    atomic_write(path, plan)
    return plan


def core_optimizer(model):
    params = []
    for is_encoder in (True, False):
        selected = [p for name, p in model.named_parameters() if name.startswith("encoder.") == is_encoder]
        params.append(dict(params=selected, lr=6e-5 if is_encoder else 3e-4))
    return torch.optim.AdamW(params, weight_decay=0.01)


def core_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        anchor, rgb, semantic, _, logits, _ = model(batch["image"])
        reference, valid = batch["reference"], batch["valid"]
        losses = (semantic_loss(logits, batch["target"]), photo_loss(anchor, reference, valid),
                  photo_loss(rgb, reference, valid), photo_loss(semantic, reference, valid))
        objective = sum(losses)
    if not bool(torch.isfinite(objective)):
        raise ValueError("Nonfinite core objective")
    objective.backward()
    for group in (model.semantic_parameters(), model.rgb_net.parameters(), model.global_affine.parameters(),
                  model.rgb_correction.parameters(), model.semantic_correction.parameters()):
        torch.nn.utils.clip_grad_norm_(group, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return [float(loss.detach().cpu()) for loss in losses]


def fit_core(data, fold, seed, model, directory):
    optimizer = core_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CORE_EPOCHS, eta_min=1e-5)
    generator = torch.Generator().manual_seed(seed + 18_300_000)
    indices = torch.tensor(fold["indices"]["core"])
    updates = 0
    history = []
    clock = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, CORE_EPOCHS + 1):
        model.train()
        model.encoder.requires_grad_(epoch > 1)
        if epoch == 1:
            model.encoder.eval()
        batches = indices[torch.randperm(len(indices), generator=generator)].split(CORE_BATCH)
        measures = []
        for chosen in batches:
            measures.append(core_step(model, optimizer, make_batch(data, chosen, generator)))
            updates += 1
        scheduler.step()
        record = dict(video=fold["held_video"], seed=seed, stage="core", epoch=epoch, updates=updates,
                      losses=np.mean(measures, axis=0).tolist(), seconds=round(time.perf_counter() - clock, 2),
                      peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3), updated_utc=now())
        history.append(record)
        atomic_write(directory / "core_curve.json", history)
        atomic_write(ROOT / "progress.json", record)
        print(record, flush=True)
    expected = math.ceil(len(indices) / CORE_BATCH) * CORE_EPOCHS
    if updates != expected:
        raise ValueError("Wrong core update count")
    model.eval()
    return dict(updates=updates, seconds=history[-1]["seconds"], peak_cuda_GiB=history[-1]["peak_GiB"])


@torch.no_grad()
def calibration_arrays(data, fold, seed, model, repeats):
    model.eval()
    generator = torch.Generator().manual_seed(seed + 18_400_000)
    index = torch.tensor(fold["indices"]["calibration"])
    features, observed, both = [], [], []
    for _ in range(repeats):
        for chosen in index.split(CORE_BATCH):
            images, reference = calibration_batch(data, chosen, generator)
            anchor, rgb, _, guided, _, prob = model(images)
            gain = tile_gain(rgb, guided, reference)
            gains = gain.reshape(4, len(chosen), *gain.shape[1:])
            pair = torch.minimum(gains[2], gains[3])
            robust = torch.stack((gains[0], gains[1], pair, pair)).flatten(0, 1)
            features.append(selector_features(images, anchor, rgb, guided, prob)[0])
            observed.append(gain)
            both.append(robust)
    return tuple(torch.cat(items) for items in (features, observed, both))


def fit_selectors(data, fold, seed, model, directory):
    started = time.perf_counter()
    arrays = calibration_arrays(data, fold, seed, model, CALIBRATION_REPEATS)
    features, gains, pairs = arrays
    if not all(bool(torch.isfinite(value).all()) for value in arrays):
        raise ValueError("Nonfinite selector calibration arrays")
    selectors = {name: TileSelector(seed).cuda() for name in SELECTORS}
    optimizers = {name: torch.optim.AdamW(module.parameters(), lr=6e-4, weight_decay=1e-3)
                  for name, module in selectors.items()}
    generator = torch.Generator().manual_seed(seed + 18_500_000)
    history = []
    for epoch in range(1, SELECTOR_EPOCHS + 1):
        sampler = torch.randperm(len(features), generator=generator).split(SELECTOR_BATCH)
        losses = {name: [] for name in selectors}
        for index in sampler:
            for name, module in selectors.items():
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                score = module(features[index])
                loss = selector_loss(name, score, gains[index], pairs[index], COVERAGE)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Nonfinite selector objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses[name].append(float(loss.detach()))
        history.append(dict(epoch=epoch, updates_per_selector=epoch * len(sampler),
                            means={name: float(np.mean(values)) for name, values in losses.items()}))
    atomic_write(directory / "selector_curve.json", history)
    final = history[-1]
    record = dict(video=fold["held_video"], seed=seed, stage="selectors_finished",
                  calibration_images=len(features), seconds=round(time.perf_counter() - started, 2),
                  **final, updated_utc=now())
    atomic_write(ROOT / "progress.json", record)
    print(record, flush=True)
    return {name: module.eval() for name, module in selectors.items()}, record


def smoke(data, fold):
    """Check the actual CUDA graph and exact switch arithmetic before the study."""
    set_seed(SEEDS[0])
    model = RegionalGuidance(SEEDS[0]).cuda()
    sample = fold["indices"]["core"][:CORE_BATCH]
    generator = torch.Generator().manual_seed(18_012)
    losses = core_step(model, core_optimizer(model), make_batch(data, sample, generator))
    model.eval()
    images, reference = calibration_batch(data, sample, generator)
    with torch.no_grad():
        anchor, rgb, _, guided, _, probabilities = model(images)
        features, _ = selector_features(images, anchor, rgb, guided, probabilities)
        gain = tile_gain(rgb, guided, reference)
        selection = choose_tiles(gain, COVERAGE)
        output = apply_choice(rgb, guided, selection)
        labels = data["mask"][sample].cuda().long().repeat(4, 1, 1)
        before, _ = image_measures(rgb, reference, boundary_band(labels))
        after, _ = image_measures(output, reference, boundary_band(labels))
        error = (before - after - (selection * gain).flatten(1).mean(1)).abs().max().item()
        if error > 2e-7 or not bool((selection.flatten(1).mean(1) == COVERAGE).all()):
            raise ValueError("Wrong coverage or arithmetic on actual CamVid calibration views")
    for name in SELECTORS:
        module = TileSelector(SEEDS[0]).cuda()
        loss = selector_loss(name, module(features), gain, gain, COVERAGE)
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()):
            raise ValueError("Selector lacks a finite gradient")
    info = dict(all_passed=True, tf32=False, core_step_losses=losses, core_updates_for_smoke=1,
                selector_optimizer_updates_for_smoke=0, training_images=len(sample) * 3,
                calibration_test_images=len(images), max_tile_error=error, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                updated_utc=now())
    atomic_write(ROOT / "cuda_smoke.json", info)
    print(info, flush=True)


def fit(data, fold, seed, plan):
    directory = ROOT / f"video-{fold['held_video']}" / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "result.json"
    core_path, full_path, table = (directory / name for name in ("core.pth", "selected.pth", "by_image.csv"))
    if result.exists():
        record = read(result)
        if (record["plan_SHA256"] != plan["plan_SHA256"] or digest(core_path) != record["core_SHA256"]
                or digest(full_path) != record["checkpoint_SHA256"] or digest(table) != record["by_image_SHA256"]):
            raise ValueError("Prior fold artifacts changed")
        return record
    set_seed(seed + 18_600_000)
    model = RegionalGuidance(seed, pretrained=not core_path.exists()).cuda()
    if core_path.exists():
        value = torch.load(core_path, map_location="cpu", weights_only=True)
        if value["plan_SHA256"] != plan["plan_SHA256"] or value["seed"] != seed or value["video"] != fold["held_video"]:
            raise ValueError("Intermediate core differs from current plan")
        model.load_state_dict(value["model"], strict=True)
        core_fit = value["training"]
    else:
        core_fit = fit_core(data, fold, seed, model, directory)
        torch_save(core_path, dict(model=model.state_dict(), seed=seed, video=fold["held_video"],
                                   training=core_fit, plan_SHA256=plan["plan_SHA256"]))
    model.requires_grad_(False).eval()
    selectors, selector_fit = fit_selectors(data, fold, seed, model, directory)
    summary, rows = evaluate(model, selectors, data, fold, seed, COVERAGE)
    torch_save(full_path, dict(format="regional-guidance-v18", seed=seed, held_video=fold["held_video"],
                               plan_SHA256=plan["plan_SHA256"], model=model.state_dict(),
                               selectors={name: module.state_dict() for name, module in selectors.items()}))
    write_csv(table, rows)
    summary.update(plan_SHA256=plan["plan_SHA256"], core=core_fit, selector=selector_fit,
                   core_SHA256=digest(core_path), checkpoint=str(full_path), checkpoint_SHA256=digest(full_path),
                   by_image=str(table), by_image_SHA256=digest(table),
                   model_parameters=sum(p.numel() for p in model.parameters()),
                   selector_parameters=sum(p.numel() for p in selectors["gain"].parameters()), finished_utc=now())
    atomic_write(result, summary)
    progress = dict(video=fold["held_video"], seed=seed, stage="complete", updated_utc=now(),
                    spatial={name: [round(item["psnr"], 4), round(item["boundary_psnr"], 4)]
                             for name, item in summary["conditions"]["spatial"]["methods"].items()})
    atomic_write(ROOT / "progress.json", progress)
    print(progress, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    data, folds = prepare_grouped_data()
    print(dict(video_splits={f["held_video"]: {n: len(i) for n, i in f["ids"].items()} for f in folds}), flush=True)
    if args.smoke:
        smoke(data, folds[0])
        return
    if not read(ROOT / "cuda_smoke.json")["all_passed"]:
        raise ValueError("CUDA feasibility check failed")
    plan = register(folds)
    completed = []
    for seed in SEEDS:
        for fold in folds:
            completed.append(fit(data, fold, seed, plan))
            gc.collect()
            torch.cuda.empty_cache()
    record = dict(all_registered_runs_complete=True, plan_SHA256=plan["plan_SHA256"], runs=completed,
                  core_fits=len(completed), core_epochs=CORE_EPOCHS * len(completed),
                  effective_core_updates=sum(r["core"]["updates"] for r in completed),
                  selector_fits=len(completed) * len(SELECTORS),
                  effective_selector_updates=sum(r["selector"]["updates_per_selector"] for r in completed) * len(SELECTORS),
                  new_test_predictions=0, finished_utc=now())
    atomic_write(ROOT / "completed.json", record)
    print({k: v for k, v in record.items() if k != "runs"}, flush=True)


if __name__ == "__main__":
    main()

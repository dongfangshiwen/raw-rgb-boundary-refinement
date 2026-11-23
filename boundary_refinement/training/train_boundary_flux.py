"""Compare learned refinements with RGB, predicted, displaced and annotated edges."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed, write_csv
from boundary_refinement.models.joint.boundary_flux import METHODS, BoundaryRefiner, edges, guidance, probe_math
from boundary_refinement.data.camvid_data import DATA, fog, photo_loss
from boundary_refinement.data.depth_correlated_fog import ROOT as V20, fixed_view, load_proxy
from boundary_refinement.evaluation.semantic.evaluate_regional_guidance import image_measures, psnr
from boundary_refinement.inference.infer_regional_guidance import load_checkpoint
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.regional_data import HEIGHT, WIDTH, VIEWS, boundary_band, prepare_grouped_data, validation_view
from boundary_refinement.training.regional_guidance import CLASSES
from boundary_refinement.training.train_regional_guidance import (
    ROOT as V18, CORE_BATCH, atomic_write, configure_cuda, torch_save,
)


ROOT = REPO / "results/boundary-flux-v21"
EPOCHS = 8
ALL_VIEWS = (*VIEWS, "aligned_dense")
METRICS = ("psnr", "boundary_psnr", "boundary_gradient_mae")


def make_batch(data, ids, generator):
    clear = data["raw"][ids].cuda().float() / 255
    target = data["mask"][ids].cuda().long()
    depth = data["proxy_distance"][ids].cuda().float()
    n = len(clear)
    flipped = (torch.rand(n, generator=generator) > 0.5).cuda()[:, None, None, None]
    clear = torch.where(flipped, clear.flip(-1), clear)
    depth = torch.where(flipped, depth.flip(-1), depth)
    target = torch.where(flipped[:, 0], target.flip(-1), target)
    air = (0.75 + 0.23 * torch.rand(n, 3, 1, 1, generator=generator)).cuda()
    t = (0.25 + 0.65 * torch.rand(n, 1, 1, 1, generator=generator)).cuda()
    field = F.interpolate(torch.rand(n, 1, 5, 7, generator=generator).cuda(), (HEIGHT, WIDTH),
                          mode="bicubic", align_corners=False)
    spatial_t = (t + 0.45 * (field - 0.5)).clamp(0.15, 0.95)
    beta = (0.7 + 1.1 * torch.rand(n, 1, 1, 1, generator=generator)).cuda()
    optical = beta * (0.15 + 0.85 * depth) * (0.87 + 0.26 * field.clamp(0, 1))
    hazy = torch.cat((clear, fog(clear, t, air), fog(clear, spatial_t, air), fog(clear, (-optical).exp(), air)))
    return (hazy * 255).round() / 255, clear.repeat(4, 1, 1, 1), target.repeat(4, 1, 1)


@torch.no_grad()
def common_inputs(core, image, labels):
    anchor, _, _, _, logits, _ = core(image)
    return anchor, logits, guidance(image, logits, labels)


def loss_for(output, reference, band, valid):
    boundary_error = (output - reference).abs().mean(1)
    boundary_average = (boundary_error * band).sum() / band.sum().clamp_min(1)
    return photo_loss(output, reference, valid) + 0.2 * boundary_average


def update(core, heads, optimizers, batch):
    image, reference, labels = batch
    anchor, _, guides = common_inputs(core, image, labels)
    band = boundary_band(labels)
    valid = torch.ones_like(labels, dtype=torch.bool)
    outcomes = {}
    for method, head in heads.items():
        optimizer = optimizers[method]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored, _, _ = head(image, anchor, guides[method])
            objective = loss_for(restored, reference, band, valid)
        if not bool(torch.isfinite(objective)):
            raise ValueError("Nonfinite restoration loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        outcomes[method] = float(objective.detach())
    return outcomes


def fit_refiners(core, data, fold, seed, folder):
    heads = {method: BoundaryRefiner(seed, method).cuda() for method in METHODS}
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=0.01)
                  for name, head in heads.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=5e-5)
                  for name, opt in optimizers.items()}
    index = torch.tensor(fold["indices"]["core"])
    generator = torch.Generator().manual_seed(seed + 21_200_000)
    started = time.perf_counter()
    training = []
    updates = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, EPOCHS + 1):
        statistics = []
        for ids in index[torch.randperm(len(index), generator=generator)].split(CORE_BATCH):
            statistics.append(update(core, heads, optimizers, make_batch(data, ids, generator)))
            updates += 1
        for scheduler in schedulers.values():
            scheduler.step()
        row = dict(video=fold["held_video"], seed=seed, epoch=epoch, steps_per_head=updates,
                   losses={name: float(np.mean([item[name] for item in statistics])) for name in METHODS},
                   peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   seconds=round(time.perf_counter() - started, 3), updated_utc=now())
        atomic_write(ROOT / "progress.json", row)
        training.append(row)
        print(row, flush=True)
    if updates != EPOCHS * math.ceil(len(index) / CORE_BATCH):
        raise ValueError("Incomplete boundary refinement updates")
    atomic_write(folder / "curves.json", training)
    return {name: head.eval() for name, head in heads.items()}, updates


def boundary_edges(labels):
    known = labels != 255
    horizontal = (labels[..., 1:] != labels[..., :-1]) & known[..., 1:] & known[..., :-1]
    vertical = (labels[..., 1:, :] != labels[..., :-1, :]) & known[..., 1:, :] & known[..., :-1, :]
    return horizontal, vertical


def gradient_error(error, masks):
    numer = 0
    count = 0
    for delta, active in zip(edges(error), masks):
        numer = numer + (delta.abs().mean(1) * active).flatten(1).sum(1)
        count = count + active.flatten(1).sum(1)
    return numer / count.clamp_min(1), count


@torch.inference_mode()
def evaluate(core, heads, data, fold, seed):
    measured = []
    conditions = {}
    max_zero_sum_error = 0.0
    ids = fold["indices"]["validation"]
    truth_count = torch.bincount(data["mask"][ids].long().flatten(), minlength=256)[:CLASSES]
    for name in ALL_VIEWS:
        cm = torch.zeros(CLASSES, CLASSES, dtype=torch.int64)
        rows = []
        for start in range(0, len(ids), 8):
            index = ids[start:start + 8]
            clear = data["raw"][index].cuda().float() / 255
            masks = data["mask"][index].cuda().long()
            if name == "aligned_dense":
                depth = data["proxy_distance"][index].cuda().float()
                image, _ = fixed_view(clear, depth, name)
            else:
                image = (validation_view(clear, name) * 255).round() / 255
            anchor, logits, guides = common_inputs(core, image, masks)
            cm += confusion_matrix(logits.argmax(1), masks, CLASSES)
            band = boundary_band(masks)
            edge_mask = boundary_edges(masks)
            has_edge = band.flatten(1).any(1).tolist()
            outputs = {"anchor": anchor}
            for method, head in heads.items():
                rgb, _, detail = head(image, anchor, guides[method])
                if method != "plain":
                    error = detail.mean((-2, -1)).abs().max().item()
                    if error > 2e-7:
                        raise ValueError("Detail step changed image mean")
                    max_zero_sum_error = max(error, max_zero_sum_error)
                outputs[method] = rgb.clamp(0, 1)
            for method, rgb in outputs.items():
                mse, band_mse = image_measures(rgb, clear, band)
                ge, count = gradient_error(rgb - clear, edge_mask)
                whole, near, gradient = psnr(mse).tolist(), psnr(band_mse).tolist(), ge.tolist()
                enough = (count > 0).tolist()
                for i, j in enumerate(index):
                    rows.append(dict(video=fold["held_video"], seed=seed, condition=name, method=method,
                                     image=data["ids"][j], psnr=whole[i], boundary_psnr=near[i] if has_edge[i] else None,
                                     boundary_gradient_mae=gradient[i] if enough[i] else None))
        if not torch.equal(cm.sum(1), truth_count):
            raise ValueError("Semantic class counts changed")
        values = {}
        for method in ("anchor", *METHODS):
            selected = [row for row in rows if row["method"] == method]
            values[method] = {}
            for metric in METRICS:
                samples = [r[metric] for r in selected if r[metric] is not None]
                values[method][metric] = float(np.mean(samples))
        conditions[name] = dict(methods=values, segmentation=segmentation_metrics(cm), cm=cm.tolist())
        measured.extend(rows)
    return conditions, measured, max_zero_sum_error


def register(parent):
    dependencies = ("boundary_refinement/models/joint/boundary_flux.py", "boundary_refinement/training/train_boundary_flux.py", "boundary_refinement/data/regional_data.py",
                    "boundary_refinement/data/depth_correlated_fog.py", "boundary_refinement/inference/infer_regional_guidance.py")
    sources = [SOURCE_ROOT / file for file in dependencies]
    sources += [DATA, V18 / "completed.json", V20 / "depth_proxy.json"]
    value = dict(
        goal="Determine whether stronger class-aware local detail transfers provide additional information to RGB restoration.",
        scope="Three CamVid development videos held out in turn, two seeds, six inherited RGB and semantic cores. Same 256x192 grid and video memberships as V18. No official test predictions.",
        design="Multiscale refiner adds independent low-resolution RGB output and fine output. Four graph arms create signed pairwise RGB transfers, with equal and opposite contributions to each endpoint. The ordinary control uses the same RGB feature guide, dimensions, seed, and output layers but predicts unrestricted detail.",
        oracle="Ground-truth mask supplies same-class edges to the diagnostic refiner during training and validation. Void is its own component. This estimates the potential of true edges and is unavailable in deployment.",
        training="CUDA BF16 refinement with fixed FP32 V18 cores. Same 8 epochs and source scenes in all arms. Random camera/uniform/spatial/depth-proxy fog views. Common RGB/structure loss, including same boundary band weight.",
        conditions=list(ALL_VIEWS), methods=list(METHODS), batch=CORE_BATCH, epochs=EPOCHS,
        base_checkpoints=[{k: row[k] for k in ("held_video", "seed", "checkpoint", "checkpoint_SHA256")}
                          for row in parent["runs"]],
        screen=dict(conditions=["spatial", "spatial_alt", "aligned_dense"], matched_controls=["rgb", "shifted", "plain"],
                    boundary_gain_dB=0.10, minimum_full_gain_dB=0.0, averaging="equal held-out video, seed, fog condition"),
        files_SHA256={str(path): digest(path) for path in sources},
    )
    fingerprint = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = ROOT / "plan.json"
    if path.exists():
        prior = read(path)
        if prior["plan_SHA256"] != fingerprint:
            raise ValueError("Boundary study changed after registration")
        return prior
    value.update(plan_SHA256=fingerprint, created_utc=now())
    atomic_write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    parent = read(V18 / "completed.json")
    if not parent["all_registered_runs_complete"]:
        raise ValueError("Previous video-grouped cores are incomplete")
    source, partitions = prepare_grouped_data()
    load_proxy(source)
    folds = {f["held_video"]: f for f in partitions}
    if args.smoke:
        mathematical = probe_math()
        seed = parent["runs"][0]["seed"]
        fold = partitions[0]
        core, _, _ = load_checkpoint(parent["runs"][0]["checkpoint"])
        core.requires_grad_(False)
        trained = {name: BoundaryRefiner(seed, name).cuda() for name in METHODS}
        optimizers = {key: torch.optim.AdamW(head.parameters(), lr=4e-4) for key, head in trained.items()}
        batch = make_batch(source, fold["indices"]["core"][:CORE_BATCH], torch.Generator().manual_seed(21))
        values = update(core, trained, optimizers, batch)
        info = dict(all_passed=True, region_transfer_sum_error=mathematical, discarded_updates=len(METHODS),
                    losses=values, max_memory_GiB=torch.cuda.max_memory_allocated() / 2**30, created_utc=now())
        atomic_write(ROOT / "smoke.json", info)
        print(info, flush=True)
        return
    if not read(ROOT / "smoke.json")["all_passed"]:
        raise ValueError("CUDA boundary-refinement preflight failed")
    plan = register(parent)
    finished = []
    for entry in plan["base_checkpoints"]:
        video, seed = entry["held_video"], entry["seed"]
        base = Path(entry["checkpoint"])
        if digest(base) != entry["checkpoint_SHA256"]:
            raise ValueError("Frozen core checkpoint changed")
        folder = ROOT / f"video-{video}" / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        record = folder / "result.json"
        path = folder / "refiners.pth"
        table_path = folder / "by_image.csv"
        if record.exists():
            row = read(record)
            if (row["plan_SHA256"] != plan["plan_SHA256"] or digest(path) != row["checkpoint_SHA256"]
                    or digest(table_path) != row["by_image_SHA256"]):
                raise ValueError("A prior result was modified")
        else:
            set_seed(seed + 21_300_000)
            core, _, _ = load_checkpoint(base)
            core.requires_grad_(False)
            heads, count = fit_refiners(core, source, folds[video], seed, folder)
            scores, detail, residual_error = evaluate(core, heads, source, folds[video], seed)
            torch_save(path, dict(format="boundary-flux-v21", video=video, seed=seed, base=str(base),
                                  base_SHA256=entry["checkpoint_SHA256"],
                                  refiners={method: module.state_dict() for method, module in heads.items()}))
            write_csv(table_path, detail)
            row = dict(video=video, seed=seed, scores=scores, steps_per_head=count, original_row_count=len(detail),
                       zero_mean_residual_error=residual_error, checkpoint=str(path), checkpoint_SHA256=digest(path),
                       by_image=str(table_path), by_image_SHA256=digest(table_path), plan_SHA256=plan["plan_SHA256"],
                       params_per_head=sum(p.numel() for p in heads["rgb"].parameters()), finished_utc=now())
            atomic_write(record, row)
            del core, heads
            gc.collect()
            torch.cuda.empty_cache()
        finished.append(row)
        scores = row["scores"]["spatial"]["methods"]
        print(dict(video=video, seed=seed, completed_groups=len(finished),
                   spatial={method: [round(v["psnr"], 5), round(v["boundary_psnr"], 5)] for method, v in scores.items()}),
              flush=True)
    record = dict(all_registered_runs_complete=True, runs=finished, plan_SHA256=plan["plan_SHA256"],
                  refiner_fits=len(finished)*len(METHODS),
                  effective_updates=sum(row["steps_per_head"] for row in finished)*len(METHODS),
                  new_test_predictions=0, finished_utc=now())
    atomic_write(ROOT / "completed.json", record)
    print({key: value for key, value in record.items() if key != "runs"}, flush=True)


if __name__ == "__main__":
    main()

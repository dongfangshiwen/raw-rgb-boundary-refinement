"""Run the SG-AOD author's restoration architecture with CamVid inputs."""
import argparse
import ast
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed, write_csv
from scripts.depth_correlated_fog_v20 import fixed_view, load_proxy
from scripts.evaluate_regional_guidance_v18 import image_measures, psnr
from scripts.infer_regional_guidance_v18 import load_checkpoint
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.regional_data_v18 import boundary_band, prepare_grouped_data, validation_view
from scripts.regional_guidance_v18 import descriptor_rgb
from scripts.train_boundary_flux_v21 import (
    ALL_VIEWS, EPOCHS, METRICS, ROOT as V21, boundary_edges, gradient_error, loss_for, make_batch,
)
from scripts.train_regional_guidance_v18 import (
    CORE_BATCH, ROOT as V18, atomic_write, configure_cuda, torch_save,
)


AUTHOR = REPO.parent / "benchmark_data/sgnet-author-v21"
SOURCE = AUTHOR / "code/AOD-Net/semantic_AOD/net.py"
ROOT = V21 / "published-sg-aod"
METHODS = ("sg_rgb", "sg_semantic")
SEED = 42


def author_factory():
    """Extract the author's four classes from the pinned file without running a script."""
    registration = read(AUTHOR / "registration.json")
    expected = next(row["SHA256"] for row in registration["files"] if Path(row["path"]) == SOURCE)
    if registration["revision"] != "576b7104d4a861ff9642e46728a33da75c85af4a" or digest(SOURCE) != expected:
        raise ValueError("Unexpected SG-AOD source version")
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = {"ResLayer", "CALayer", "PALayer", "multiseg_attention_dehaze_net"}
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in classes]
    if {node.name for node in nodes} != classes:
        raise ValueError("Incomplete author restoration architecture")
    values = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), values)
    return values["multiseg_attention_dehaze_net"]


def construct(factory, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 21_500_000)
        model = factory(num_classes=11)
    # Both variants start at the same haze image, matching the identity start of
    # the refinement heads with their corresponding input images.
    torch.nn.init.zeros_(model.conv4.weight)
    torch.nn.init.zeros_(model.conv4.bias)
    return model.cuda()


@torch.no_grad()
def hints(core, image):
    from scripts.dual_coordinate_v17 import canonical_rgb
    _, semantic = core.semantic(canonical_rgb(image, core.mean, "channel"))
    mean = F.avg_pool2d(image, 8)
    raw = descriptor_rgb(image, F.interpolate(mean, image.shape[-2:], mode="bilinear", align_corners=False))
    center = raw.mean((-2, -1), keepdim=True)
    scale = raw.std((-2, -1), keepdim=True, unbiased=False).clamp_min(0.01)
    return {"sg_rgb": ((raw - center) / scale).clamp(-4, 4), "sg_semantic": semantic}


def update(core, models, optimizers, batch):
    image, clean, target = batch
    guidance = hints(core, image)
    valid = torch.ones_like(target, dtype=torch.bool)
    band = boundary_band(target)
    scores = {}
    for name, model in models.items():
        optimizers[name].zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            restored = model(image, guidance[name]).float()
            objective = loss_for(restored, clean, band, valid)
        if not torch.isfinite(objective):
            raise ValueError("Nonfinite SG-AOD reconstruction objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizers[name].step()
        scores[name] = float(objective.detach())
    return scores


@torch.inference_mode()
def evaluate(core, models, source, fold):
    results, averages = [], {}
    indices = fold["indices"]["validation"]
    for view in ALL_VIEWS:
        records = []
        for start in range(0, len(indices), 4):
            batch = indices[start:start + 4]
            clean = source["raw"][batch].cuda().float() / 255
            if view == "aligned_dense":
                image, _ = fixed_view(clean, source["proxy_distance"][batch].cuda().float(), view)
            else:
                image = (255 * validation_view(clean, view)).round() / 255
            target = source["mask"][batch].cuda().long()
            band = boundary_band(target)
            guide = hints(core, image)
            for method, model in models.items():
                rgb = model(image, guide[method]).float().clamp(0, 1)
                p, q = image_measures(rgb, clean, band)
                grad, n = gradient_error(rgb - clean, boundary_edges(target))
                values = zip(psnr(p).tolist(), psnr(q).tolist(), grad.tolist(), (n > 0).tolist())
                for i, (whole, edge, error, valid) in enumerate(values):
                    records.append(dict(video=fold["held_video"], seed=SEED, condition=view, method=method,
                                        image=source["ids"][batch[i]], psnr=whole,
                                        boundary_psnr=edge if valid else None,
                                        boundary_gradient_mae=error if valid else None))
        averages[view] = {
            method: {key: float(np.mean([r[key] for r in records if r["method"] == method and r[key] is not None]))
                     for key in METRICS}
            for method in METHODS
        }
        results.extend(records)
    return averages, results


def register(parent):
    registration = read(AUTHOR / "registration.json")
    value = dict(
        source=registration["source"], revision=registration["revision"], architecture="multiseg_attention_dehaze_net",
        source_SHA256=digest(SOURCE), study_code_SHA256=digest(__file__),
        publication="https://openaccess.thecvf.com/content/ACCV2022/html/Hong_SG-Net_Semantic_Guided_Network_for_Image_Dehazing_ACCV_2022_paper.html",
        scope="Author SG-AOD RGB restoration architecture, changed to 11 logits from the same frozen video-specific SegFormer. Matched RGB descriptor control. Single seed across the three development videos.",
        adaptation="Same 64-channel architecture, feature fusion and attention. SegFormer replaces RefineNet; source-only RGB/structure objective replaces author's semantic-feature loss and datasets. Direct fog input and identity initialization. No author performance reproduction.",
        conditions=list(ALL_VIEWS), seeds=[SEED], epochs=EPOCHS, batch=CORE_BATCH,
        main_plan_SHA256=read(V21 / "plan.json")["plan_SHA256"],
        main_complete_SHA256=digest(V21 / "completed.json"),
        base=[{k: row[k] for k in ("held_video", "seed", "checkpoint", "checkpoint_SHA256")}
              for row in parent["runs"] if row["seed"] == SEED],
    )
    sha = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = ROOT / "plan.json"
    if path.exists():
        found = read(path)
        if found["plan_SHA256"] != sha:
            raise ValueError("Published comparison plan changed")
        return found
    value.update(plan_SHA256=sha, created_utc=now())
    atomic_write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    factory = author_factory()
    parent = read(V18 / "completed.json")
    data, splits = prepare_grouped_data()
    load_proxy(data)
    by_video = {fold["held_video"]: fold for fold in splits}
    if args.smoke:
        base = parent["runs"][0]
        core, _, _ = load_checkpoint(base["checkpoint"])
        core.requires_grad_(False)
        models = {name: construct(factory, SEED) for name in METHODS}
        options = {key: torch.optim.AdamW(model.parameters(), lr=4e-4) for key, model in models.items()}
        batch = make_batch(data, by_video[base["held_video"]]["indices"]["core"][:CORE_BATCH],
                           torch.Generator().manual_seed(21))
        values = update(core, models, options, batch)
        artifact = dict(all_passed=True, scope="BF16 forward/backward with real RGB and loaded semantic logits.",
                        discarded_updates=len(METHODS), losses=values,
                        peak_GiB=torch.cuda.max_memory_allocated() / 2**30, updated_utc=now())
        atomic_write(ROOT / "smoke.json", artifact)
        print(artifact, flush=True)
        return
    if not read(ROOT / "smoke.json")["all_passed"]:
        raise ValueError("Author model preflight is missing")
    plan = register(parent)
    outcomes = []
    for row in plan["base"]:
        video = row["held_video"]
        if digest(row["checkpoint"]) != row["checkpoint_SHA256"]:
            raise ValueError("SG-AOD base segmenter changed")
        directory = ROOT / video
        directory.mkdir(parents=True, exist_ok=True)
        record, checkpoint, table = (directory / name for name in ("result.json", "trained.pth", "by_image.csv"))
        if record.exists():
            outcome = read(record)
            if (outcome["plan_SHA256"] != plan["plan_SHA256"] or digest(checkpoint) != outcome["checkpoint_SHA256"]
                    or digest(table) != outcome["by_image_SHA256"]):
                raise ValueError("A prior SG-AOD result changed")
        else:
            set_seed(21_700_000 + SEED)
            core, _, _ = load_checkpoint(row["checkpoint"])
            core.requires_grad_(False)
            models = {name: construct(factory, SEED) for name in METHODS}
            optimizers = {name: torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
                          for name, model in models.items()}
            schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=5e-5)
                          for name, opt in optimizers.items()}
            steps = 0
            started = time.perf_counter()
            fold = by_video[video]
            indices = torch.tensor(fold["indices"]["core"])
            generator = torch.Generator().manual_seed(SEED + 21_200_000)
            curves = []
            for epoch in range(1, EPOCHS + 1):
                [model.train() for model in models.values()]
                losses = []
                for ids in indices[torch.randperm(len(indices), generator=generator)].split(CORE_BATCH):
                    losses.append(update(core, models, optimizers, make_batch(data, ids, generator)))
                    steps += 1
                for scheduler in schedulers.values():
                    scheduler.step()
                info = dict(video=video, epoch=epoch, steps_per_method=steps,
                            losses={m: float(np.mean([obj[m] for obj in losses])) for m in METHODS},
                            elapsed=round(time.perf_counter()-started, 2), updated_utc=now())
                curves.append(info)
                atomic_write(ROOT / "progress.json", info)
                print(info, flush=True)
            for model in models.values():
                model.eval()
            scores, results = evaluate(core, models, data, fold)
            write_csv(table, results)
            atomic_write(directory / "curves.json", curves)
            torch_save(checkpoint, dict(format="sg-aod-camvid-v21", video=video, seed=SEED, base=row["checkpoint"],
                                        base_SHA256=row["checkpoint_SHA256"],
                                        models={m: x.state_dict() for m, x in models.items()}))
            outcome = dict(video=video, seed=SEED, scores=scores, steps=steps, checkpoint=str(checkpoint),
                           checkpoint_SHA256=digest(checkpoint), by_image=str(table), by_image_SHA256=digest(table),
                           params_per_method=sum(parameter.numel() for parameter in models["sg_rgb"].parameters()),
                           plan_SHA256=plan["plan_SHA256"], finished_utc=now())
            atomic_write(record, outcome)
            del core, models
            gc.collect()
            torch.cuda.empty_cache()
        outcomes.append(outcome)
        print(dict(video=video, completed=len(outcomes), spatial=outcome["scores"]["spatial"]), flush=True)
    result = dict(all_registered_runs_complete=True, checkpoints=len(outcomes), fits=len(outcomes)*len(METHODS),
                  effective_updates=sum(r["steps"] for r in outcomes)*len(METHODS), runs=outcomes,
                  new_official_test_predictions=0, finished_utc=now())
    atomic_write(ROOT / "completed.json", result)
    print({k: v for k, v in result.items() if k != "runs"})


if __name__ == "__main__":
    main()

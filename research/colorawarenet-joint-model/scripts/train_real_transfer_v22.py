"""Train matched real-haze transfer recipes and an author-code restoration reference."""
import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.appearance_refiner_v22 import AppearanceRefiner, RECIPES, appearance_loss, restoration_anchor
from scripts.boundary_flux_v21 import raw_affinities
from scripts.depth_correlated_fog_v20 import ROOT as V20, load_proxy
from scripts.infer_boundary_flux_v21 import load_refiners
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.published_runtime_v12 import SOURCE as AUTHOR_SOURCE, dehazeformer_t
from scripts.real_haze_data_v22 import ROOT, load_data, mask_for
from scripts.regional_data_v18 import prepare_grouped_data
from scripts.train_boundary_flux_v21 import ROOT as V21, make_batch
from scripts.train_regional_guidance_v18 import CORE_BATCH, atomic_write, configure_cuda, torch_save


EPOCHS = 8
AUTHOR = "dehazeformer_t_paired"
TRAINING_DOMAINS = ("synthetic", "donor_mean", "donor_spatial", "paired")


def prepare_real():
    packet = load_data("training")
    rows = packet["training"]
    if [row["id"] for row in rows] != [f"{i:02d}" for i in range(1, 46)]:
        raise ValueError("Only the 45 registered NH-HAZE pairs may enter training")
    return dict(hazy=torch.stack([r["hazy"] for r in rows]).cuda().float()/255,
                clean=torch.stack([r["clean"] for r in rows]).cuda().float()/255,
                valid=torch.cat([mask_for(r["roi"]) for r in rows]).cuda(), fields=packet["fields"].cuda())


def flipped(value, use_flip):
    return torch.where(use_flip, value.flip(-1), value)


def inputs(source, scene, real, synthesis_rng, donor_rng):
    hazy, clean, _ = make_batch(source, scene, synthesis_rng)
    n = len(scene)
    split = 3*n
    clear = clean[split:]
    valid = torch.ones_like(hazy[:, 0], dtype=torch.bool)
    ids = torch.randint(len(real["fields"]), (n,), generator=donor_rng).cuda()
    flip = (torch.rand(n, generator=donor_rng) > .5).cuda()[:, None, None, None]
    spatial = F.interpolate(flipped(real["fields"][ids], flip), clear.shape[-2:], mode="bilinear",
                            align_corners=False)
    homogeneous = spatial.mean((-2, -1), keepdim=True)
    diffuse = lambda field: ((clear*field[:, :1] + field[:, 1:]).clamp(0, 1)*255).round()/255
    tails = dict(synthetic=(hazy[split:], clear, valid[split:]),
                 donor_mean=(diffuse(homogeneous), clear, valid[split:]),
                 donor_spatial=(diffuse(spatial), clear, valid[split:]),
                 paired=(flipped(real["hazy"][ids], flip), flipped(real["clean"][ids], flip),
                         flipped(real["valid"][ids], flip[:, 0])))
    return (hazy[:split], clean[:split], valid[:split]), tails


def concatenate(common, tail):
    return tuple(torch.cat((head, end)) for head, end in zip(common, tail))


def author_rgb(model, image):
    return .5*(model(2*image-1).float() + 1)


def make_models(seed, author):
    if author:
        return {AUTHOR: dehazeformer_t().cuda()}
    return {name: AppearanceRefiner(seed, choice).cuda() for name, (_, choice) in RECIPES.items()}


def make_optimizers(models):
    return {name: torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.01)
            for name, model in models.items()}


def update(core, anchor_head, models, optimizers, common, tails, author):
    if not author:
        with torch.no_grad():
            combined = torch.cat((common[0], *(tails[domain][0] for domain in TRAINING_DOMAINS)))
            anchor = restoration_anchor(core, anchor_head, combined)
            k, n = len(common[0]), len(tails["synthetic"][0])
            batches = {domain: concatenate(common, tail) for domain, tail in tails.items()}
            anchors = {domain: torch.cat((anchor[:k], extra)) for domain, extra in
                       zip(TRAINING_DOMAINS, anchor[k:].split(n))}
            weights = {domain: raw_affinities(batch[0]) for domain, batch in batches.items()}
    else:
        batches = {"paired": concatenate(common, tails["paired"])}
    statistics = {}
    for name, model in models.items():
        domain = "paired" if author else RECIPES[name][0]
        image, reference, valid = batches[domain]
        optimizer = optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = author_rgb(model, image) if author else model(image, anchors[domain], weights[domain])
            objective = appearance_loss(output, reference, valid)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        statistics[name] = objective.detach()
    return torch.stack(tuple(statistics.values())).tolist()


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def fit(core, anchor, source, real, fold, seed, author):
    models = make_models(seed, author)
    optimizers = make_optimizers(models)
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=5e-5)
                 for opt in optimizers.values()]
    scene = torch.tensor(fold["indices"]["core"])
    synthesis_rng = torch.Generator().manual_seed(seed + 22_200_000)
    donor_rng = torch.Generator().manual_seed(seed + 22_300_000)
    curves, updates = [], 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        statistics = []
        for ids in scene[torch.randperm(len(scene), generator=synthesis_rng)].split(CORE_BATCH):
            common, tails = inputs(source, ids, real, synthesis_rng, donor_rng)
            statistics.append(update(core, anchor, models, optimizers, common, tails, author))
        for schedule in schedules:
            schedule.step()
        updates += len(statistics)
        row = dict(video=fold["held_video"], seed=seed, epoch=epoch, updates_per_model=updates,
                   loss=dict(zip(models, np.mean(statistics, axis=0).tolist())),
                   max_GiB=round(torch.cuda.max_memory_allocated()/2**30, 3),
                   seconds=round(time.perf_counter()-started, 2), updated_utc=now())
        curves.append(row)
        atomic_write(ROOT / ("author_progress.json" if author else "progress.json"), row)
        print(row, flush=True)
    if updates != EPOCHS*math.ceil(len(scene)/CORE_BATCH):
        raise ValueError("Training budget changed")
    return models, curves, updates


def register():
    parent = read(V21 / "completed.json")
    if not parent["all_registered_runs_complete"] or len(parent["runs"]) != 6:
        raise ValueError("Six verified V21 checkpoints are required")
    dependencies = ("train_real_transfer_v22.py", "real_haze_data_v22.py", "appearance_refiner_v22.py",
                    "train_boundary_flux_v21.py", "boundary_flux_v21.py", "published_runtime_v12.py",
                    "regional_data_v18.py", "depth_correlated_fog_v20.py", "infer_boundary_flux_v21.py")
    paths = [REPO / "scripts" / name for name in dependencies]
    paths.extend((ROOT / "data_manifest.json", V20 / "depth_proxy.json",
                  AUTHOR_SOURCE / "author-dehazeformer/dehazeformer.py"))
    bases = [{name: row[name] for name in ("video", "seed", "checkpoint", "checkpoint_SHA256")}
             for row in parent["runs"]]
    value = dict(epochs=EPOCHS, scenes_per_step=CORE_BATCH, methods={name: list(recipe) for name, recipe in RECIPES.items()},
                 author=dict(method=AUTHOR, seeds=[42], source="https://github.com/IDKiro/DehazeFormer",
                             scope="Author DehazeFormer-T initialized from scratch using the same paired exposure, size, loss and updates as paired_free. Inherited weights of our trunk differ."),
                 fixed_base=bases, source_SHA256={str(path): digest(path) for path in paths},
                 held_out="NH-HAZE 46–55 after final epoch, Dense-Haze55 and O-HAZE45 RGB; grouped CamVid development. No official CamVid test use.",
                 schedule="CUDA BF16 trained weights, FP32 frozen V21 RGB anchor. Shared camera, synthetic uniform and spatial data in 3/4 of each batch. Remaining 1/4 follows each recipe. Same underlying donor draws for all methods. Final epoch, no external selection.",
                 contrasts=dict(paired_vs_synthetic=["paired_free", "synthetic_free"],
                                spatial_vs_mean=["affine_spatial", "affine_mean"],
                                observed_paired=["paired_detail", "paired_free"],
                                observed_synthetic=["synthetic_detail", "synthetic_free"]))
    path = ROOT / "plan.json"
    if path.exists():
        prior = read(path)
        if prior != value:
            raise ValueError("Protocol or checked file changed since registration")
    else:
        atomic_write(path, value)
    return value


def open_anchor(entry):
    path = Path(entry["checkpoint"])
    if digest(path) != entry["checkpoint_SHA256"]:
        raise ValueError("V21 checkpoint changed")
    core, modules, info = load_refiners(path, ("rgb",))
    if (info["video"], info["seed"]) != (entry["video"], entry["seed"]):
        raise ValueError("Anchor belongs to another fold")
    return core.requires_grad_(False), modules["rgb"].requires_grad_(False)


def smoke(source, real, fold, entry, author):
    models = make_models(entry["seed"], author)
    core, anchor = (None, None) if author else open_anchor(entry)
    ids = fold["indices"]["core"][:CORE_BATCH]
    common, tails = inputs(source, ids, real, torch.Generator().manual_seed(2026),
                           torch.Generator().manual_seed(2027))
    units = {name: parameters(model) for name, model in models.items()}
    with torch.inference_mode():
        if not author:
            x = common[0]
            value = restoration_anchor(core, anchor, x)
            plain, _, _, _, logits, _ = core(x)
            expected, _, _ = anchor(x, plain, raw_affinities(x))
            if not torch.equal(value, expected.clamp(0, 1)) or logits.shape[1] != 11:
                raise ValueError("Fast RGB anchor disagrees with deployed dual output")
            orig = {name: model(x, value, raw_affinities(x)) for name, model in models.items()}
            if len(set(units.values())) != 1 or any(not torch.equal(value, output) for output in orig.values()):
                raise ValueError("Architecture controls must begin from the same anchor with equal capacity")
        if torch.equal(tails["donor_mean"][0], tails["donor_spatial"][0]):
            raise ValueError("Spatial donor contrast collapsed")
        if not torch.equal(tails["donor_mean"][1], tails["donor_spatial"][1]):
            raise ValueError("Donor contrast changed the target image")
    objectives = update(core, anchor, models, make_optimizers(models), common, tails, author)
    if not np.isfinite(objectives).all():
        raise ValueError("CUDA smoke update contained nonfinite values")
    row = dict(passed=True, discarded_updates=len(models), parameters=units, initial_loss=dict(zip(models, objectives)),
               cuda=torch.version.cuda, torch=torch.__version__, gpu=torch.cuda.get_device_name(), updated_utc=now())
    atomic_write(ROOT / ("author_smoke.json" if author else "smoke.json"), row)
    print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--author-only", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    plan = register()
    plan_hash = digest(ROOT / "plan.json")
    source, folds = prepare_grouped_data()
    load_proxy(source)
    by_video = {f["held_video"]: f for f in folds}
    real = prepare_real()
    entries = [e for e in plan["fixed_base"] if not args.author_only or e["seed"] in plan["author"]["seeds"]]
    if args.smoke:
        smoke(source, real, by_video[entries[0]["video"]], entries[0], args.author_only)
        return
    if not read(ROOT / ("author_smoke.json" if args.author_only else "smoke.json"))["passed"]:
        raise ValueError("CUDA preflight should pass before training")
    runs = []
    for entry in entries:
        video, seed = entry["video"], entry["seed"]
        folder = ROOT / ("author" if args.author_only else "refiners") / f"video-{video}" / f"seed-{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        path, result = folder / "weights.pth", folder / "result.json"
        if result.exists():
            row = read(result)
            if row["plan_SHA256"] != plan_hash or row["checkpoint_SHA256"] != digest(path):
                raise ValueError("Completed V22 training artifact changed")
        else:
            set_seed(seed + 22_400_000)
            core, anchor = (None, None) if args.author_only else open_anchor(entry)
            models, curves, updates = fit(core, anchor, source, real, by_video[video], seed, args.author_only)
            units = {name: parameters(module) for name, module in models.items()}
            torch_save(path, dict(format="real-transfer-v22", video=video, seed=seed,
                                  base=entry, plan_SHA256=plan_hash,
                                  models={name: module.state_dict() for name, module in models.items()}))
            atomic_write(folder / "curves.json", curves)
            row = dict(video=video, seed=seed, updates_per_model=updates, models=units, checkpoint=str(path),
                       checkpoint_SHA256=digest(path), plan_SHA256=plan_hash, finished_utc=now())
            atomic_write(result, row)
            del models, core, anchor
            gc.collect()
            torch.cuda.empty_cache()
        runs.append(row)
        print(dict(video=video, seed=seed, completed_groups=len(runs), author=args.author_only), flush=True)
    summary = dict(all_complete=True, runs=runs, plan_SHA256=plan_hash, finished_utc=now(),
                   fits=sum(len(r["models"]) for r in runs), updates=sum(r["updates_per_model"]*len(r["models"]) for r in runs))
    atomic_write(ROOT / ("author_completed.json" if args.author_only else "completed.json"), summary)
    print({key: value for key, value in summary.items() if key != "runs"}, flush=True)


if __name__ == "__main__":
    main()

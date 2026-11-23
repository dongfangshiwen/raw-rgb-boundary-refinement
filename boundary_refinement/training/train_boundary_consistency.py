"""Compare RGB adaptation depth, then matched train-only boundary objectives."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
import gc
import math
import time

import numpy as np
import torch

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.appearance_refiner import appearance_loss
from boundary_refinement.models.joint.boundary_consistency import (
    LOSSES, SCOPES, adjacent_margins, copy_restorer, edge_count, from_v22,
    load_adaptation, semantic_auxiliary, teacher_targets, trainable_weights,
)
from boundary_refinement.models.joint.boundary_flux import raw_affinities
from boundary_refinement.data.depth_correlated_fog import ROOT as V20, load_proxy
from boundary_refinement.evaluation.paired.evaluate_real_transfer import load_transformer
from boundary_refinement.inference.infer_real_transfer import restore as restore_v22
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.baselines.published_runtime import SOURCE as AUTHORS
from boundary_refinement.data.real_haze_data import ROOT as V22
from boundary_refinement.data.regional_data import prepare_grouped_data
from boundary_refinement.training.train_boundary_flux import make_batch
from boundary_refinement.training.train_real_transfer import AUTHOR, author_rgb, flipped, prepare_real
from boundary_refinement.training.train_regional_guidance import CORE_BATCH, atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/boundary-consistency-v23"
EPOCHS = 6
STAGES = ("adaptation", "consistency")
AUX_WEIGHT = .20


def register():
    own = read(V22 / "completed.json")
    authors = read(V22 / "author_completed.json")
    if not own["all_complete"] or not authors["all_complete"] or len(own["runs"]) != 6 or len(authors["runs"]) != 3:
        raise ValueError("Need six V22 RGB and three matched DehazeFormer-T checkpoints")
    paths = [SOURCE_ROOT / name for name in (
        "boundary_refinement/models/joint/boundary_consistency.py", "boundary_refinement/training/train_boundary_consistency.py", "boundary_refinement/models/joint/boundary_flux.py",
        "boundary_refinement/training/train_boundary_flux.py", "boundary_refinement/training/train_real_transfer.py", "boundary_refinement/models/restoration/appearance_refiner.py", "boundary_refinement/data/camvid_data.py",
        "boundary_refinement/inference/infer_real_transfer.py", "boundary_refinement/data/regional_data.py", "boundary_refinement/data/depth_correlated_fog.py",
        "boundary_refinement/evaluation/paired/evaluate_real_transfer.py", "boundary_refinement/models/joint/dual_coordinate.py",
    )]
    paths.extend((V22 / "completed.json", V22 / "author_completed.json", V22 / "data_manifest.json",
                  V20 / "depth_proxy.json", AUTHORS / "author-dehazeformer/dehazeformer.py"))
    plan = dict(
        starting_points=own["runs"], author=authors["runs"], source_SHA256={str(p): digest(p) for p in paths},
        epochs_per_stage=EPOCHS, stages=list(STAGES), scenes_per_batch=CORE_BATCH,
        adaptation=dict(head_only="V22 correction head", decoder_tail="head, V21 RGB refiner, NAF last upsampler, last decoder and output, global affine output",
                        full_rgb="head, V21 refiner, all of NAF and GlobalAffine"),
        consistency=dict(start="Use full_rgb final epoch for all four variants, fixed before evaluation",
                         photo_only="Additional RGB-only training",
                         global_semantic="KL to clear frozen CamVid11 teacher, ignore void, temperature 2",
                         rgb_boundary="Same adjacent-class locations as class_boundary. Match directed RGB contrast to the clear reference",
                         class_boundary="Match both sides' signed two-class logit margins to a correct-sign clean teacher in annotated adjacent classes, using a sigmoid with temperature 2",
                         auxiliary_scale="Fixed per fold/seed and loss using %.2f of RGB output-gradient norm at start on four train-only batches" % AUX_WEIGHT),
        data="All runs replay the same 3/4 CamVid camera, uniform and spatial views and 1/4 NH-HAZE 01–45 paired RGB. Additional supervision uses two CamVid fog views and labels. Real RGB pairs provide no semantic labels. CamVid official test excluded.",
        optimizer="BF16 CUDA, AdamW, .01 decay and gradient clipping 1. Separate cosine starts each stage. V22 correction LR .0002 and other weights .0001; each ends at 20%. Segmenters always frozen.",
        author_scope="Real V22 author DehazeFormer-T code recovered at its V22 final weights, two equal photo-only stages for seed 42. Earlier weight histories differ.",
        endpoints="Fixed final epochs for every run, no target selection. CamVid held-video RGB and 11-class segmentation-through-RGB diagnostic. O-HAZE, NH holdout and Dense-Haze rechecks. I-HAZE as a new paired evaluation if the official download is verified. Prior real-road labels only for repeated diagnostic.",
    )
    path = ROOT / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Registered V23 training protocol or dependency changed")
    if not path.exists():
        atomic_write(path, plan)
    return plan, digest(path)


def training_batch(source, ids, real, scenes_rng, real_rng):
    hazy, clean, labels = make_batch(source, ids, scenes_rng)
    n = len(ids)
    end = 3*n
    donors = torch.randint(len(real["hazy"]), (n,), generator=real_rng).cuda()
    flip = (torch.rand(n, generator=real_rng) > .5).cuda()[:, None, None, None]
    return dict(hazy=torch.cat((hazy[:end], flipped(real["hazy"][donors], flip))),
                clean=torch.cat((clean[:end], flipped(real["clean"][donors], flip))),
                valid=torch.cat((torch.ones_like(labels[:end], dtype=torch.bool), flipped(real["valid"][donors], flip[:, 0]))),
                labels=labels[:n], aux=slice(n, end), n=n)


def random_generators(seed, stage):
    cell = 23_000_000 + 200_000 * STAGES.index(stage) + seed
    return torch.Generator().manual_seed(cell), torch.Generator().manual_seed(cell + 100_000)


def targets_for_batch(core, batch):
    return teacher_targets(core, batch["clean"][:batch["n"]], batch["labels"], repeats=2)


def restore_and_loss(model, batch, guides, author):
    output = author_rgb(model, batch["hazy"]) if author else model(batch["hazy"], guides)
    objective = appearance_loss(output, batch["clean"], batch["valid"])
    return output, objective


def calibration(restorer, core, source, indices, real, seed):
    train_rng, real_rng = random_generators(seed + 30_000, "consistency")
    statistics = {name: [] for name in LOSSES[1:]}
    counts = []
    cut = indices[torch.randperm(len(indices), generator=train_rng)[:4*CORE_BATCH]]
    for ids in cut.split(CORE_BATCH):
        batch = training_batch(source, ids, real, train_rng, real_rng)
        target = targets_for_batch(core, batch)
        counts.append(edge_count(target))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = restorer(batch["hazy"], raw_affinities(batch["hazy"]))
        output = output.detach().requires_grad_(True)
        photo = appearance_loss(output, batch["clean"], batch["valid"])
        primary = torch.autograd.grad(photo, output)[0].norm().item()
        for name in statistics:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = semantic_auxiliary(name, core, output[batch["aux"]].clamp(0, 1), target)
            norm = torch.autograd.grad(loss, output)[0].norm().item()
            statistics[name].append(dict(photo=primary, auxiliary=norm))
    scales = {}
    for name, records in statistics.items():
        old = np.mean([r["photo"] for r in records])
        new = np.mean([r["auxiliary"] for r in records])
        if not np.isfinite([old, new]).all() or not 0 < new:
            raise ValueError("Boundary auxiliary lacked a usable training signal")
        scales[name] = float(AUX_WEIGHT*old/new)
    return scales, dict(gradient_norms=statistics, valid_pair_counts=counts)


def optimizer_for(model):
    groups = [dict(params=[p for name, p in model.named_parameters() if p.requires_grad and name.startswith("correction.")],
                   lr=2e-4),
              dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("correction.")],
                   lr=1e-4)]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=.01)


def update(models, optimizers, batch, core, coefficients, author):
    guides = None if author else raw_affinities(batch["hazy"])
    target = targets_for_batch(core, batch) if coefficients else None
    photo_losses, auxiliary_losses = [], []
    for name, model in models.items():
        opt = optimizers[name]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, photo = restore_and_loss(model, batch, guides, author)
            weighted = (coefficients[name] * semantic_auxiliary(name, core, out[batch["aux"]].clamp(0, 1), target)
                        if name in coefficients else photo.new_zeros(()))
            loss = photo + weighted
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        opt.step()
        photo_losses.append(photo.detach())
        auxiliary_losses.append(weighted.detach())
    numbers = torch.stack((*photo_losses, *auxiliary_losses)).tolist()
    if not np.isfinite(numbers).all():
        raise ValueError("Nonfinite learned loss")
    return numbers


def fit(models, core, source, fold, real, seed, stage, coefficients, author=False):
    optimizers = {name: optimizer_for(m) for name, m in models.items()}
    schedules = [torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: .2 + .8*(.5+.5*math.cos(math.pi*epoch/EPOCHS)))
                 for opt in optimizers.values()]
    selected = torch.tensor(fold["indices"]["core"])
    scenes_rng, donors_rng = random_generators(seed, stage)
    curves, updated = [], 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for ids in selected[torch.randperm(len(selected), generator=scenes_rng)].split(CORE_BATCH):
            batch = training_batch(source, ids, real, scenes_rng, donors_rng)
            losses.append(update(models, optimizers, batch, core, coefficients, author))
        updated += len(losses)
        averaged = np.mean(losses, axis=0).tolist()
        row = dict(stage=stage, author=author, video=fold["held_video"], seed=seed, epoch=epoch,
                   steps_per_model=updated, photo_loss=dict(zip(models, averaged[:len(models)])),
                   weighted_auxiliary=dict(zip(models, averaged[len(models):])),
                   peak_GiB=round(torch.cuda.max_memory_allocated()/2**30, 3),
                   seconds=round(time.perf_counter() - started, 2), updated_utc=now())
        for schedule in schedules:
            schedule.step()
        curves.append(row)
        atomic_write(ROOT / ("author_progress.json" if author else "progress.json"), row)
        print(row, flush=True)
    if updated != EPOCHS*math.ceil(len(selected)/CORE_BATCH):
        raise ValueError("Unmatched stage update counts")
    return curves, updated


def verified_start(entry):
    if digest(entry["checkpoint"]) != entry["checkpoint_SHA256"]:
        raise ValueError("V22 starting weights changed")
    return from_v22(entry["checkpoint"])


def directory(entry):
    return ROOT / f"video-{entry['video']}" / f"seed-{entry['seed']}"


def existing_record(folder, stage, plan_hash):
    path = folder / (stage + ".json")
    if not path.exists():
        return None
    prior = read(path)
    if prior["plan_SHA256"] != plan_hash or digest(prior["checkpoint"]) != prior["checkpoint_SHA256"]:
        raise ValueError("A completed stage or its registered source changed")
    return prior


def save_stage(folder, stage, entry, models, updates, curves, plan_hash, initializer, **details):
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / (stage + ".pth")
    state = {method: trainable_weights(model) for method, model in models.items()}
    torch_save(checkpoint, dict(stage=stage, video=entry["video"], seed=entry["seed"], starting_point=entry,
                                initialized_from=initializer, plan_SHA256=plan_hash, models=state))
    atomic_write(folder / (stage + "_curves.json"), curves)
    row = dict(stage=stage, video=entry["video"], seed=entry["seed"], checkpoint=str(checkpoint),
               checkpoint_SHA256=digest(checkpoint), initialized_from=initializer,
               steps_per_model=updates, trainable={name: sum(x.numel() for x in params.values()) for name, params in state.items()},
               plan_SHA256=plan_hash, finished_utc=now(), **details)
    atomic_write(folder / (stage + ".json"), row)
    return row


def paired_training(entry, plan_hash, source, fold, real):
    folder = directory(entry)
    adaptation = existing_record(folder, "adaptation", plan_hash)
    consistency = existing_record(folder, "consistency", plan_hash)
    if adaptation and consistency:
        return [adaptation, consistency]
    seed = entry["seed"]
    set_seed(23_700_000 + seed)
    core, start, _ = verified_start(entry)
    if adaptation is None:
        models = {name: copy_restorer(start, name) for name in SCOPES}
        curves, steps = fit(models, core, source, fold, real, seed, "adaptation", {})
        adaptation = save_stage(folder, "adaptation", entry, models, steps, curves, plan_hash, entry["checkpoint_SHA256"])
        del models
        gc.collect()
        torch.cuda.empty_cache()
    record = torch.load(adaptation["checkpoint"], map_location="cpu", weights_only=True)
    start = load_adaptation(start, record["models"]["full_rgb"], "full_rgb")
    weights, calibration_log = calibration(start, core, source, torch.tensor(fold["indices"]["core"]), real, seed)
    models = {name: copy_restorer(start, "full_rgb") for name in LOSSES}
    curves, steps = fit(models, core, source, fold, real, seed, "consistency", weights)
    consistency = save_stage(folder, "consistency", entry, models, steps, curves, plan_hash,
                             adaptation["checkpoint_SHA256"], train_only_scale=weights, calibration=calibration_log)
    return [adaptation, consistency]


def continue_author(entry, plan_hash, source, fold, real):
    folder = directory(entry) / "author"
    if digest(entry["checkpoint"]) != entry["checkpoint_SHA256"]:
        raise ValueError("Author input checkpoint changed")
    model = load_transformer(entry)
    results, initialization = [], entry["checkpoint_SHA256"]
    for stage in STAGES:
        item = existing_record(folder, stage, plan_hash)
        if item:
            state = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
            parameters = dict(model.named_parameters())
            if set(parameters) != set(state["models"][AUTHOR]):
                raise ValueError("Author parameter set changed")
            with torch.no_grad():
                for name, value in state["models"][AUTHOR].items():
                    parameters[name].copy_(value)
        else:
            models = {AUTHOR: model}
            curves, steps = fit(models, None, source, fold, real, entry["seed"], stage, {}, author=True)
            item = save_stage(folder, stage, entry, models, steps, curves, plan_hash, initialization)
        initialization = item["checkpoint_SHA256"]
        results.append(item)
    return results


def preflight(plan, data, fold, real):
    entry = plan["starting_points"][0]
    core, model, _ = verified_start(entry)
    author = load_transformer(plan["author"][0])
    ids = torch.tensor(fold["indices"]["core"])
    rng, donor_rng = random_generators(entry["seed"], "adaptation")
    batch = training_batch(data, ids[:CORE_BATCH], real, rng, donor_rng)
    with torch.inference_mode():
        image = batch["hazy"]
        value = model(image, raw_affinities(image)).clamp(0, 1)
        previous = restore_v22(core, model.anchor, {"paired_free": model.correction}, image)["paired_free"]
        logits = torch.tensor([[[[3., 0.]], [[0., 4.]]]], device="cuda")
        a, b = torch.zeros((1, 1, 1), dtype=torch.long, device="cuda"), torch.ones((1, 1, 1), dtype=torch.long, device="cuda")
        margins = adjacent_margins(logits, a, b, -1).flatten()
        if not torch.equal(value, previous) or margins.tolist() != [3., 4.] or not torch.equal(margins, -adjacent_margins(logits, b, a, -1).flatten()):
            raise ValueError("V22 initialization or adjacent margin orientation failed")
    modules = {name: copy_restorer(model, name) for name in SCOPES}
    counts = [sum(p.numel() for p in module.parameters() if p.requires_grad) for module in modules.values()]
    if not counts[0] < counts[1] < counts[2]:
        raise ValueError("Adaptation scopes are not nested")
    expect = calibration(model, core, data, ids, real, entry["seed"])[0]
    adaptive = update(modules, {key: optimizer_for(m) for key, m in modules.items()}, batch, core, {}, False)
    modules = {name: copy_restorer(model, "full_rgb") for name in LOSSES}
    auxiliary = update(modules, {key: optimizer_for(m) for key, m in modules.items()}, batch, core, expect, False)
    published = update({AUTHOR: author}, {AUTHOR: optimizer_for(author)}, batch, None, {}, True)
    witness = dict(passed=True, trainable=dict(zip(SCOPES, counts)), fixed_total=sum(p.numel() for p in model.parameters()),
                   output_exact_v22=True, calibration_weights=expect, losses=adaptive+auxiliary+published,
                   discarded_optimizer_updates=len(SCOPES)+len(LOSSES)+1, cuda=torch.version.cuda,
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(), finished_utc=now())
    atomic_write(ROOT / "smoke.json", witness)
    print(witness, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--author-only", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    ROOT.mkdir(parents=True, exist_ok=True)
    plan, sha = register()
    source, folds = prepare_grouped_data()
    load_proxy(source)
    by_video = {f["held_video"]: f for f in folds}
    real = prepare_real()
    if args.smoke:
        preflight(plan, source, by_video[plan["starting_points"][0]["video"]], real)
        return
    if not read(ROOT / "smoke.json")["passed"]:
        raise ValueError("Run the CUDA preflight before training")
    runs = []
    for entry in plan["author"] if args.author_only else plan["starting_points"]:
        fit_group = continue_author if args.author_only else paired_training
        runs.extend(fit_group(entry, sha, source, by_video[entry["video"]], real))
        gc.collect()
        torch.cuda.empty_cache()
        print(dict(video=entry["video"], seed=entry["seed"], author=args.author_only, complete_stages=len(runs)), flush=True)
    report = dict(all_complete=True, runs=runs, plan_SHA256=sha,
                  fits=sum(len(r["trainable"]) for r in runs),
                  updates=sum(r["steps_per_model"]*len(r["trainable"]) for r in runs), finished_utc=now())
    atomic_write(ROOT / ("author_completed.json" if args.author_only else "completed.json"), report)
    print({k: value for k, value in report.items() if k != "runs"}, flush=True)


if __name__ == "__main__":
    main()

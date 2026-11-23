"""Evaluate frozen adaptation scopes and consistency losses on the same grids."""
from collections import defaultdict
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from dehaze_seg.utils import write_csv
from scripts.boundary_consistency_v23 import LOSSES, SCOPES, semantic_scores
from scripts.boundary_flux_v21 import raw_affinities
from scripts.depth_correlated_fog_v20 import load_proxy
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_parallel_rgb_v8 import bind as bind_rgb
from scripts.evaluate_real_transfer_v22 import (
    FULL, METRICS, SPATIAL, add_metrics, averages, camvid_batches, load_transformer, real_batches,
)
from scripts.infer_boundary_consistency_v23 import restore_from_state
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_haze_data_v22 import ROOT as V22, load_archived_pairs, load_data
from scripts.regional_data_v18 import boundary_band, prepare_grouped_data
from scripts.train_boundary_consistency_v23 import ROOT, STAGES, verified_start
from scripts.train_real_transfer_v22 import AUTHOR, author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


SEMANTIC_INPUTS = ("hazy_input", "v22", "full_rgb", *LOSSES, "dehazeformer_t_continued")
CONTRASTS = {
    "tail_minus_head": ("decoder_tail", "head_only"),
    "full_minus_head": ("full_rgb", "head_only"),
    "full_minus_tail": ("full_rgb", "decoder_tail"),
    "photo_minus_v22": ("photo_only", "v22"),
    "class_minus_photo": ("class_boundary", "photo_only"),
    "class_minus_global": ("class_boundary", "global_semantic"),
    "class_minus_rgb": ("class_boundary", "rgb_boundary"),
    "class_minus_paper": ("class_boundary", "dehazeformer_t_continued"),
}


def training_lock():
    path = ROOT / "plan.json"
    plan, sha = read(path), digest(path)
    for file, expected in plan["source_SHA256"].items():
        if digest(file) != expected:
            raise ValueError("Training dependency changed: " + file)
    own = read(ROOT / "completed.json")
    authors = read(ROOT / "author_completed.json")
    for group, expected, starting in ((own, 42, "starting_points"), (authors, 6, "author")):
        if not group["all_complete"] or group["plan_SHA256"] != sha or group["fits"] != expected:
            raise ValueError("Need all fully trained V23 groups")
        inherited = {(row["video"], row["seed"]): row for row in plan[starting]}
        rows = {(row["video"], row["seed"], row["stage"]): row for row in group["runs"]}
        if set(rows) != {(*key, stage) for key in inherited for stage in STAGES} or len(rows) != len(group["runs"]):
            raise ValueError("A training stage is missing or duplicated")
        for key, old in inherited.items():
            for stage in STAGES:
                current = rows[*key, stage]
                if digest(current["checkpoint"]) != current["checkpoint_SHA256"] or current["initialized_from"] != old["checkpoint_SHA256"]:
                    raise ValueError("Weight provenance changed")
                old = current
    return plan, own, authors


def real_sources():
    datasets = load_data("evaluation")
    parent = read(V22 / "held_out_images.json")
    held, _, reference = load_archived_pairs("NH-HAZE.zip", "nh_repeat", parent["subsets"]["nh_holdout"])
    if reference["SHA256"] != parent["archive"]["SHA256"]:
        raise ValueError("NH-HAZE test source changed")
    datasets["nh_holdout"] = held
    ihaze = ROOT / "i_haze_manifest.json"
    if ihaze.exists():
        info = read(ihaze)
        if digest(info["file"]) != info["cache_SHA256"] or digest(info["archive"]) != info["archive_SHA256"]:
            raise ValueError("I-HAZE cache or source archive changed")
        datasets["i_haze"] = torch.load(info["file"], map_location="cpu", weights_only=True)
        if len(datasets["i_haze"]) != info["eligible"]:
            raise ValueError("I-HAZE paired count differs from audit")
    return datasets


def load_fit_group(parent, rows):
    core, initial, info = verified_start(parent)
    models = dict(v22=initial.requires_grad_(False))
    for result in rows:
        state = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        if any(state[k] != result[k] for k in ("stage", "video", "seed", "plan_SHA256", "initialized_from")):
            raise ValueError("Saved methods and run metadata differ")
        for name in state["models"]:
            models[name] = restore_from_state(initial, state, name)
    return core, models, info


def load_paper(parent, final):
    old, new = load_transformer(parent), load_transformer(parent)
    state = torch.load(final["checkpoint"], map_location="cpu", weights_only=True)
    parameters = dict(new.named_parameters())
    weights = state["models"][AUTHOR]
    if set(parameters) != set(weights) or state["stage"] != "consistency":
        raise ValueError("Incomplete continued DehazeFormer weights")
    for name, parameter in parameters.items():
        parameter.copy_(weights[name])
    return {"dehazeformer_t_v22": old, "dehazeformer_t_continued": new}


def predictions(models, authors, image):
    affinity = raw_affinities(image)
    result = dict(hazy_input=image, **{name: obj(image, affinity).clamp(0, 1) for name, obj in models.items()})
    result.update({name: author_rgb(model, image).clamp(0, 1) for name, model in authors.items()})
    return result


@torch.inference_mode()
def evaluate_group(parent, rows, paper, real, data, fold):
    core, models, info = load_fit_group(parent, rows)
    authors = load_paper(*paper) if paper else {}
    video, seed = info["video"], info["seed"]
    measurements, semantics = [], []
    for dataset, samples in real.items():
        for image, clean, ids, rects in real_batches(samples):
            add_metrics(measurements, predictions(models, authors, image), clean, ids, rects,
                        dataset, "paired", video, seed)
    for view in ("camera", "uniform035", *SPATIAL):
        names = [name for name in SEMANTIC_INPUTS if name in models or name in authors or name == "hazy_input"]
        whole = {name: torch.zeros((11, 11), dtype=torch.int64) for name in names}
        edge = {name: torch.zeros((11, 11), dtype=torch.int64) for name in names}
        native_counts = torch.zeros(11, dtype=torch.int64)
        edge_counts = torch.zeros(11, dtype=torch.int64)
        for image, clean, labels, ids in camvid_batches(data, fold, view):
            outputs = predictions(models, authors, image)
            add_metrics(measurements, outputs, clean, ids, [FULL]*len(image), "camvid_dev", view, video, seed)
            # Known classes within three pixels of a true class transition.
            nearby = labels.masked_fill(~boundary_band(labels), 255)
            native_counts += torch.bincount(labels.flatten(), minlength=256)[:11].cpu()
            edge_counts += torch.bincount(nearby.flatten(), minlength=256)[:11].cpu()
            for name in names:
                logits = semantic_scores(core, outputs[name])
                argmax = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
                whole[name] += confusion_matrix(argmax, labels, 11)
                edge[name] += confusion_matrix(argmax, nearby, 11)
        for name in names:
            if not torch.equal(whole[name].sum(1), native_counts) or not torch.equal(edge[name].sum(1), edge_counts):
                raise ValueError("Full-grid or boundary mask class counts changed")
            semantics.append(dict(video=video, seed=seed, condition=view, method=name,
                                  whole=segmentation_metrics(whole[name]), boundary=segmentation_metrics(edge[name]),
                                  whole_cm=whole[name].tolist(), boundary_cm=edge[name].tolist()))
    return measurements, semantics


@torch.inference_mode()
def pretrained_mct(real):
    bind_rgb()
    model, reference = load_author_reference()
    records = []
    for dataset, rows in real.items():
        for image, clean, ids, rects in real_batches(rows):
            output = {"dehazeformer_mct": author_rgb(model, image).clamp(0, 1)}
            add_metrics(records, output, clean, ids, rects, dataset, "paired", "external", -1)
    return records, reference


def paired_effects(records, datasets):
    source = defaultdict(dict)
    for row in records:
        if row["dataset"] in datasets:
            source[row["dataset"]][row["image"], row["video"], row["seed"], row["method"]] = row
    rng = np.random.default_rng(2309)
    result = {}
    for dataset, entries in source.items():
        result[dataset] = {}
        for name, (high, base) in CONTRASTS.items():
            scenes, groups = defaultdict(list), defaultdict(list)
            for (image, video, seed, method), score in entries.items():
                peer = entries.get((image, video, seed, base))
                if method == high and peer:
                    value = np.array([score[k]-peer[k] for k in METRICS])
                    scenes[image].append(value)
                    groups[f"{video}/{seed}"].append(float(value[0]))
            if len(scenes) != datasets[dataset]:
                raise ValueError("A paired contrast lacks the registered sample count")
            combined = np.array([np.mean(values, axis=0) for values in scenes.values()])
            samples = rng.integers(0, len(combined), size=(5000, len(combined)))
            bootstrap = np.percentile(combined[samples].mean(1), [2.5, 97.5], axis=0)
            result[dataset][name] = dict(mean=dict(zip(METRICS, combined.mean(0).tolist())),
                                         interval=dict(zip(METRICS, bootstrap.T.tolist())),
                                         psnr_by_group={key: float(np.mean(v)) for key, v in groups.items()},
                                         images=len(scenes), groups=len(groups), left=high, right=base)
    return result


def semantic_means(rows):
    selected = defaultdict(list)
    for row in rows:
        if row["condition"] in SPATIAL:
            selected[row["method"]].append(row)
    return {name: {key: float(np.mean([r[key]["miou"] for r in values])) for key in ("whole", "boundary")}
            for name, values in selected.items()}


def main():
    configure_cuda()
    plan, own, paper = training_lock()
    protocol_files = (Path(__file__), ROOT / "completed.json", ROOT / "author_completed.json", V22 / "held_out_images.json")
    extra = ROOT / "i_haze_manifest.json"
    sources = {str(p): digest(p) for p in (*protocol_files, *((extra,) if extra.exists() else ()))}
    design = dict(hash=sources, contrasts=CONTRASTS, real_RGB_grid=[256, 192], semantic_inputs=list(SEMANTIC_INPUTS),
                  scope="I-HAZE new RGB evaluation only if verified at entry. NH/Dense/O repeat V22. Model checkpoints fixed before scoring. All RGB metrics exclude letterbox padding. CamVid11 also measures 3px labeled boundary-band IoU with frozen segmenter fed alternative images.")
    lock = ROOT / "evaluation_plan.json"
    design["contrasts"] = {name: list(pair) for name, pair in CONTRASTS.items()}
    if lock.exists() and read(lock) != design:
        raise ValueError("Locked evaluation inputs changed")
    if not lock.exists():
        atomic_write(lock, design)
    real = real_sources()
    sizes = {k: len(v) for k, v in real.items()}
    data, folds = prepare_grouped_data()
    load_proxy(data)
    by_video = {f["held_video"]: f for f in folds}
    original_authors = {row["video"]: row for row in plan["author"]}
    adapted_authors = {row["video"]: row for row in paper["runs"] if row["stage"] == "consistency"}
    results, semantics = [], []
    for inherited in plan["starting_points"]:
        v, s = inherited["video"], inherited["seed"]
        subset = [row for row in own["runs"] if (row["video"], row["seed"]) == (v, s)]
        author = (original_authors[v], adapted_authors[v]) if s == 42 else None
        rows, masks = evaluate_group(inherited, subset, author, real, data, by_video[v])
        results.extend(rows)
        semantics.extend(masks)
        print(dict(video=v, seed=s, quality_rows=len(rows), semantic_rows=len(masks)), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    comparisons, mct = pretrained_mct(real)
    results.extend(comparisons)
    weighted = averages(results)
    table, sem = ROOT / "rgb_by_image.csv", ROOT / "camvid_semantics.json"
    write_csv(table, results)
    atomic_write(sem, semantics)
    report = dict(training_fits=own["fits"]+paper["fits"], updates=own["updates"]+paper["updates"], rgb_records=len(results),
                  RGB_real_pairs=sizes, rgb=weighted, effects=paired_effects(results, sizes),
                  camvid_spatial_mIoU=semantic_means(semantics), semantic_records=len(semantics),
                  bootstrap="Sample pairs within each dataset after averaging matched frozen model effects. Does not model retraining or independent sites.",
                  table_SHA256=digest(table), camvid_semantics_SHA256=digest(sem), mct=mct, finished_utc=now())
    atomic_write(ROOT / "evaluation.json", report)
    print(dict(training_fits=report["training_fits"], updates=report["updates"], pairs=sizes, rgb_rows=len(results),
               semantic_rows=len(semantics)), flush=True)


if __name__ == "__main__":
    main()

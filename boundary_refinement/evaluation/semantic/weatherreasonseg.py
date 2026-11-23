"""Evaluate positive semantic coverage on the released real-fog WeatherReasonSeg masks."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.segmentation.contrastive_response import (
    ROOT as V85, PRIMARY, predictions, load, reference_weights,
)
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.resolution_interaction import DualScaleFeatures, SEEDS, signed_views
from boundary_refinement.models.segmentation.restoration_affinity import compress_scores
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT = REPO.parent / "benchmark_data/weatherreasonseg-v85"
REVISION = "40f3a64857cde50632c38f767b956a9fe15802ba"
CLASSES = {6: "traffic light", 7: "traffic sign", 11: "person", 12: "rider", 13: "car",
           14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle"}
SMALL = (6, 7, 11, 12, 17, 18)
EXPECTED = {"GOPR0476": 46, "GP010476": 32, "GP020475": 22}


def metadata():
    release = read(ROOT / "release.json")
    if release["revision"] != REVISION or release["source"] != "https://huggingface.co/datasets/wanwan1111/WeatherReasonSeg":
        raise ValueError("Unexpected WeatherReasonSeg release")
    return release


def verify_files():
    files = metadata()["files"]
    for entry in files:
        file = ROOT / Path(entry["path"]).name
        if file.stat().st_size != entry["size"] or digest(file) != entry["lfs"]["oid"]:
            raise ValueError(f"Official parquet hash mismatch: {file.name}")
        yield file


def parse():
    local = ROOT / "deps"
    if local.is_dir():
        sys.path.insert(0, str(local))
    import pyarrow.parquet as parquet

    data, queries, seen = {}, Counter(), set()
    for path in verify_files():
        sequence = path.stem.removeprefix("fog_")
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=8):
            for item in batch.to_pylist():
                key, ann = item["image_id"], item["ann_id"]
                if not re.fullmatch(fr"{sequence}_frame_\d+_rgb_anon", key):
                    raise ValueError(f"Image and sequence disagree: {key}")
                if not ann.startswith(f"{key}_") or ann in seen:
                    raise ValueError(f"Unexpected or conflicting positive annotation {ann}")
                k = int(ann.removeprefix(f"{key}_"))
                if k not in CLASSES:
                    raise ValueError(f"Unexpected class index {ann}")
                seen.add(ann)
                size = item["img_height"], item["img_width"]
                rgb_id = sha256(item["image"]).hexdigest()
                if key not in data:
                    data[key] = (item["image"], rgb_id, np.full(size, 255, np.uint8), Counter())
                _, saved, label, regions = data[key]
                if rgb_id != saved or size != label.shape:
                    raise ValueError(f"RGB alignment mismatch in {key}")
                with Image.open(BytesIO(item["mask"])) as image:
                    region = np.array(image)
                if region.ndim != 2 or region.shape != size or not set(np.unique(region)) <= {0, 1, 255}:
                    raise ValueError(f"Invalid positive mask {ann}")
                active = region != 0
                if not active.any() or (label[active] != 255).any():
                    raise ValueError(f"Empty or cross-class overlapping mask {ann}")
                label[active] = k
                regions[k] += int(active.sum())
                queries[k] += 1
    cohorts = Counter(key.split("_")[0] for key in data)
    if cohorts != EXPECTED or sum(queries.values()) != 354:
        raise ValueError(f"Real-fog coverage differs from registration {cohorts}")
    if len({row[1] for row in data.values()}) != len(data):
        raise ValueError("Exact RGB duplicate across distinct frame keys")
    return data, queries


def open_rgb(data):
    with Image.open(BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


def registration():
    samples, queries = parse()
    draws, frame_rows, class_count = {}, [], Counter()
    for key, (source, identity, annotation, support) in samples.items():
        prefix = key.split("_")[0]
        frame_rows.append(dict(id=key, sequence=prefix, recording=prefix[-4:], rgb_SHA256=identity,
                               mask_SHA256=sha256(annotation.tobytes()).hexdigest(),
                               labelled_pixels=dict(support), total_pixels=annotation.size))
        class_count.update(support)
        for k in support:
            if k in draws:
                continue
            y, x = np.nonzero(annotation == k)
            # Pick the interior maximum of the largest local labelled mass for a reproducible visual audit.
            blocks, yi, xi = np.histogram2d(y, x, bins=[np.arange(0, annotation.shape[0] + 100, 80),
                                                       np.arange(0, annotation.shape[1] + 100, 80)])
            row, col = np.unravel_index(blocks.argmax(), blocks.shape)
            cy, cx = int(yi[row]) + 40, int(xi[col]) + 40
            a = min(max(0, cy - 96), annotation.shape[0] - 192)
            b = min(max(0, cx - 144), annotation.shape[1] - 288)
            area = np.s_[a:a + 192, b:b + 288]
            raw = open_rgb(source)[area]
            region = annotation[area].copy()
            region[region != k] = 255
            draws[k] = (f"{k} {CLASSES[k]} / {key}", [raw, blend(raw, region)], ("RGB crop", "Released positive mask"))
    out = ROOT / "positive-coverage.json"
    atlas = ROOT / "mask-check.png"
    draw(atlas, [draws[k] for k in CLASSES])
    atomic_write(out, dict(release_SHA256=digest(ROOT / "release.json"), code_SHA256=digest(__file__),
                 frames=frame_rows, counts=EXPECTED, query_counts=dict(queries), labelled_pixels=dict(class_count),
                 classes=CLASSES, recording_groups=sorted({row["recording"] for row in frame_rows}),
                 scope="Previously unused real WeatherReasonSeg validation fog. Annotation ID integer suffixes "
                       "correspond to Cityscapes train IDs on visual review. Use only named positive masks, ignore all "
                       "unannotated pixels. Collapse matched GoPro chapter numbers into the same recording. "
                       "The underlying frames are derived from ACDC, also the source dataset used for LiDSeg fine-tuning. "
                       "This is a new project evaluation and a limited transfer diagnostic.",
                 primary="Macro class positive-mask recall and traffic light/truck recall at original label resolution. "
                         "Report each of the two recording groups without inferential intervals. No 19-class mIoU.",
                 mask_check_SHA256=digest(atlas), utc=now()))
    print(dict(frames=len(frame_rows), positive_masks=sum(queries.values()), classes=dict(queries), registry=str(out)), flush=True)


def group_recall(rows):
    stats = defaultdict(list)
    for row in rows:
        cm = np.asarray(row["counts"], dtype=np.int64)
        stats["all", row["method"], row["seed"]].append(cm)
        stats[row["group"], row["method"], row["seed"]].append(cm)
    by_group = defaultdict(list)
    for (group, method, seed), counts in stats.items():
        cm = np.sum(counts, axis=0, dtype=np.int64)
        denominator = cm.sum(1)
        valid = [c for c in CLASSES if denominator[c] > 0]
        values = {c: 100. * cm[c, c] / denominator[c] for c in valid}
        small = [values[c] for c in SMALL if c in values]
        by_group[group, method].append(dict(seed=seed, class_recall=values, images=len(counts),
                                           mean_recall=float(np.mean(list(values.values()))),
                                           small_mean_recall=float(np.mean(small)),
                                           labelled_pixels=int(denominator.sum())))
    groups = {}
    for (group, method), detail in by_group.items():
        groups.setdefault(group, {})[method] = dict(
            each_seed=detail, class_recall={c: float(np.mean([r["class_recall"][c] for r in detail]))
                                           for c in detail[0]["class_recall"]},
            mean_recall=float(np.mean([r["mean_recall"] for r in detail])),
            small_mean_recall=float(np.mean([r["small_mean_recall"] for r in detail])))
    return groups


@torch.inference_mode()
def score():
    configure_cuda()
    reg = ROOT / "positive-coverage.json"
    card = read(reg)
    if digest(ROOT / "release.json") != card["release_SHA256"] or not matches_registered(card['code_SHA256'], digest(__file__)):
        raise ValueError("Positive-mask protocol changed")
    data, _ = parse()
    references, model = reference_weights(), load()
    engine = DualScaleFeatures()
    seg, mean, std, city = load_segmenter()
    rows, panels = [], {}
    for row in card["frames"]:
        key, group = row["id"], row["recording"]
        source, identity, target, _ = data[key]
        if identity != row["rgb_SHA256"] or sha256(target.tobytes()).hexdigest() != row["mask_SHA256"]:
            raise ValueError(f"The scored target has changed: {key}")
        original = open_rgb(source)
        truth = torch.from_numpy(target).cuda().long()
        views, restored, raw = engine(original)
        shifted = signed_views(views)
        fixed = torch.from_numpy(original).cuda().permute(2, 0, 1)[None].float() / 255.
        original_prediction = seg(pixel_values=(resize(fixed, restored.shape[-2:], antialias=True) - mean) / std).logits.float()
        raw_seg = resize(original_prediction, truth.shape).argmax(1)[0]
        rest_seg = resize(seg(pixel_values=(restored - mean) / std).logits.float(), truth.shape).argmax(1)[0]
        for seed in SEEDS:
            result = predictions(raw, views, model, references, seed, truth.shape, shifted)
            result["segformer_city_raw"] = raw_seg
            result["segformer_city_restored"] = rest_seg
            rows.extend(dict(dataset="weatherreasonseg_fog", id=key, group=group, method=method, seed=seed,
                             counts=confusion_matrix(z, truth, 19).tolist()) for method, z in result.items())
            if seed == SEEDS[0] and row["sequence"] not in panels:
                tags = ("raw_extra", "v84_fusion", "average_views", PRIMARY)
                masks = [blend(original, result[tag].byte().cpu().numpy()) for tag in tags]
                panels[row["sequence"]] = (key, [original, to_rgb(restored), *masks, paint(target)],
                                           ("Fog", "Restored RGB", "Raw continuation", "V84 fusion", "Ordinary mean",
                                            "Tone subtraction", "Positive masks"))
        if len(rows) % 600 == 0:
            print(dict(positive_prediction_records=len(rows)), flush=True)
    file, picture = V85 / "weatherreasonseg-individual.json.gz", V85 / "weatherreasonseg-predictions.png"
    compress_scores(file, rows)
    draw(picture, list(panels.values()))
    result = group_recall(rows)
    atomic_write(V85 / "weatherreasonseg-test.json", dict(metrics=result, frames=len(data), positive_masks=354,
                 data_SHA256=digest(reg), rows_SHA256=digest(file), picture_SHA256=digest(picture),
                 checkpoint_ledger_SHA256=digest(V85 / "completed.json"), segformer=city["model"], utc=now()))
    print({m: round(v["mean_recall"], 4) for m, v in result["all"].items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "score"))
    args = cli.parse_args()
    if args.stage == "register":
        registration()
    else:
        score()

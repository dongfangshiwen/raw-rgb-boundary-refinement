"""Test frozen RGB interventions with a common Cityscapes19 head on real fog."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.data.cohorts.audit_real_fog import city_train_ids
from boundary_refinement.evaluation.photographic.evaluate_author_mct import load_author_reference, OFFICIAL
from boundary_refinement.evaluation.photographic.evaluate_parallel_rgb import bind as bind_author
from boundary_refinement.inference.infer_real_transfer import load_model, restore
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_haze_data import ROOT, STORE
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ZURICH = REPO / "results/foggy-zurich-reservation/testv2_manifest.json"
DRIVING = REPO / "results/real-fog-alignment-v17/manifest.json"
SEGFORMER = STORE / "segformer-city19"
LONG_SIDE = 1024
REFINERS = ("synthetic_free", "affine_spatial", "paired_free", "paired_detail")
BASELINES = ("hazy_input", "dehazeformer_mct")


def load_segmenter():
    card = read(SEGFORMER / "registration.json")
    if (card["model"], card["revision"]) != ("nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                                            "21b3847fae21ddee674abd31129307b6a1235bd9"):
        raise ValueError("Unexpected public Cityscapes19 revision")
    for item in card["files"]:
        if digest(SEGFORMER / item["name"]) != item["SHA256"]:
            raise ValueError("Public Cityscapes19 file changed")
    model = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(SEGFORMER, local_files_only=True))
    weights = torch.load(SEGFORMER / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    if model.config.num_labels != 19 or model.config.id2label.get(0) != "road":
        raise ValueError("Expected the original Cityscapes train-ID sequence")
    processor = read(SEGFORMER / "preprocessor_config.json")
    mean, std = (torch.tensor(processor[key], device="cuda")[None, :, None, None] for key in ("image_mean", "image_std"))
    return model.cuda().eval(), mean, std, card


def verified_image(path, expected_sha, label):
    if digest(path) != expected_sha:
        raise ValueError("An evaluation-only image or label changed")
    with Image.open(path) as image:
        return np.array(image if label else image.convert("RGB"))


def load_pairs(source):
    if source == "foggy_zurich_testv2":
        manifest = read(ZURICH)
        if not manifest["all_passed"] or len(manifest["pairs"]) != 40:
            raise ValueError("Expected the official Foggy Zurich testv2 list")
        for row in manifest["pairs"]:
            a, b = row["rgb"], row["label"]
            yield Path(a["file"]).stem, row["sequence"], verified_image(a["file"], a["SHA256"], False), \
                  verified_image(b["file"], b["SHA256"], True)
    else:
        manifest = read(DRIVING)
        rows = [row for row in manifest["rows"] if row["annotation_kind"] == "gtFine"]
        if len(rows) != 33 or not manifest["original_GT_unchanged"]:
            raise ValueError("Expected the 33 official Foggy Driving fine annotations")
        for row in rows:
            image = verified_image(row["rgb"], row["rgb_SHA256"], False)
            target = city_train_ids(verified_image(row["annotation"], row["mask_SHA256"], True))
            yield row["id"], None, image, target


def resize_image(rgb):
    height, width = rgb.shape[:2]
    scale = LONG_SIDE/max(height, width)
    target = (round(height*scale), round(width*scale))
    raw = Image.fromarray(rgb).resize(target[::-1], Image.Resampling.BILINEAR)
    data = torch.from_numpy(np.array(raw)).permute(2, 0, 1)[None].cuda().float()/255
    return data, target


def restore_at_size(core, anchor, heads, image, size):
    h, w = size
    padded = F.pad(image, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
    return {name: v[..., :h, :w] for name, v in restore(core, anchor, heads, padded).items()}


def measure(model, mean, std, image, truth):
    logits = model(pixel_values=(image-mean)/std).logits.float()
    logits = F.interpolate(logits, truth.shape[-2:], mode="bilinear", align_corners=False)
    return confusion_matrix(logits.argmax(1), truth, 19)


def measured_record(source, pair, sequence, method, video, seed, matrix):
    return dict(dataset=source, image=pair, sequence=sequence, method=method, video=video, seed=seed,
                confusion=matrix.tolist())


def aggregate(rows):
    group = defaultdict(list)
    for row in rows:
        group[row["dataset"], row["method"], row["video"], row["seed"]].append(row["confusion"])
    by_fold = []
    summary = defaultdict(list)
    for (dataset, method, video, seed), values in group.items():
        matrix = np.sum(values, axis=0, dtype=np.int64)
        metrics = segmentation_metrics(torch.from_numpy(matrix))
        by_fold.append(dict(dataset=dataset, method=method, video=video, seed=seed, images=len(values),
                            miou=metrics["miou"], iou=metrics["class_iou"], pixel_acc=metrics["pixel_acc"],
                            truth_labels=int(np.count_nonzero(matrix.sum(1)))))
        summary[dataset, method].append(metrics["miou"])
    return by_fold, {f"{dataset}/{method}": float(np.mean(values)) for (dataset, method), values in summary.items()}


@torch.inference_mode()
def main():
    configure_cuda()
    trained = read(ROOT / "completed.json")
    if not trained["all_complete"] or trained["fits"] != 36:
        raise ValueError("Wait for all 36 frozen refiners")
    for run in trained["runs"]:
        if digest(run["checkpoint"]) != run["checkpoint_SHA256"]:
            raise ValueError("Restorer changed before real semantic inference")
    deps = (Path(__file__), ZURICH, DRIVING, ROOT / "completed.json", SEGFORMER / "registration.json",
            OFFICIAL / "reference_registration.json")
    design = dict(data={"foggy_zurich_testv2": 40, "foggy_driving_fine": 33}, image_long_side=LONG_SIDE,
                  hashes={str(p): digest(p) for p in deps}, refiners=list(REFINERS), baselines=list(BASELINES),
                  scope="Fixed Cityscapes19 SegFormer-B0 on all image interventions. Bilinear logits on native label grid. All 19 classes in macro IoU, ID 255 ignored. Outputs and variants fixed before test. No semantic training with these images.",
                  cautions="Foggy Driving was seen in earlier binary experiments. Both sets lack aligned RGB references. The fixed common Cityscapes segmenter is a diagnostic of restoration impact.")
    lock = ROOT / "real_semantic_plan.json"
    if lock.exists() and read(lock) != design:
        raise ValueError("Registered real semantic evaluation changed")
    if not lock.exists():
        atomic_write(lock, design)
    model, mean, std, card = load_segmenter()
    bind_author()
    mct, _ = load_author_reference()
    variants = [load_model(run["checkpoint"], REFINERS) for run in trained["runs"]]
    rows, counts = [], {}
    for source, expected in design["data"].items():
        count = 0
        for pair, sequence, rgb, annotation in load_pairs(source):
            if rgb.shape[:2] != annotation.shape or not np.isin(np.unique(annotation), [*range(19), 255]).all():
                raise ValueError("Native labels do not match the registered 19-class image")
            truth = torch.from_numpy(annotation.astype(np.int64, copy=False))[None].cuda()
            label_counts = torch.bincount(truth.flatten(), minlength=256)[:19].cpu()
            image, size = resize_image(rgb)
            methods = dict(hazy_input=image, dehazeformer_mct=author_rgb(mct, image).clamp(0, 1))

            def score(inputs, video, seed):
                for name, value in inputs.items():
                    matrix = measure(model, mean, std, value, truth)
                    if not torch.equal(matrix.sum(1), label_counts):
                        raise ValueError("Semantic intervention changed valid label counts")
                    rows.append(measured_record(source, pair, sequence, name, video, seed, matrix))

            score(methods, "external", -1)
            for core, anchor, heads, info in variants:
                score(restore_at_size(core, anchor, heads, image, size), info["video"], info["seed"])
            count += 1
            if count % 10 == 0:
                print(dict(dataset=source, scenes=count, expected=expected), flush=True)
        if count != expected:
            raise ValueError("Incomplete real fog test")
        counts[source] = count
    by_group, mean_iou = aggregate(rows)
    matrix_path = ROOT / "real_semantic_confusions.json"
    atomic_write(matrix_path, rows)
    report = dict(image_counts=counts, mean_equal_video_seed=mean_iou, by_group=by_group, segformer=card,
                  class_names=[model.config.id2label[i] for i in range(19)],
                  confusion_SHA256=digest(matrix_path), finished_utc=now())
    atomic_write(ROOT / "real_semantics.json", report)
    print(dict(evaluated=counts, mIoU={k: round(v, 5) for k, v in mean_iou.items()}), flush=True)


if __name__ == "__main__":
    main()

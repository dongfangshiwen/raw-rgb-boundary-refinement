"""Run the measured two-scale semantic baseline alongside native restoration."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import blend, boundary_ids, native_labels, paint, rgb_checkpoint
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_pairs, load_segmenter, resize_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT = REPO / "results/dual-view-v43"
B1 = REPO.parent / "benchmark_data/segformer-b1-v42"
RELEASE = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
REVISION = "ec86afeba68e656629ccf47e0c8d2902f964917b"
HASHES = {
    "config.json": "967a4a40a37783615f4e6767e6c2669ce8fae67411e6a8eda2426b0d9c536e23",
    "preprocessor_config.json": "697d92c83640ee381065f575ebdaef42d00d44483a827b92265026c0173dcbbf",
    "pytorch_model.bin": "9a7c40538e41d92f93ad6501d54e6333833a1d96bda7b0906c7255240e927832",
}


def load_b1():
    for name, value in HASHES.items():
        if digest(B1 / name) != value:
            raise ValueError("The larger SegFormer author release changed")
    module = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(B1, local_files_only=True))
    module.load_state_dict(torch.load(B1 / "pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
    if module.config.num_labels != 19 or module.config.id2label.get(0) != "road":
        raise ValueError("Expected the original Cityscapes train IDs")
    prep = read(B1 / "preprocessor_config.json")
    mean, std = (torch.tensor(prep[k], device="cuda")[None, :, None, None] for k in ("image_mean", "image_std"))
    return module.cuda().eval(), mean, std


def two_views(model, mean, std, image):
    coarse = model(pixel_values=(image - mean) / std).logits.float()
    shape = tuple(round(1.5 * size) for size in image.shape[-2:])
    bigger = F.interpolate(image, shape, mode="bilinear", align_corners=False)
    fine = model(pixel_values=(bigger - mean) / std).logits.float()
    align = F.interpolate(fine, coarse.shape[-2:], mode="bilinear", align_corners=False)
    mixture = (.5 * (coarse.softmax(1) + align.softmax(1))).clamp_min(1e-12).log()
    return coarse, fine, mixture


@torch.inference_mode()
def reference():
    configure_cuda()
    model, mean, std = load_b1()
    ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    control = read(REPO / "results/edge-disagreement-v42/test.json")
    for dataset, expected in (("foggy_zurich_testv2", 40), ("foggy_driving_fine", 33)):
        counts = 0
        scores = defaultdict(lambda: [torch.zeros((19, 19), dtype=torch.int64) for _ in range(2)])
        for _, _, rgb, mask in load_pairs(dataset):
            x, _ = resize_image(rgb)
            truth = torch.from_numpy(mask.astype(np.int64, copy=False)).cuda()
            edge = boundary_ids(truth)
            for key, y in zip(("low_raw", "high_raw", "prob_mean"), two_views(model, mean, std, x), strict=True):
                pred = native_labels(y, truth.shape)
                scores[key][0] += confusion_matrix(pred, truth, 19)
                scores[key][1] += confusion_matrix(pred, edge, 19)
            counts += 1
        if counts != expected:
            raise ValueError("Larger backbone test missed evaluation images")
        results[dataset] = {
            f"b1_{key}": dict(miou=100. * segmentation_metrics(v[0])["miou"],
                              edge_miou=100. * segmentation_metrics(v[1])["miou"]) for key, v in scores.items()
        }
        results[dataset].update({f"b0_{k}": control["summary"][f"{dataset}/{k}"]
                                 for k in ("low_raw", "high_raw", "prob_mean")})
    atomic_write(ROOT / "reference.json", dict(author=RELEASE, revision=REVISION, files=HASHES,
        trainable_parameters=0, model_parameters=sum(p.numel() for p in model.parameters()), results=results,
        protocol="Same 19-class real fog masks and native logit mapping for every model. B1 is an author checkpoint trained on Cityscapes with a larger backbone. No real fog weights are fit in this reference.",
        b0_SHA256=digest(REPO / "results/edge-disagreement-v42/test.json"), finished_utc=now()))
    print(results, flush=True)


@torch.inference_mode()
def infer(image, output):
    configure_cuda()
    seg, mean, std, card = load_segmenter()
    restorer, weights = rgb_checkpoint()
    with Image.open(image) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    raw, _ = resize_image(rgb)
    low, _, combined = two_views(seg, mean, std, raw)
    native = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].cuda().float() / 255.
    restored = to_rgb(pgl_rgb(restorer, native))
    mask = native_labels(combined, rgb.shape[:2]).byte().cpu().numpy()
    before = native_labels(low, rgb.shape[:2]).byte().cpu().numpy()
    if mask.shape != restored.shape[:2] or not np.isin(mask, range(19)).all():
        raise ValueError("Dual view outputs differ from the native RGB grid")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in (("restored", restored), ("mask", mask), ("color_mask", paint(mask)),
                        ("overlay", blend(rgb, mask)), ("baseline_overlay", blend(rgb, before))):
        path = output / (name + ".png")
        Image.fromarray(value).save(path)
        hashes[path.name] = digest(path)
    packet = dict(source=str(image.resolve()), input_SHA256=digest(image), size=list(mask.shape[::-1]),
                  outputs=hashes, rgb_checkpoint_SHA256=weights, semantic_checkpoint=card["model"],
                  semantic_input="Raw RGB at long edge 1024 plus the 1536-pixel view; equal probabilities on a common stride-four grid.",
                  pixels_changed_vs_low_raw=int(np.count_nonzero(mask != before)), finished_utc=now())
    atomic_write(output / "metadata.json", packet)
    print(dict(size=packet["size"], pixels_changed=packet["pixels_changed_vs_low_raw"], outputs=list(hashes)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest="action", required=True)
    cli.add_parser("reference")
    run = cli.add_parser("infer")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "reference":
        reference()
    else:
        if args.input.resolve() in {(args.output / n).resolve() for n in (
            "restored.png", "mask.png", "color_mask.png", "overlay.png", "baseline_overlay.png", "metadata.json"
        )}:
            parser.error("Use different input and output paths")
        infer(args.input, args.output)

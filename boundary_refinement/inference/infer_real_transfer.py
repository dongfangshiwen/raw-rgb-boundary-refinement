"""Restore RGB and predict CamVid11 labels on the original image grid."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from boundary_refinement.models.restoration.appearance_refiner import AppearanceRefiner, RECIPES, restoration_anchor
from boundary_refinement.models.joint.boundary_flux import raw_affinities
from boundary_refinement.models.joint.dual_coordinate import canonical_rgb
from boundary_refinement.inference.infer_boundary_flux import encoded_predictions
from boundary_refinement.data.provenance import write
from boundary_refinement.training.train_real_transfer import open_anchor
from boundary_refinement.training.train_regional_guidance import configure_cuda


def load_model(checkpoint, names=tuple(RECIPES)):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["format"] != "real-transfer-v22" or set(state["models"]) != set(RECIPES):
        raise ValueError("Expected matched V22 restoration checkpoint")
    if any(state[k] != state["base"][k] for k in ("video", "seed")):
        raise ValueError("Frozen trunk and weights disagree on fold")
    core, anchor = open_anchor(state["base"])
    heads = {}
    for name in names:
        model = AppearanceRefiner(state["seed"], RECIPES[name][1])
        model.load_state_dict(state["models"][name], strict=True)
        heads[name] = model.cuda().eval()
    info = {k: state[k] for k in ("video", "seed", "plan_SHA256")}
    return core, anchor, heads, info


def semantic_logits(core, image):
    _, logits = core.semantic(canonical_rgb(image, core.mean, "channel"))
    return F.interpolate(logits, image.shape[-2:], mode="bilinear", align_corners=False)


def restore(core, anchor_head, heads, image):
    baseline = restoration_anchor(core, anchor_head, image)
    guides = raw_affinities(image)
    return {"v21_rgb": baseline, **{name: model(image, baseline, guides).clamp(0, 1)
                                   for name, model in heads.items()}}


@torch.inference_mode()
def predict(core, anchor, heads, image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected B,3,H,W float RGB")
    h, w = image.shape[-2:]
    pad_h = max(32, 16*math.ceil(h/16)) - h
    pad_w = max(32, 16*math.ceil(w/16)) - w
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    images = restore(core, anchor, heads, padded)
    logits = semantic_logits(core, padded)
    return {name: rgb[..., :h, :w] for name, rgb in images.items()}, logits[..., :h, :w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=RECIPES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    core, anchor, heads, info = load_model(args.checkpoint, (args.method,))
    with Image.open(args.input) as source:
        data = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    image = torch.from_numpy(data).permute(2, 0, 1)[None].cuda().float()/255
    restored, logits = predict(core, anchor, heads, image)
    rgb, labels, overlay = encoded_predictions(restored[args.method], logits)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("restored.png", rgb), ("mask.png", labels), ("overlay.png", overlay)):
        Image.fromarray(pixels).save(args.output / name)
    write(args.output / "metadata.json", dict(size=list(labels.shape[::-1]), method=args.method, **info))


if __name__ == "__main__":
    main()

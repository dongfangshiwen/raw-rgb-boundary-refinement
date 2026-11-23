"""Export original-grid restored RGB and CamVid11 labels from adapted weights."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from boundary_refinement.models.joint.boundary_consistency import LOSSES, SCOPES, copy_restorer, load_adaptation, rgb, semantic_scores
from boundary_refinement.inference.infer_boundary_flux import encoded_predictions
from boundary_refinement.data.provenance import write
from boundary_refinement.training.train_boundary_consistency import verified_start
from boundary_refinement.training.train_regional_guidance import configure_cuda


METHODS = {**dict(zip(SCOPES, SCOPES)), **dict.fromkeys(LOSSES, "full_rgb")}


def restore_from_state(model, state, name):
    if state["stage"] not in ("adaptation", "consistency"):
        raise ValueError("Expected a registered V23 RGB training stage")
    registered = SCOPES if state["stage"] == "adaptation" else LOSSES
    if name not in registered or set(state["models"]) != set(registered):
        raise ValueError("Method and trained V23 stage differ")
    value = copy_restorer(model, METHODS[name])
    return load_adaptation(value, state["models"][name], METHODS[name]).eval().requires_grad_(False)


def load_model(checkpoint, name):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    core, start, info = verified_start(state["starting_point"])
    if any(state[k] != info[k] for k in ("video", "seed")):
        raise ValueError("The selected adaptation and segmenter belong to different video groups")
    return core, restore_from_state(start, state, name), {k: state[k] for k in ("video", "seed", "stage", "plan_SHA256")}


@torch.inference_mode()
def predict(core, restorer, image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expect a float RGB tensor with shape B,3,H,W")
    h, w = image.shape[-2:]
    ph, pw = max(32, 16*math.ceil(h/16)), max(32, 16*math.ceil(w/16))
    padded = F.pad(image, (0, pw-w, 0, ph-h), mode="replicate")
    image_scores = rgb(restorer, padded)
    logits = F.interpolate(semantic_scores(core, padded), padded.shape[-2:], mode="bilinear", align_corners=False)
    return image_scores[..., :h, :w], logits[..., :h, :w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    core, restorer, info = load_model(args.checkpoint, args.method)
    with Image.open(args.input) as im:
        array = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    image = torch.from_numpy(array).permute(2, 0, 1)[None].cuda().float()/255
    restoration, logits = predict(core, restorer, image)
    rgb, mask, overlay = encoded_predictions(restoration, logits)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("restored.png", rgb), ("mask.png", mask), ("overlay.png", overlay)):
        Image.fromarray(pixels).save(args.output / name)
    write(args.output / "metadata.json", dict(original_size=list(mask.shape[::-1]), method=args.method, **info))


if __name__ == "__main__":
    main()

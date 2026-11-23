"""Export a restoration and CamVid11 mask on the original raster."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from boundary_refinement.inference.infer_decision import PALETTE
from boundary_refinement.data.provenance import write
from boundary_refinement.training.regional_guidance import (
    CONTROLS, SELECTORS, TILE, RegionalGuidance, TileSelector, apply_choice, choose_tiles, selector_features,
)


def load_checkpoint(path, device="cuda"):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["format"] != "regional-guidance-v18" or set(value["selectors"]) != set(SELECTORS):
        raise ValueError("Invalid regional guidance checkpoint")
    model = RegionalGuidance(value["seed"], pretrained=False)
    model.load_state_dict(value["model"], strict=True)
    selectors = {}
    for name, state in value["selectors"].items():
        selector = TileSelector(value["seed"])
        selector.load_state_dict(state, strict=True)
        selectors[name] = selector.to(device).eval()
    return model.to(device).eval(), selectors, value


@torch.inference_mode()
def predict(model, selectors, image, method, device="cuda"):
    if method not in CONTROLS:
        raise ValueError("Choose a trained selector or an RGB control")
    array = torch.from_numpy(np.array(image.convert("RGB"))).to(device, dtype=torch.float32)
    raw = array.permute(2, 0, 1)[None] / 255
    height, width = raw.shape[-2:]
    full_h = max(2 * TILE, math.ceil(height / TILE) * TILE)
    full_w = max(2 * TILE, math.ceil(width / TILE) * TILE)
    padded = F.pad(raw, (0, full_w - width, 0, full_h - height), mode="replicate")
    anchor, rgb, _, guided, logits, probabilities = model(padded)
    if method in ("anchor", "rgb", "semantic"):
        restored = {"anchor": anchor, "rgb": rgb, "semantic": guided}[method]
    else:
        features, confidence = selector_features(padded, anchor, rgb, guided, probabilities)
        score = confidence if method == "entropy" else selectors[method](features)
        restored = apply_choice(rgb, guided, choose_tiles(score))
    saved_rgb = (restored[0, :, :height, :width].clamp(0, 1) * 255).round()
    saved_rgb = saved_rgb.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    mask = logits[0, :, :height, :width].argmax(0).to(torch.uint8).cpu().numpy()
    overlay = np.rint(0.60 * saved_rgb + 0.40 * PALETTE[mask]).astype(np.uint8)
    return saved_rgb, mask, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=CONTROLS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for regional guidance inference")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, selectors, checkpoint = load_checkpoint(args.checkpoint)
    with Image.open(args.input) as image:
        arrays = predict(model, selectors, image, args.method)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, array in zip(("restored.png", "mask.png", "overlay.png"), arrays):
        Image.fromarray(array).save(args.output / name)
    write(args.output / "metadata.json", dict(
        method=args.method, size=[int(n) for n in arrays[1].shape[::-1]], held_video=checkpoint["held_video"],
        seed=checkpoint["seed"], checkpoint=str(args.checkpoint.resolve()), device="cuda", classes=11,
    ))


if __name__ == "__main__":
    main()

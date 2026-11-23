"""Export the learned RGB correction and CamVid11 prediction on the input grid."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from boundary_refinement.models.joint.boundary_flux import METHODS, BoundaryRefiner, guidance
from boundary_refinement.inference.infer_decision import PALETTE
from boundary_refinement.inference.infer_regional_guidance import load_checkpoint
from boundary_refinement.data.provenance import digest, write
from boundary_refinement.training.train_regional_guidance import configure_cuda


DEPLOYABLE = tuple(method for method in METHODS if method != "oracle")


def load_refiners(checkpoint, methods=DEPLOYABLE):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["format"] != "boundary-flux-v21" or any(name not in state["refiners"] for name in methods):
        raise ValueError("Refiner checkpoint format or method mismatch")
    base = Path(state["base"])
    if digest(base) != state["base_SHA256"]:
        raise ValueError("Refiner and core checkpoint mismatch")
    core, _, metadata = load_checkpoint(base)
    if state["video"] != metadata["held_video"] or state["seed"] != metadata["seed"]:
        raise ValueError("Refiner and core video or seed differ")
    modules = {}
    for name in methods:
        refiner = BoundaryRefiner(state["seed"], name)
        refiner.load_state_dict(state["refiners"][name], strict=True)
        modules[name] = refiner.cuda().eval()
    return core, modules, {"video": state["video"], "seed": state["seed"]}


@torch.inference_mode()
def predict_many(core, refiners, source):
    """Float RGB on the original grid, with bottom/right replication padding internally."""
    if source.ndim != 4 or source.shape[1] != 3 or "oracle" in refiners:
        raise ValueError("Use a B,3,H,W RGB tensor and image-only refiners")
    h, w = source.shape[-2:]
    pad_h = max(32, 16 * math.ceil(h / 16)) - h
    pad_w = max(32, 16 * math.ceil(w / 16)) - w
    padded = F.pad(source, (0, pad_w, 0, pad_h), mode="replicate")
    anchor, _, _, _, logits, _ = core(padded)
    restored = {"anchor": anchor[..., :h, :w]}
    weights = guidance(padded, logits)
    for name, module in refiners.items():
        value, _, _ = module(padded, anchor, weights[name])
        restored[name] = value[..., :h, :w].clamp(0, 1)
    return restored, logits[..., :h, :w]


def encoded_predictions(restored, logits):
    rgb = (restored[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    labels = logits[0].argmax(0).byte().cpu().numpy()
    overlay = np.rint(0.60 * rgb + 0.40 * PALETTE[labels]).astype(np.uint8)
    return rgb, labels, overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=DEPLOYABLE)
    args = parser.parse_args()
    configure_cuda()
    base, modules, meta = load_refiners(args.checkpoint, (args.method,))
    with Image.open(args.input) as im:
        tensor = torch.from_numpy(np.array(im.convert("RGB"))).cuda().float()
    restored, logits = predict_many(base, modules, tensor.permute(2, 0, 1)[None] / 255)
    rgb, mask, overlay = encoded_predictions(restored[args.method], logits)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("restored.png", rgb), ("mask.png", mask), ("overlay.png", overlay)):
        Image.fromarray(pixels).save(args.output / name)
    write(args.output / "metadata.json", dict(method=args.method, size=list(mask.shape[::-1]), **meta))


if __name__ == "__main__":
    main()

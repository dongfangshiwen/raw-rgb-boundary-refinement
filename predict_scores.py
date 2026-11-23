"""Apply the source-fitted raw-RGB operator to a frozen, spatially aligned class-score grid."""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments" / "S3"))
from decoder import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help='An NPZ with a finite float array "scores" of shape C x h x w or 1 x C x h x w')
    parser.add_argument("guide", type=Path, help="Original RGB image covering the score grid's field of view")
    parser.add_argument("output", type=Path, help="Native class IDs as a PNG")
    parser.add_argument("--size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), help="Output size, defaulting to the guide size")
    parser.add_argument("--seed", type=int, choices=(101, 211, 307), default=101)
    parser.add_argument("--method", choices=("raw", "fixed"), default="raw")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with np.load(args.scores, allow_pickle=False) as archive:
        data = archive["scores"]
    if data.ndim == 3:
        data = data[None]
    if (data.ndim != 4 or data.shape[0] != 1 or not 1 <= data.shape[1] <= 256
            or min(data.shape[2:]) < 1 or data.dtype.kind != "f" or not np.isfinite(data).all()):
        parser.error("Expected one finite floating-point score grid with 1–256 channels")
    with Image.open(args.guide) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    shape = tuple(args.size) if args.size else rgb.shape[:2]
    if min(shape) < 1 or args.output.suffix.lower() != ".png":
        parser.error("Choose positive dimensions and a PNG output file")
    scores = torch.as_tensor(data, dtype=torch.float32, device=args.device)
    guide = torch.as_tensor(rgb, device=args.device).permute(2, 0, 1)[None].float() / 255
    with torch.inference_mode():
        labels = decode(scores, guide, shape, method=args.method, seed=args.seed).argmax(1)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels.byte().cpu().numpy()).save(args.output)


if __name__ == "__main__":
    main()

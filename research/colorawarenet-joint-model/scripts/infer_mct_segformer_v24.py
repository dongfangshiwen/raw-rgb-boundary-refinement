"""Run author MCT restoration and raw-image Cityscapes19 segmentation in parallel."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_parallel_rgb_v8 import bind as bind_rgb
from scripts.evaluate_real_semantics_v22 import load_segmenter, resize_image
from scripts.projective_ledger_v12 import digest, now
from scripts.real_detail_data_v24 import image_tensor
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


CITY_COLORS = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                       (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                       (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                       (0, 80, 100), (0, 0, 230), (119, 11, 32)], dtype=np.uint8)


@torch.inference_mode()
def predict(restorer, segmenter, mean, std, image):
    native = image_tensor(image)[None].cuda().float()/255
    rgb = author_rgb(restorer, native).clamp(0, 1)
    raw = np.asarray(image)
    semantic_input, _ = resize_image(raw)
    logits = segmenter(pixel_values=(semantic_input - mean)/std).logits.float()
    labels = F.interpolate(logits, native.shape[-2:], mode="bilinear", align_corners=False).argmax(1)[0]
    if rgb.shape != native.shape or tuple(labels.shape) != tuple(native.shape[-2:]):
        raise ValueError("Dual outputs must match the original pixel grid")
    if not torch.isfinite(rgb).all() or not torch.isfinite(logits).all():
        raise ValueError("The dual output contains nonfinite predictions")
    formatted = (255*rgb[0]).round().byte().permute(1,2,0).cpu().numpy()
    mask = labels.byte().cpu().numpy()
    overlay = np.rint(.6*formatted + .4*CITY_COLORS[mask]).astype(np.uint8)
    return formatted, mask, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    configure_cuda()
    bind_rgb()
    restorer, official = load_author_reference()
    segmenter, mean, std, city = load_segmenter()
    with Image.open(args.input) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    values = predict(restorer, segmenter, mean, std, image)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, data in zip(("restored.png", "mask.png", "overlay.png"), values):
        file = args.output / name
        Image.fromarray(data).save(file)
        files[name] = digest(file)
    metadata = dict(input=str(args.input.resolve()), input_sha256=digest(args.input), native_size=list(image.size),
                    restored_with=official["checkpoint_sha256"], segmentation=city["model"],
                    segmentation_input="raw_original", labels=segmenter.config.id2label,
                    output_sha256=files, updated_utc=now())
    atomic_write(args.output/"metadata.json", metadata)
    print(dict(original=metadata["native_size"], outputs=list(files)),flush=True)


if __name__ == "__main__":
    main()

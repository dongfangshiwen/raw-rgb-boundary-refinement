"""Predict a calibrated dehazed RGB and a native Cityscapes train-ID mask for a single RGB file."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from scripts import calibrate_tone_v99 as color
from scripts import semantic_subpixel_v94 as weighted
from scripts import subpixel_transport_v92 as base
from scripts.coarse_semantics_v39 import paint
from scripts.contrastive_response_v85 import raw_anchor
from scripts.cross_scale_rgb_v44 import resize
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import digest
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


@torch.inference_mode()
def predict(path, output, mode, seed):
    configure_cuda()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty prediction directory")
    with Image.open(path) as im:
        rgb = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    engine = base.DualScaleFeatures()
    views, deployed, _ = engine(rgb)
    if mode == "luma":
        restored = deployed
    else:
        fine = resize(torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.,deployed.shape[-2:],antialias=True)
        g = color.road.geometry(fine,engine.core.rgb)
        s = color.SEEDS[0]
        anchor,a,w = color.endpoints(g,engine.core.bank,color.heads(),s)
        restored = color.mixture(anchor,a,w,mode,color.load()[mode,s])
    z = raw_anchor(base.anchors()["raw_detail",seed],views)
    guides = base.render_guides(rgb,restored,z.shape[-2:])
    stencil = weighted.prepare(guides,z,"dual_core")
    delta = base.prediction("dual_free",weighted.load()["dual_core",seed],z,stencil)
    mask = (resize(z,rgb.shape[:2])+resize(delta,rgb.shape[:2])).argmax(1)[0].byte().cpu().numpy()
    output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(to_rgb(resize(restored,rgb.shape[:2]))).save(output/"restored.png")
    Image.fromarray(mask).save(output/"mask_train_ids.png")
    Image.fromarray(paint(mask)).save(output/"mask_color.png")
    atomic_write(output/"prediction.json",dict(input=digest(path),rgb_mode=mode,color_seed=color.SEEDS[0],
                 segmentation_seed=seed,shape=list(mask.shape),mask_labels="Cityscapes train IDs 0..18",
                 files={name:digest(output/name) for name in ("restored.png","mask_train_ids.png","mask_color.png")}))


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("image",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--rgb",choices=("curve","luma","constant"),default="curve")
    cli.add_argument("--seed",type=int,choices=base.SEEDS,default=base.SEEDS[0])
    args=cli.parse_args()
    predict(args.image,args.output,args.rgb,args.seed)

"""Predict a calibrated dehazed RGB and a native Cityscapes train-ID mask for a single RGB file."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from boundary_refinement.training import luminance_blend as color
from boundary_refinement.models.operators import donor_colors as weighted
from boundary_refinement.models.operators import subpixel_transport as base
from boundary_refinement.models.segmentation.coarse_semantics import paint
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import digest
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


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

"""Export a dehazed image and a native mask using frozen Mask2Former plus raw-image-guided decoding."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

from scripts import idd_fog_models_v100 as fit
from scripts import joint_tone_prototype_v99 as j
from scripts.idd_peer_source_v100 import BASE,SHA
from scripts.projective_ledger_v12 import digest,read,write


class ParallelPredictor:
    def __init__(self):
        fit.configure_cuda()
        if digest(BASE/"model.safetensors")!=SHA:
            raise ValueError("Cityscapes model does not match the registered release")
        self.segmenter=Mask2FormerForUniversalSegmentation.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).cuda().eval()
        cfg=read(BASE/"preprocessor_config.json")
        self.mean,self.std=(torch.tensor(cfg[k],device="cuda")[None,:,None,None] for k in ("image_mean","image_std"))
        self.mct,_=j.road.load_source();self.road_heads=j.road.load_trained("rgb")
        self.heads=j.calib.heads();self.coeff=j.calib.load()["curve",j.road.SEEDS[0]]
        self.decoders=j.weighted.load()

    @torch.inference_mode()
    def features(self,rgb):
        orig=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
        h,w=rgb.shape[:2];scale=min(1.,512/max(h,w))
        size=(max(1,round(h*scale)),max(1,round(w*scale)))
        x=j.resize(orig,size,antialias=True);h,w=size
        padding=(0,(-w)%32,0,(-h)%32)
        inp=F.pad((x-self.mean)/self.std,padding)
        valid=F.pad(torch.ones((1,h,w),device="cuda",dtype=torch.long),padding)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            q=self.segmenter(pixel_values=inp,pixel_mask=valid)
        native_grid=q.masks_queries_logits.shape[-2:]
        logits=fit.peer.query_to_scores(q.class_queries_logits,q.masks_queries_logits,native_grid,native_grid)
        z=logits[...,:((h+3)//4),:((w+3)//4)].contiguous()
        g=j.road.geometry(x,self.mct)
        u,a,win=j.calib.endpoints(g,self.road_heads,self.heads,j.road.SEEDS[0])
        restored=j.calib.mixture(u,a,win,"curve",self.coeff)
        # Align partial right/bottom cells with the segmentation feature lattice.
        raw4,rest4=(F.pad(t,(0,(-w)%4,0,(-h)%4),mode="replicate") for t in (x,restored))
        guides=torch.cat((raw4,rest4)).clamp(0.,1.).mul(255.).round().byte()
        return z,guides,restored,size

    @staticmethod
    def native_mask(z,delta,work,size):
        h,w=work;gh,gw=z.shape[-2:]
        if (h,w)==(4*gh,4*gw):
            return (j.resize(z,size)+j.resize(delta,size)).argmax(1)[0].byte()
        # Sample the valid content of the padded four-child grid directly.
        y=(torch.arange(size[0],device=z.device,dtype=z.dtype)+.5)*(2*h/(size[0]*4*gh))-1
        x=(torch.arange(size[1],device=z.device,dtype=z.dtype)+.5)*(2*w/(size[1]*4*gw))-1
        yy,xx=torch.meshgrid(y,x,indexing="ij");grid=torch.stack((xx,yy),dim=-1)[None]
        up=lambda t:F.grid_sample(t,grid,mode="bilinear",padding_mode="border",align_corners=False)
        return (up(z)+up(delta)).argmax(1)[0].byte()

    @torch.inference_mode()
    def predict(self,rgb,seed=101):
        z,guides,restored,size=self.features(rgb)
        weights=z.new_full((z.shape[-2]*z.shape[-1],16),1./16)
        stencil=j.weighted.prepare(guides[[0,0]],z,"dual_core",weighting=weights)
        delta=j.base.prediction("dual_free",self.decoders["dual_core",seed],z,stencil)
        mask=self.native_mask(z,delta,size,rgb.shape[:2])
        return j.to_rgb(j.resize(restored,rgb.shape[:2])),mask.cpu().numpy()


def export(input_path,output,seed):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty folder for the native predictions")
    with Image.open(input_path) as im:
        rgb=np.array(ImageOps.exif_transpose(im).convert("RGB"))
    pred=ParallelPredictor()
    restored,ids=pred.predict(rgb,seed)
    output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(restored).save(output/"restored.png")
    Image.fromarray(ids).save(output/"mask_train_ids.png")
    Image.fromarray(j.paint(ids)).save(output/"mask_color.png")
    write(output/"prediction.json",dict(input=digest(input_path),seed=seed,native_size=list(ids.shape),
        architecture="Parallel calibrated MCT restoration and frozen Cityscapes Mask2Former with uniform raw-RGB donor transport; 512 long side.",
        mask="Cityscapes train IDs 0..18",files={p.name:digest(p) for p in output.glob("*.png")}))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("image",type=Path);cli.add_argument("output",type=Path)
    cli.add_argument("--seed",type=int,choices=fit.SEEDS,default=fit.SEEDS[0]);args=cli.parse_args()
    export(args.image,args.output,args.seed)

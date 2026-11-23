"""Evaluate frozen RGB outputs against the matching WeatherProof clear crop."""
from pathlib import Path
from math import exp
import numpy as np
from PIL import Image
import torch
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.data.provenance import digest, now
from boundary_refinement.training.road_heads import load_source
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import configure_cuda
from boundary_refinement.data.staging.weatherproof_files import OUT, WORK, read, write

def tensor(path):
    with Image.open(path) as im:
        a=np.array(im.convert("RGB"))
    return torch.from_numpy(a).cuda().permute(2,0,1)[None].float()/255

@torch.inference_mode()
def run():
    configure_cuda()
    rows=read(OUT/"wp-manifest.json")["rows"]
    chosen=read(OUT/"evaluation.json")["example_ids"]["wp"]
    weights=WORK/"experiment_exports/raw-guidance-validation-2026-11-20/photo-weights.json"
    scalar=read(weights)["models"]["all/scalar"]
    assert scalar["slopes"]==[0.,0.,0.] and len(scalar["training_cities"])==23
    alpha=1/(1+exp(-scalar["bias"]))
    mct,mct_receipt=load_source()
    pgl,pgl_receipt=load_pgl()
    output=[]
    for i,row in enumerate(rows,1):
        files=[WORK/k for k in row["paths"][:2]]
        assert list(map(digest,files))==row["sha256"][:2]
        raw,clear=map(tensor,files)
        h,w=raw.shape[-2:]
        assert raw.shape==clear.shape
        ratio=min(1.,512/max(h,w));size=(round(ratio*h),round(ratio*w))
        x,y=(resize(v,size,antialias=True) for v in (raw,clear))
        corrected=author_rgb(mct,x).clamp(0,1)
        candidates=dict(input=x,mct=corrected,scalar=x+alpha*(corrected-x),pgl=pgl_rgb(pgl,x).clamp(0,1))
        losses={k:float((a-y).double().square().mean()) for k,a in candidates.items()}
        assert all(np.isfinite(m) and m>=0 for m in losses.values())
        rec=dict(id=row["id"],group=row["group"],weather=row["weather"],split=row["split"],size=size,mse=losses,
                 psnr={k:float(-10*np.log10(max(1e-12,v))) for k,v in losses.items()})
        output.append(rec)
        if row["id"]==chosen:
            directory=OUT/"examples/wp/photographs"
            directory.mkdir(parents=True,exist_ok=True)
            for name,im in {**candidates,"clear":y}.items():
                img=im[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
                Image.fromarray(img).save(directory/(name+".png"))
            write(directory/"metrics.json",rec)
        if i%40==0:print("Paired natural RGB",i,len(rows),flush=True)
    write(OUT/"rgb-individual.json",dict(rows=output,alpha=alpha,source=digest(weights),mct=mct_receipt,
         pgl=pgl_receipt,protocol=digest(OUT/"protocol.json"),manifest=digest(OUT/"wp-manifest.json"),
         code=digest(__file__),time=now()))

if __name__=="__main__":
    run()

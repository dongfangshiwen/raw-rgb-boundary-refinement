"""Save actual held-out scalar RGB predictions and verify each illustration against its score."""
from hashlib import sha256
from itertools import chain

import torch
import torch.nn.functional as F
from PIL import Image

from scripts import external_tone_v99 as ex
from scripts.fit_range_v101 import OUT
from scripts.photo_calibration_v101 import apply_gate
from scripts.projective_ledger_v12 import read
from scripts.train_real_transfer_v22 import author_rgb

@torch.inference_mode()
def main():
    ex.configure_cuda()
    mct,_=ex.road.load_source();models=read(OUT/"photo-weights.json");rows=read(OUT/"photo-individual.json")
    seen=set()
    # Both are the first frame in the original dataset enumerations, fixed in the photo protocol.
    for ds,key,group,hazy,ref,roi in chain(ex.city_pairs(ex.city_manifest()),((*r,None) for r in ex.indoor.pairs())):
        if ds in seen:continue
        seen.add(ds);fold=models["folds"][group] if ds=="bedde" else "all"
        h,w=hazy.shape[-2:];ratio=min(1.,576/max(h,w));sz=round(h*ratio),round(w*ratio)
        x,y=(ex.resize(t[None].cuda().float()/255,sz,antialias=True) for t in (hazy,ref))
        mask=torch.ones_like(y[0,0]).bool() if roi is None else F.interpolate(roi[None,None].cuda().float(),sz,mode="nearest-exact")[0,0].bool()
        m=author_rgb(mct,x).clamp(0,1)
        pred,alpha=apply_gate(x,m,models["models"][f"{fold}/scalar"])
        phase=[b%4 for b in sha256((ds+key).encode()).digest()[:2]]
        stats=ex.statistics(pred,y,mask,phase,ex.srgb_to_lab(y[...,phase[0]::4,phase[1]::4]))
        saved=next(r for r in rows if r["ds"]==ds and r["key"]==key and r["arm"]=="scalar")
        assert abs(saved["psnr"]-stats["psnr"])<.001 and abs(saved["deltaE"]-stats["deltaE"])<.002
        Image.fromarray(pred[0].permute(1,2,0).mul(255).round().byte().cpu().numpy()).save(OUT/f"photo-{ds}-scalar.png")
        print(ds,key,"fold",fold,"MCT fraction",float(alpha),"PSNR",stats["psnr"],flush=True)
        if len(seen)==2:break
if __name__=="__main__":main()

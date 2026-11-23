"""Frozen photographic transfer with author valid masks for BeDDE and real indoor REVIDE captures."""
from collections import Counter
from hashlib import sha256
from itertools import chain

import torch
import torch.nn.functional as F

from boundary_refinement.training import luminance_blend as c
from boundary_refinement.training import road_heads as road
from boundary_refinement.data.staging import stage_revide_remaining as indoor
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.training.conditional_fusion import BEDDE, city_manifest, city_pairs
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.metrics.color import ciede2000_lab, srgb_to_lab
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


def statistics(pred, gt, mask, phase, gt_lab):
    mse = ((pred-gt).square().mean(1)[0] * mask).sum()/mask.sum()
    small = mask[phase[0]::4,phase[1]::4]
    lab = srgb_to_lab(pred[...,phase[0]::4,phase[1]::4])
    if not bool(small.any()):
        raise ValueError("Empty color metric ROI")
    return dict(psnr=float(-10.*mse.clamp_min(1e-12).log10()),deltaE=float(ciede2000_lab(lab,gt_lab)[0][small].mean()),
                ab_error=float((lab-gt_lab)[0,...,1:].norm(dim=-1)[small].mean()))


@torch.inference_mode()
def run():
    configure_cuda()
    methods = c.load()
    author, release = road.load_source()
    public, pgl_release = load_pgl()
    cfg = dict(script=digest(__file__), coefficients=digest(c.ROOT/"coefficients.json"), frozen=digest(c.matched.ROOT/"completed.json"),
               sources=[digest(BEDDE),digest(indoor.MANIFEST)], refs=dict(mct=release,pgl=pgl_release),
               rule="Frozen evaluation on all 208 BeDDE images (23 cities, valid-author-mask only) and all 16 "
                    "previously reserved REVIDE images (8 capture codes). Neither cohort in calibration or model "
                    "training here; both were used in earlier project experiments. Group-equal paired comparisons. "
                    "Full RGB metric on <=576 px working grid, color metrics at a fixed 4x4 sampling phase. Also "
                    "score the unmodified public MCT and PGL checkpoints; their pretraining and budgets differ.")
    lock = c.ROOT/"external-protocol.json"
    if lock.exists() and read(lock)!=cfg:
        raise ValueError("Frozen external protocol changed")
    if not lock.exists():
        atomic_write(lock,cfg)
    trained, controlled = road.load_trained("rgb"), c.heads()
    rows, counts = [], Counter()
    for ds,key,group,hazy,ref,roi in chain(city_pairs(city_manifest()), ((*r,None) for r in indoor.pairs())):
        h,w = hazy.shape[-2:]
        scale = min(1.,576/max(h,w))
        size = (round(h*scale),round(w*scale))
        x,y = (resize(im[None].cuda().float()/255.,size,antialias=True) for im in (hazy,ref))
        mask = torch.ones_like(y[0,0]).bool() if roi is None else F.interpolate(
            roi[None,None].cuda().float(),size,mode="nearest-exact")[0,0].bool()
        phase = [v%4 for v in sha256((ds+key).encode()).digest()[:2]]
        gt_lab = srgb_to_lab(y[...,phase[0]::4,phase[1]::4])
        geometry = road.geometry(x,author)
        pictures = {("public_mct",0):geometry.base,("public_pgl",0):pgl_rgb(public,x).clamp(0.,1.),("input",0):x}
        for s in c.SEEDS:
            anchor,a,w = c.endpoints(geometry,trained,controlled,s)
            pictures["affine",s], pictures["luma",s] = a,w
            for m in c.GATES:
                pictures[m,s] = c.mixture(anchor,a,w,m,methods[m,s])
        for (arm,s),pic in pictures.items():
            rows.append(dict(ds=ds,key=key,group=group,arm=arm,seed=s,valid_pixels=int(mask.sum()),
                             **statistics(pic,y,mask,phase,gt_lab)))
        counts[ds]+=1
        if sum(counts.values())%50==0:
            print("photographic transfer",dict(counts),flush=True)
    if counts!=dict(bedde=208,revide_final=16):
        raise ValueError("Missing transfer cases")
    data=c.ROOT/"external-individual.json"
    atomic_write(data,rows)
    atomic_write(c.ROOT/"external-test.json",dict(ledger=digest(data),protocol=digest(lock),sets=c.summary(rows),counts=counts,utc=now()))


if __name__=="__main__":
    run()

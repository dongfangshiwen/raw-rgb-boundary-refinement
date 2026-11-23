"""A matched one-scalar diagnostic for semantic versus image-only color directions."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.models.operators import subpixel_transport as old
from boundary_refinement.evaluation.semantic import range_transfer as ev
from boundary_refinement.training.fit_range_kernel import OUT
from boundary_refinement.models.operators.range_kernel import prepare, RangeDecoder, RANGES
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band

AXES=("class_axis","rgb_axis","luma_axis")
LOCK=OUT/"axis-protocol.json"

def basis(z,guide,st):
    active,_,_,_,_,donor=st
    h,w=z.shape[-2:]
    rgb=old.pack(guide[0],h,w).float()/255
    means=rgb.mean(1)
    offsets=torch.cartesian_prod(*(torch.arange(-1,2,device=z.device),)*2)
    nb=((active[:,None]//w+offsets[:,0]).clamp(0,h-1)*w+
        (active[:,None]%w+offsets[:,1]).clamp(0,w-1))
    neighbors=means[nb]
    delta=rgb[active,:,None]-neighbors[:,None]
    posterior=donor.softmax(-1)
    top=posterior.mean(1).topk(2,-1).indices
    alpha=posterior.gather(-1,top[:,None].expand(-1,9,-1))
    centroids=(alpha/alpha.sum(1,keepdim=True).clamp_min(1e-12)).transpose(1,2) @ neighbors
    pairwise=(neighbors[:,:,None]-neighbors[:,None,:]).square().sum(-1)
    farthest=pairwise.flatten(1).argmax(-1)
    k=torch.arange(active.numel(),device=z.device)
    axes={"class_axis":centroids[:,0]-centroids[:,1],
          "rgb_axis":neighbors[k,farthest//9]-neighbors[k,farthest%9],
          "luma_axis":z.new_tensor([.2126,.7152,.0722]).expand(len(k),-1)}
    out={}
    for name,axis in axes.items():
        norm=axis.square().sum(-1,keepdim=True).sqrt()
        projection=(delta*(axis/norm.clamp_min(1e-7))[:,None,None]).sum(-1).square()
        out[name]=torch.where((norm>1e-7)[:,:,None],projection,st[3])
    return out

class ProjectedRange(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_mix=nn.Parameter(torch.tensor(-2.))
        self.kernel=RangeDecoder("bilinear_bounded4").requires_grad_(False)

    def correction(self,z,st,projected):
        idx,base,dist,d,_,donor=st
        adjusted=d+self.log_mix.sigmoid()*(projected-d)
        features=1-(-.5*adjusted.clamp_min(0)[...,None]/z.new_tensor(RANGES).square()).exp()
        return self.kernel.correction(z,(idx,base,dist,d,features,donor))

def fit():
    old.configure_cuda()
    bank=old.train_frames(old.DualScaleFeatures(),old.anchors(),False)
    runs=[]
    for i,seed in enumerate(ev.SEEDS):
        old.set_seed(seed);rng=np.random.default_rng(92001+seed)
        nets={a:ProjectedRange().cuda() for a in AXES}
        opt={a:torch.optim.AdamW([m.log_mix],lr=.035,weight_decay=0.) for a,m in nets.items()}
        sched={a:torch.optim.lr_scheduler.CosineAnnealingLR(o,320,eta_min=.0035) for a,o in opt.items()}
        curve={a:[] for a in AXES}
        for step in range(320):
            data=[]
            for f in (False,True):
                z,g,truth=old.train_crop(bank,rng,i,f,step);g=g[:1]
                st=prepare(z,g);axes=basis(z,g,st)
                native=F.interpolate(z,scale_factor=4,mode="bilinear",align_corners=False)
                t=truth[8:-8,8:-8];edge=boundary_band(truth)[8:-8,8:-8]
                wt=(t<19).float()*(1+3*edge+2*((t==6)|(t==7)|(t==14)))
                data.append((z,st,axes,native,t,wt))
            for a,net in nets.items():
                opt[a].zero_grad();loss=z.new_zeros(())
                for logits,st,axes,native,t,wt in data:
                    y=(native+net.correction(logits,st,axes[a]))[...,8:-8,8:-8]
                    ce=F.cross_entropy(y,t[None],reduction="none",ignore_index=255)[0]
                    loss=loss+.5*(ce*wt).sum()/wt.sum().clamp_min(1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([net.log_mix],5,error_if_nonfinite=True)
                opt[a].step();sched[a].step();curve[a].append(loss.item())
        for a,net in nets.items():
            runs.append(dict(arm=a,seed=seed,log_mix=net.log_mix.item(),alpha=net.log_mix.sigmoid().item(),
                        losses=curve[a],sampler_end=hashlib.sha256(str(rng.bit_generator.state).encode()).hexdigest()))
        print("Finished three direction fits",seed,flush=True)
    old.atomic_write(OUT/"axis-weights.json",dict(protocol=old.digest(LOCK),implementation=old.digest(__file__),
        initial_fitting_source=old.digest(OUT/"training.json"),source_images=sum(map(len,bank)),groups=len(bank),updates=2880,runs=runs))

@torch.inference_mode()
def score(cohort,name):
    old.configure_cuda();backbone=ev.Target(name)
    result=old.read(OUT/"axis-weights.json")
    assert result["protocol"]==old.digest(LOCK) and matches_registered(result['implementation'], old.digest(__file__))
    models={}
    for row in result["runs"]:
        m=ProjectedRange().cuda().eval();m.log_mix.fill_(row["log_mix"]);models[row["arm"],row["seed"]]=m
    path=OUT/f"counts-{name}-{cohort}-axis";path.mkdir(exist_ok=True)
    for i,r in enumerate(ev.rows(cohort)):
        file=path/(r["id"].replace("/","_")+".npz")
        rgb,truth=ev.scene(r,cohort);size=truth.shape
        metrics=ev.ev.Counts(truth)
        batch,(h,w)=ev.setup.peer_input(rgb)
        image=batch[...,:h,:w].contiguous()
        z,_=backbone(image,size);guide=image.clamp(0,1).mul(255).round().byte()
        st=prepare(z,guide);axes=basis(z,guide,st)
        native=ev.resize(z,size)
        arms=[]
        for a in AXES:
            multi=[]
            for s in ev.SEEDS:
                dz=models[a,s].correction(z,st,axes[a])
                pred=(native+ev.resize(dz,size)).argmax(1)[0]
                multi.append(np.stack(metrics.score(pred)))
            arms.append(np.stack(multi,axis=1))
        counts=np.stack(arms,axis=1).astype(np.int32)
        temp=file.with_suffix(".part")
        with temp.open("wb") as f:
            np.savez_compressed(f,counts=counts,arms=AXES,seeds=ev.SEEDS,id=r["id"],group=r["group"],
                                lock=old.digest(OUT/"axis-weights.json"))
        temp.replace(file)
    print(name,cohort,"axis counts",i+1,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("step",choices=("fit","score"));p.add_argument("--cohort",default="bddval")
    ar=p.parse_args()
    if ar.step=="fit":fit()
    else:
        for target in ("mask2former","segformer"):score(ar.cohort,target)

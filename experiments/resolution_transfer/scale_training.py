"""Fit a native-supervised raw-colour decoder at one or two score resolutions."""
from boundary_refinement.paths import SOURCE_ROOT
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.resolution_transfer.cohort import OUT, PRIOR, ROOT, digest, read, reserve, write
from experiments.recording_gate import gate
from boundary_refinement.evaluation.controls.audit_dual_output import AUTHOR_HASH

SIDES=(512,768,1024)
TRAIN_SIDES=(512,1024)
SEEDS=(101,211,307)
METHODS=("low_only","high_only","shared","by_scale","shuffled_scale")
STEPS=320
PER_REGION=768


def now():return datetime.now(timezone.utc).isoformat()


def lock():
    reserve()
    src=gate.source_rows()
    main={"question":"Is learning different colour weights for coarse and fine score grids necessary, and do mixed-resolution weights transfer better than the manuscript or fixed RGB ranges?",
          "model":"Preserve manuscript score placement for every colour arm: directly enlarged coarse class scores plus a virtual-grid colour residual. No acceptance gate. Four bounded RGB ranges and one spatial exponent per expert. Compare fixed, shifted guide, original three paper seeds and released backbone.",
          "trained_arms":list(METHODS),"source":"Frozen, unadapted author LiDSeg on 96 MUSES fog frames from nine training recordings only. At native RGB resolution per training frame fix 768 legal reference-boundary and 768 uniformly valid pixels. Two input scales share their native coordinates. Every step has two slots with two distinct recordings each and 128 + 128 pixels per frame (balanced boundary and uniform). Same frame/pixels in all arms. Single-scale arms use that scale in both slots, scale-aware and pooled process one slot per scale. The equal-capacity shuffled control routes the two scales in one half of source frames with a deterministic per-frame swap independent of labels. Mix effective coefficients linearly in log2 input scale at 768. No target optimization.",
          "optimization":"Three common source seeds, AdamW at .035, zero decay, cosine decay to .0035; 320 fixed updates; native cross entropy. All arms have matched native samples per step and initial four ranges. No stopping by target feedback.",
          "evaluation":"Native radius-six reference-boundary and full mIoU; fixed 19-class macro, zero for zero union. Primary target Mask2Former on preregistered new label-unseen random BDD prefixes at long side 768 which no fitted arm sees. Secondary SegFormer-B1, endpoints 512/1024, MUSES 24+58 real fog frames on nine known evaluation recordings, and four previously reported visually classified BDD fog keyframes. Report previously inspected MUSES as replication only. For MUSES pool counts and show both splits; for 4 no inferential CI. Three seeds averaged after pooling matrices; paired percentile bootstrap with 5000 draws, RNG 11321.",
          "decision":"Call scale-conditioning supported only if by_scale beats both shared and shuffled_scale (paired lower 95% above zero) at primary new 768 endpoint and the second backbone has the same sign against each. Treat independent fitted advantage over original and fixed as a separate test requiring positive lower bounds vs both controls and a positive sign on the other backbone. No new fog generalization claim on the random-weather set. Report signed losses and nulls. Do not tune on these outcomes.",
          "sources":{n:digest(p) for n,p in {"trainer":Path(__file__),"source_loader":Path(gate.__file__),"reserve":OUT/"reserve.json",
                  "fit":ROOT/"benchmark_data/muses-v70/fit.json","calibrate":ROOT/"benchmark_data/muses-v70/calibrate.json",
                  "held":ROOT/"benchmark_data/muses-v70/held.json","published_operator":SOURCE_ROOT / "boundary_refinement/models/operators/range_kernel.py"}.items()},
          "source_recordings":sorted({r["group"] for r in src["fit"]}),"seeds":list(SEEDS),"scales":list(SIDES)}
    write(OUT/"protocol.json",main,True)
    return digest(OUT/"protocol.json")


def image_at(rgb, side):
    h,w=rgb.shape[:2]
    shape=tuple(round(n*side/max(h,w)/4)*4 for n in (h,w))
    x=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
    return gate.ev.resize(x,shape,antialias=True).contiguous()


def geometry(ids, src_shape, dst_shape):
    """Four input samples of align_corners=False bilinear interpolation at output ids."""
    h,w=src_shape;H,W=dst_shape
    yy=(ids//W+.5)*(h/H)-.5; xx=(ids%W+.5)*(w/W)-.5
    y0=yy.floor().long();x0=xx.floor().long()
    a,b=yy-y0,xx-x0
    top=y0.clamp(0,h-1)*w;bot=(y0+1).clamp(0,h-1)*w
    left=x0.clamp(0,w-1);right=(x0+1).clamp(0,w-1)
    neighbors=torch.stack((top+left,top+right,bot+left,bot+right),-1)
    weights=torch.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b),-1)
    return neighbors,weights


def sample_features(z, image, shape, ids, y):
    h,w=z.shape[-2:];wh,ww=image.shape[-2:]
    assert (wh,ww)==(4*h,4*w)
    raw=image.mul(255).round().clamp(0,255).div(255)
    pixels=raw[0].permute(1,2,0).reshape(-1,3)
    centers=F.avg_pool2d(raw,4,4)[0].permute(1,2,0).reshape(-1,3)
    classes=z.argmax(1,keepdim=True).float()
    active=(F.max_pool2d(classes,3,1,1)!=-F.max_pool2d(-classes,3,1,1)).flatten()
    children,blend=geometry(ids,(wh,ww),shape)
    blend=blend*active[(children//ww//4)*w+(children%ww//4)]
    donor,prior=geometry(children,(h,w),(wh,ww))
    dist=(pixels[children][:,:,None,:]-centers[donor]).square().mean(-1)
    bands=z.new_tensor((.02,.05,.10,.20))
    features=-torch.expm1(-.5*dist[...,None]/bands.square())
    scores=z[0].permute(1,2,0).reshape(h*w,-1)
    direct,weights=geometry(ids,(h,w),shape)
    base=(scores[direct]*weights[...,None]).sum(1)
    return base,blend,prior,features,scores[donor],y.flatten()[ids]


def output(pack, effective):
    base,blend,prior,features,donors,_=pack
    strength,exponent=effective[:4],effective[4]
    distance=(features*strength).sum(-1)
    logw=exponent*prior.clamp_min(1e-12).log()-distance
    correction=((logw.masked_fill(prior==0,-torch.inf).softmax(-1)-prior)[...,None]*donors).sum(-2)
    return base+(correction*blend[...,None]).sum(1)


def coefficients(theta):
    return torch.cat((6*theta[...,:4].sigmoid(),.25+1.5*theta[...,4:].sigmoid()),-1)


def weights_at(theta, side):
    pairs=coefficients(theta)
    if len(pairs)==1:return pairs[0]
    ratio=math.log2(side/512)
    return torch.lerp(pairs[0],pairs[1],ratio)


def range_model(effective):
    e=torch.as_tensor(effective,dtype=torch.float32,device="cuda")
    m=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    m.mix.copy_(torch.logit((e[:4]/6).clamp(1e-7,1-1e-7)))
    m.spatial.copy_(torch.logit(((e[4]-.25)/1.5).clamp(1e-7,1-1e-7)))
    return m


def checked_coordinates(row,y):
    edge=gate.boundary_band(y).flatten();valid=(y<19).flatten()
    seed=int.from_bytes(sha256(("source-native-113/"+row["id"]).encode()).digest()[:4],"big")
    gen=torch.Generator(device="cuda").manual_seed(seed)
    parts=[]
    for mask in (edge&valid,valid):
        idx=torch.nonzero(mask).flatten()
        if not len(idx):idx=torch.nonzero(valid).flatten()
        assert len(idx)
        parts.append(idx[torch.randint(len(idx),(PER_REGION,),device="cuda",generator=gen)])
    return torch.cat(parts)


@torch.no_grad()
def encode():
    model=gate.Source()
    bank=defaultdict(list);audit=[];coord_hash=sha256()
    swap_rng=np.random.default_rng(11300)
    rows=gate.source_rows()["fit"]
    flags=np.repeat((False,True),len(rows)//2);swap_rng.shuffle(flags)
    for i,r in enumerate(rows):
        rgb,y=gate.open_muses(r)
        ids=checked_coordinates(r,y)
        coord_hash.update(ids.cpu().numpy().tobytes())
        pair={}
        for scale in TRAIN_SIDES:
            im=image_at(rgb,scale)
            z,_=model(im,(1,1))
            pack=sample_features(z,im,y.shape,ids,y)
            pair[scale]=tuple(t.contiguous() for t in pack)
            if i in (0,len(rows)-1):
                # Float32 sparse training scores must match the full deployed decoder on the native grid.
                e=z.new_tensor([1.1,2.7,1.6,2.3,.79])
                kernel=range_model(e)
                full=kernel(z,im.mul(255).round().byte(),y.shape)[0].reshape(19,-1)[:,ids].T
                err=(output(pack,e)-full).abs().max().item()
                if not torch.allclose(output(pack,e),full,atol=.002,rtol=.0005):
                    raise ValueError(f"Native sparse decoder disagrees with dense reference: {err}")
                audit.append(dict(id=r["id"],scale=scale,max_abs=err))
        bank[r["group"]].append((pair,bool(flags[i])))
        if (i+1)%24==0:print("Encoded native source targets",i+1,len(rows),flush=True)
    return list(bank.values()),audit,coord_hash.hexdigest()


def fit():
    gate.ev.setup.configure_cuda();lock_sha=lock()
    dest=OUT/"training.json"
    if dest.is_file():
        assert read(dest)["protocol_sha256"]==lock_sha
        return
    start=time.monotonic()
    bank,audit,coord_sha=encode()
    runs=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed+11300); gen=torch.Generator(device="cuda").manual_seed(seed+11300)
        params={m:torch.nn.Parameter(torch.tensor([[-.5]*4+[0.]]*(2 if m in ("by_scale","shuffled_scale") else 1),
                                                  dtype=torch.float32,device="cuda")) for m in METHODS}
        optim=torch.optim.AdamW(list(params.values()),lr=.035,weight_decay=0)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,STEPS,eta_min=.0035)
        trace={m:[] for m in METHODS}
        sample_sha=sha256()
        for step in range(STEPS):
            optim.zero_grad(set_to_none=True)
            losses={m:[] for m in METHODS}
            for slot,scale in enumerate(TRAIN_SIDES):
                for group in rng.choice(len(bank),2,replace=False):
                    fi=int(rng.integers(len(bank[group])))
                    pair,swap=bank[group][fi]
                    select=torch.cat([torch.randint(PER_REGION,(128,),generator=gen,device="cuda")+k*PER_REGION for k in (0,1)])
                    sample_sha.update(np.array([slot,group,fi],np.int32).tobytes()+select.cpu().numpy().tobytes())
                    sub={res:tuple(x[select] for x in values) for res,values in pair.items()}
                    for m,p in params.items():
                        side={"low_only":512,"high_only":1024}.get(m,scale)
                        use=(1536-scale) if m=="shuffled_scale" and swap else scale
                        pack=sub[side]
                        losses[m].append(F.cross_entropy(output(pack,weights_at(p,use)),pack[-1]))
            means={m:torch.stack(vals).mean() for m,vals in losses.items()}
            sum(means.values()).backward()
            for m,p in params.items():
                torch.nn.utils.clip_grad_norm_([p],5,error_if_nonfinite=True)
                trace[m].append(float(means[m].detach()))
            optim.step();sched.step()
        for m,p in params.items():
            runs.append(dict(method=m,seed=seed,steps=STEPS,parameters=p.numel(),theta=p.detach().cpu().tolist(),
                             effective={str(s):weights_at(p.detach(),s).cpu().tolist() for s in SIDES},
                             cross_entropy=trace[m],sample_sha256=sample_sha.hexdigest()))
        print("Finished five matched source fits for seed",seed,flush=True)
    write(dest,dict(runs=runs,updates=STEPS*len(runs),images=96,source_recordings=len(bank),
                    source_coordinate_sha256=coord_sha,protocol_sha256=lock_sha,reference_checks=audit,
                    source_id="Unadapted author LiDSeg",source_model_sha256=AUTHOR_HASH,
                    device=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
                    seconds=round(time.monotonic()-start,2),finished_utc=now()),True)


if __name__=="__main__":fit()

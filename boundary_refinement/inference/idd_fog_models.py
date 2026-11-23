"""Fit a donor-variance intervention on MUSES and transfer the published Dehaze-Seg-S port."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.evaluation.paired import muses_transmission as j
from boundary_refinement.evaluation.semantic import resolution_interaction as anchor
from boundary_refinement.models.operators import subpixel_transport as base
from boundary_refinement.baselines import camvid as peer
from boundary_refinement.data.cohorts.idd_fog_data import OUT, fixed
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import paths
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


SEEDS=base.SEEDS
STEPS=320
CAM=REPO/"results/dual-coordinate-v17/models/seed_42/published_scratch/best.pth"
METHODS=("constant_variance","local_variance")


class FastJoint:
    """The raw-detail head only reads the raw slots. Evaluate each used input once."""
    def __init__(self):
        self.engine=base.DualScaleFeatures()
        self.heads=j.calib.heads()
        self.coeff=j.calib.load()["curve",j.road.SEEDS[0]]

    @torch.no_grad()
    def encode(self, rgb, side=None):
        tensor=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
        h,w=rgb.shape[:2]
        ratio=min(1.,(side or anchor.FINE_SIDE)/max(h,w))
        size=round(h*ratio),round(w*ratio)
        fine=j.resize(tensor,size,antialias=True)
        grid=tuple((n+3)//4 for n in size)
        high=self.engine.logits(fine)
        if side is None:
            coarse,_=anchor.resize_image(rgb)
            low=self.engine.logits(coarse)
        else:
            low=high
        def raw_slots(z):
            return j.resize(z,grid).unsqueeze(1).expand(-1,3,-1,-1,-1)
        views=anchor.Views(raw_slots(low),raw_slots(high))
        core=self.engine.core
        g=j.road.geometry(fine,core.rgb)
        u,a,wnd=j.calib.endpoints(g,core.bank,self.heads,j.road.SEEDS[0])
        restored=j.calib.mixture(u,a,wnd,"curve",self.coeff)
        return views,restored,low,wnd

    def __call__(self,rgb):
        views,restored,low,_=self.encode(rgb)
        return views,restored,low


class DonorVariance(nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff=nn.Parameter(torch.full((2,),-2.0))

    def stencil(self, guides,z,mode,weights=None,template=None):
        h,w=z.shape[-2:]
        if weights is None:
            weights=j.weighted.color_weights(z,"dual_core")
        if template is None:
            template=base.guide_stencil(guides[:0],z.argmax(1)[0])
        idx,neighbor,prior,dist,_=template
        kernels=[]
        for ch in (0,1):
            pixel=base.pack(guides[ch],h,w).float()/255.
            donor=(weights[...,None]*pixel).sum(1)
            if mode=="local_variance":
                variance=(weights*(pixel-donor[:,None]).square().mean(-1)).sum(1)
                extra=variance[neighbor][:,None,:,None]
            elif mode=="constant_variance":
                extra=.05**2
            else:
                raise ValueError(mode)
            err=(pixel[idx,:,None]-donor[neighbor][:,None]).square().mean(-1,keepdim=True)
            bands=z.new_tensor(base.RANGES).square()+self.coeff[ch].sigmoid()*extra
            kernels.append(1.-(-.5*err/bands).exp())
        return idx,neighbor,prior,dist,kernels


def register():
    root=REPO/"results"
    sources=[base.DATA/"fit.json",base.ROOT/"completed.json",j.weighted.ROOT/"completed.json",
             anchor.ROOT/"completed.json",j.road.ROOT/"rgb-completed.json",j.calib.ROOT/"coefficients.json",
             j.calib.matched.ROOT/"completed.json",CAM,peer.__file__,
             REPO.parent/"benchmark_data/lidseg-author-v66/lidseg_pidnetl_uber.pth",
             REPO.parent/"benchmark_data/published-models-v12/source_registration.json"]
    for s in SEEDS:
        sources.extend((j.weighted.ROOT/f"dual_core-{s}.pth",base.ROOT/f"dysample-{s}.pth",
                        anchor.ROOT/f"raw_detail-{s}.pth"))
    for a in ("affine","luma"):
        sources.append(j.calib.matched.ROOT/f"{a}-{j.road.SEEDS[0]}.pth")
    plan=dict(implementation=digest(__file__),frozen={str(p):digest(p) for p in sources},seeds=list(SEEDS),updates=STEPS,
        fit="Only the 96 MUSES fog training frames from nine drives. No IDD labels or images for fitting. Fixed last step.",
        variance="Freeze the deployed dual_core color transport. Add two sigmoid coefficients to inflate the four "
                 "squared bandwidths by either each donor's weighted RGB variance or a constant 0.05 squared. "
                 "Both arms use identical seeds, two 28-cell crops per update and boundary-weighted CE. Adam .03 cosine -> .003.",
        published="Published Dehaze-Seg-S adaptation: original S backbone, frequency neck and Mask2Former port, local "
                  "CamVid11 1104-update model as initialization. Replace the 11-class classifier with 19+void; adapt frequency "
                  "neck, pixel decoder, query decoder and classifier using one drive-uniform MUSES frame per step at 512 long edge. "
                  "Frozen image backbone and RGB branch. AdamW 0.0001 (classifier .001), cosine x0.1, fixed 320 steps. "
                  "Three runs. Native labels and same single 512-long-side input for published comparisons. "
                  "This is an adaptation, not the UAV checkpoint or original optimizer/dataset. Initialization and compute differ.",
        evaluation="Native 6-pixel valid semantic-boundary mIoU (primary), full mIoU (secondary), 17 exact shared "
                   "IDD-Cityscapes classes, per-drive paired bootstrap 5000 draws averaging three seeds. Fog and mixed cohorts "
                   "separate. Compare fixed vs uniform (same net) and DySample, candidate local vs fixed and constant. "
                   "Published comparisons at matched single-512 input; deploy-size comparisons separately. Retain all outcomes.")
    fixed(OUT/"model-protocol.json",plan)
    return digest(OUT/"model-protocol.json")


def pdf_path(method,s):
    return OUT/f"{method}-{s}.pth"


def existing(method,s,lock):
    path=pdf_path(method,s)
    note=path.with_suffix(".json")
    if note.exists():
        receipt=read(note)
        if receipt["weights"]!=digest(path) or receipt["protocol"]!=lock:
            raise ValueError(f"Changed {method}-{s}")
        return receipt


def finish(method,s,state,losses,lock,params):
    path=pdf_path(method,s)
    torch_save(path,{k:v.detach().cpu() for k,v in state.items()})
    rec=dict(method=method,seed=s,steps=STEPS,weights=digest(path),protocol=lock,trainable=params,
             losses=losses,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),utc=now())
    atomic_write(path.with_suffix(".json"),rec)
    print("finished",method,s,losses[-1],flush=True)
    return rec


def fit_variances(lock):
    if all(existing(m,s,lock) for m in METHODS for s in SEEDS):
        return
    core=FastJoint()
    fitted=j.weighted.load()
    raws=base.anchors()
    # Verify the shorter path on a training image before using any external outcome.
    sample=next(paths("fit"))[-1]
    with torch.no_grad():
        old,wnd,_=core.engine(sample)
        views,_,_,check=core.encode(sample)
        diff=max((j.raw_anchor(raws["raw_detail",s],old)-j.raw_anchor(raws["raw_detail",s],views)).abs().max().item()
                 for s in SEEDS)
        torch.testing.assert_close(check,wnd,atol=1e-5,rtol=1e-5)
        if diff>1e-5:
            raise ValueError(f"Different raw anchor {diff}")
        fixed(OUT/"path-parity.json",dict(logit_max_error=diff,window_rgb_max_error=(wnd-check).abs().max().item()))
    pool=base.train_frames(core,raws,False)
    for s in SEEDS:
        for method in METHODS:
            if existing(method,s,lock):
                continue
            set_seed(s)
            model=DonorVariance().cuda()
            opt=torch.optim.Adam(model.parameters(),lr=.03)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.003)
            rng=np.random.default_rng(92001+s)
            history=[]
            for step in range(STEPS):
                opt.zero_grad(set_to_none=True)
                for focus in (False,True):
                    z,guides,truth=base.train_crop(pool,rng,SEEDS.index(s),focus,step)
                    stencil=model.stencil(guides,z,method)
                    logits=F.interpolate(z,scale_factor=4.,mode="bilinear",align_corners=False)
                    logits=logits+base.prediction("dual_free",fitted["dual_core",s],z,stencil)
                    loss=j.gate.seg_loss(logits,truth)/2
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step();sched.step()
                if step%40==0 or step+1==STEPS:
                    history.append(dict(step=step+1,last_crop_loss=float(loss.detach())*2,
                                        strength=model.coeff.detach().sigmoid().cpu().tolist()))
            finish(method,s,model.state_dict(),history,lock,2)


def peer_model(seed):
    model=peer.PublishedCamVid11(42)
    src=torch.load(CAM,weights_only=True,map_location="cpu")
    model.load_state_dict(src["model_state"])
    old=model.seg_head.head.class_predictor
    # Exact CamVid semantic matches or broader parent groups for initialization only.
    parents=[3,4,1,None,7,2,None,6,5,None,0,9,None,8,8,8,None,None,10]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        clf=nn.Linear(old.in_features,20)
    with torch.no_grad():
        for k,p in enumerate([*parents,11]):
            if p is not None:
                clf.weight[k].copy_(old.weight[p]);clf.bias[k].copy_(old.bias[p])
    model.seg_head.head.class_predictor=clf
    return model.cuda()


def peer_input(rgb):
    x=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
    h,w=rgb.shape[:2]
    scale=min(1.,512/max(h,w))
    size=(round(scale*h),round(scale*w))
    x=j.resize(x,size,antialias=True)
    return F.pad(x,(0,512-size[1],0,512-size[0]),mode="replicate"),size


def peer_forward(model,features,size):
    cls,mask=model.seg_head(model.freq_neck(features))
    return peer.query_to_scores(cls,mask,(512,512),size)


def fit_peer(lock):
    if all(existing("published",s,lock) for s in SEEDS):
        return
    model=peer_model(SEEDS[0]).eval()
    pool=defaultdict(list)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for key,g,_,_,truth,rgb in paths("fit"):
            x,size=peer_input(rgb)
            _,*feat=model.backbone((x-model.mean)/model.std)
            target=F.interpolate(truth[None,None].float(),size,mode="nearest-exact")[0,0].byte().cpu()
            pool[g].append(([x.detach().to(device="cpu",dtype=torch.bfloat16) for x in feat],size,target))
    del model
    if len(pool)!=9 or sum(map(len,pool.values()))!=96:
        raise ValueError("Peer source split mismatch")
    pool=list(pool.values())
    for s in SEEDS:
        if existing("published",s,lock):
            continue
        set_seed(s)
        model=peer_model(s)
        model.backbone.eval().requires_grad_(False)
        model.freq_neck.train();model.seg_head.train()
        head=model.seg_head.head.class_predictor
        groups=[p for p in model.parameters() if p.requires_grad and all(p is not q for q in head.parameters())]
        opt=torch.optim.AdamW([dict(params=groups,lr=.0001),dict(params=head.parameters(),lr=.001)],weight_decay=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:.1+.9*(1+np.cos(np.pi*k/STEPS))/2)
        rng=np.random.default_rng(100920+s)
        history=[]
        for step in range(STEPS):
            group=pool[int(rng.integers(len(pool)))]
            features,size,y=group[int(rng.integers(len(group)))]
            truth=y.cuda().long()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                scores=peer_forward(model,[f.cuda() for f in features],size)
                w=(truth<19).float()*(1+3*j.boundary_band(truth))
                ce=F.cross_entropy(scores,truth[None],ignore_index=255,reduction="none")[0]
                loss=(ce*w).sum()/w.sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();sched.step()
            if step%80==0 or step+1==STEPS:
                history.append(dict(step=step+1,loss=float(loss.detach())))
                print("published",s,history[-1],flush=True)
        part={k:v for k,v in model.state_dict().items() if k.startswith(("freq_neck.","seg_head."))}
        finish("published",s,part,history,lock,sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model,opt,part,groups,head,sched


def trained_peer(seed):
    model=peer_model(seed)
    adapted=torch.load(pdf_path("published",seed),weights_only=True,map_location="cpu")
    current=model.state_dict()
    if set(adapted)!={k for k in current if k.startswith(("freq_neck.","seg_head."))}:
        raise ValueError("Incomplete published semantic branch")
    current.update(adapted)
    model.load_state_dict(current)
    return model.eval().requires_grad_(False)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part",choices=("register","fit","peer","all"))
    args=cli.parse_args()
    configure_cuda()
    lock=register()
    if args.part in ("fit","all"):
        fit_variances(lock)
    if args.part in ("peer","all"):
        fit_peer(lock)

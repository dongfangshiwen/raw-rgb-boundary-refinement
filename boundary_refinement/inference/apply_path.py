"""Portable frozen decoder. Inputs are released backbone logits and its aligned raw RGB."""
import torch
import torch.nn.functional as F
from boundary_refinement.models.operators.path_kernel import Stencil, RANGES
from boundary_refinement.models.operators.range_kernel import prepare

@torch.inference_mode()
def predict(logits,working_rgb,native_size,checkpoint,*,road_manuscript=True):
    """Return native class IDs for one image. Use road_manuscript=False for the ADE probe."""
    if logits.ndim!=4 or logits.shape[0]!=1 or not logits.is_floating_point():
        raise ValueError("Expected one C-channel floating-point logit tensor [1,C,H,W]")
    if working_rgb.ndim!=4 or working_rgb.shape[:2]!=(1,3) or not working_rgb.is_floating_point():
        raise ValueError("Expected aligned raw RGB [1,3,H,W] in [0,1]")
    if not ((working_rgb>=0)&(working_rgb<=1)).all():
        raise ValueError("RGB must be in [0,1]")
    z=logits.float();guide=working_rgb.to(device=z.device,dtype=torch.float32)
    size=tuple(native_size)
    if len(size)!=2 or min(size)<1:raise ValueError("Invalid native output size")
    state=torch.load(checkpoint,map_location=z.device,weights_only=True)
    w2,tau,gamma=(state[k] for k in ("range_by_view","spatial","strength"));w=w2.sum(0)
    if w2.shape!=(2,4) or tau.numel()!=1 or gamma.numel()!=1:raise ValueError("Invalid decoder checkpoint")
    up=lambda a:F.interpolate(a,size,mode="bilinear",align_corners=False)
    native=up(z);st=Stencil(z,guide);ids=st.active.nonzero().flatten()
    if not len(ids):return native.argmax(1)[0]
    lift=lambda d:up(d.reshape(4*st.h,4*st.w,st.c).permute(2,0,1).contiguous()[None])
    correction=z.new_zeros((16*st.h*st.w,st.c));delta=torch.zeros_like(correction)
    if road_manuscript:
        idx,base,_,_,phi,donor=prepare(z,st.image.mul(255).round().byte())
        cost=phi@w2[0]+phi@w2[1]
        log=(tau*base.clamp_min(1e-10).log()-cost).masked_fill(base==0,-torch.inf)
        d=(log.softmax(-1)-base)@donor;grid=z.new_zeros((st.h*st.w,16,st.c)).index_copy_(0,idx,d)
        grid=grid.reshape(st.h,st.w,4,4,st.c).permute(4,0,2,1,3).reshape(1,st.c,4*st.h,4*st.w)
        native=native+up(grid)
    for chunk in ids.split(6144):
        b,dist,path,_,donor=st.features(chunk);bands=dist.new_tensor(RANGES)
        log=b.log()*tau-(-torch.expm1(-.5*dist[...,None]/bands.square()))@w
        baseline=log.softmax(-1);augmented=(log-gamma*path).softmax(-1)
        delta[chunk]=((augmented-baseline)[:,None]@donor).squeeze(1)
        if not road_manuscript:correction[chunk]=((baseline-b)[:,None]@donor).squeeze(1)
    if not road_manuscript:native=native+lift(correction)
    return (native+lift(delta)).argmax(1)[0]

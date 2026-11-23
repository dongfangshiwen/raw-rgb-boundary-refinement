"""Fourfold score decoding from an aligned, unit-range original RGB image.
scores: [1,C,h,w] frozen logits (Mask2Former uses logged aggregated scores).
rgb: [1,3,H,W], original view at the same field of view as scores.
native_size: (target_height,target_width). Returns native class scores.
"""
from pathlib import Path
from hashlib import sha256
import json
import torch
import torch.nn.functional as F
from torch import nn
from range_decoder import RangeDecoder,prepare

BASE=Path(__file__).resolve().parent
def resize(x,size,antialias=False):
    return F.interpolate(x,size,mode="bilinear",align_corners=False,antialias=antialias)

class AdaptiveConv:
    @staticmethod
    def apply(image,weights):
        k=weights.shape[-1]
        return (image.unfold(2,k,1).unfold(3,k,1)*weights[:,None]).sum((-2,-1))

def bilateral(seed,device):
    file=BASE/"featup-author/upsamplers.py"
    assert sha256(file.read_bytes()).hexdigest()=="e213e36c1f592c30aebbc527954539c006342f5d1a7fa01e382f5166c6cb0004"
    code=file.read_text(encoding="utf-8");line="from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv"
    assert code.count(line)==1
    ns={"__name__":"featup_pinned","AdaptiveConv":AdaptiveConv}
    exec(compile(code.replace(line,""),str(file),"exec"),ns)
    block=ns["JBULearnedRange"]
    m=nn.Module();m.up1=block(3,19,32,radius=3);m.up2=block(3,19,32,radius=3)
    weights=torch.load(BASE/f"weights/featup-{seed}.pth",map_location="cpu",weights_only=True)
    m.load_state_dict(weights,strict=True)
    return m.to(device).eval()

@torch.inference_mode()
def decode(scores,rgb,native_size,method="raw",seed=101):
    if scores.shape[0]!=1 or rgb.shape[:2]!=(1,3):raise ValueError("Expected one image and one score grid")
    z=scores.float();shape=tuple(4*x for x in z.shape[-2:])
    guide=rgb.to(z)
    if guide.shape[-2:]!=shape:guide=resize(guide,shape,True)
    quant=guide.clamp(0,1).mul(255).round().byte()
    if method in ("raw","fixed"):
        n=RangeDecoder("bilinear_bounded4").to(z.device).eval()
        if method=="raw":
            params=json.loads((BASE/"design/fixed-evaluation.json").read_text())["settings"][str(seed)]
            n.mix.copy_(torch.logit(z.new_tensor(params["colour"])/6))
            n.spatial.copy_(torch.logit(z.new_tensor((params["exponent"]-.25)/1.5)))
        corr=n.correction(z,prepare(z,quant))
    elif method in ("featup","featup-gated"):
        if z.shape[1]!=19:raise ValueError("The fitted FeatUp comparison used 19 road classes")
        net=bilateral(seed,z.device);guide=quant.to(z)/255
        guide=(guide-z.new_tensor([.485,.456,.406])[None,:,None,None])/z.new_tensor([.229,.224,.225])[None,:,None,None]
        x=z
        for stage in (net.up1,net.up2):
            x=stage(x,F.adaptive_avg_pool2d(guide,tuple(2*d for d in x.shape[-2:])))
        corr=x-resize(z,shape)
        if method=="featup-gated":
            lab=z.argmax(1,keepdim=True).float()
            gate=F.max_pool2d(lab,3,1,1)!=-F.max_pool2d(-lab,3,1,1)
            corr=corr*gate.repeat_interleave(4,2).repeat_interleave(4,3)
    else:raise ValueError(method)
    return resize(z,native_size)+resize(corr,native_size)

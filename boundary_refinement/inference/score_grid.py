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
from boundary_refinement.models.operators.range_kernel import RangeDecoder, prepare
from boundary_refinement.paths import PACKAGE_ROOT, SOURCE_ROOT

S3 = SOURCE_ROOT / "experiments" / "S3"
COEFFICIENTS = PACKAGE_ROOT / "models" / "parameters" / "decoder_coefficients.json"
def resize(x,size,antialias=False):
    return F.interpolate(x,size,mode="bilinear",align_corners=False,antialias=antialias)

class AdaptiveConv:
    @staticmethod
    def apply(image,weights):
        k=weights.shape[-1]
        return (image.unfold(2,k,1).unfold(3,k,1)*weights[:,None]).sum((-2,-1))

def bilateral(seed, device, assets=None):
    directory = S3 if assets is None else Path(assets)
    file = directory / "featup-author/upsamplers.py"
    if not file.is_file():
        raise FileNotFoundError(f"Missing FeatUp source in {directory}. Use the expanded S3 directory.")
    contents = file.read_bytes()
    if sha256(contents).hexdigest() != "e213e36c1f592c30aebbc527954539c006342f5d1a7fa01e382f5166c6cb0004":
        raise ValueError(f"Unexpected FeatUp source digest: {file}")
    code=contents.decode("utf-8");line="from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv"
    if code.count(line) != 1:
        raise ValueError("The pinned FeatUp module has an unexpected adaptive convolution import")
    ns={"__name__":"featup_pinned","AdaptiveConv":AdaptiveConv}
    exec(compile(code.replace(line,""),str(file),"exec"),ns)
    block=ns["JBULearnedRange"]
    m=nn.Module();m.up1=block(3,19,32,radius=3);m.up2=block(3,19,32,radius=3)
    weight_path = directory / f"weights/featup-{seed}.pth"
    manifest = json.loads((directory / "SHA256SUMS.json").read_text(encoding="utf-8"))
    expected = manifest[f"weights/featup-{seed}.pth"]
    if sha256(weight_path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"FeatUp weights differ from the S3 release: {weight_path}")
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    m.load_state_dict(weights,strict=True)
    return m.to(device).eval()

@torch.inference_mode()
def decode(scores, rgb, native_size, method="raw", seed=101, *, s3_assets=None):
    if scores.shape[0]!=1 or rgb.shape[:2]!=(1,3):raise ValueError("Expected one image and one score grid")
    z=scores.float();shape=tuple(4*x for x in z.shape[-2:])
    guide=rgb.to(z)
    if guide.shape[-2:]!=shape:guide=resize(guide,shape,True)
    quant=guide.clamp(0,1).mul(255).round().byte()
    if method in ("raw","fixed"):
        n=RangeDecoder("bilinear_bounded4").to(z.device).eval()
        if method=="raw":
            params=json.loads(COEFFICIENTS.read_text())["settings"][str(seed)]
            n.mix.copy_(torch.logit(z.new_tensor(params["colour"])/6))
            n.spatial.copy_(torch.logit(z.new_tensor((params["exponent"]-.25)/1.5)))
        corr=n.correction(z,prepare(z,quant))
    elif method in ("featup","featup-gated"):
        if z.shape[1]!=19:raise ValueError("The fitted FeatUp comparison used 19 road classes")
        net=bilateral(seed,z.device,s3_assets);guide=quant.to(z)/255
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

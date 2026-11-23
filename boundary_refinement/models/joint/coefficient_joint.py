"""Scale-stable color-field decoders with the V8b non-serial semantic branch.

Color-line/atmospheric priors and low-resolution color transforms are established
ideas. This is a controlled candidate, not a claim of novel image formation.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.models.joint.partial_sharing import PartialSharing
from boundary_refinement.training.losses import partial_segmentation_loss
from boundary_refinement.models.joint.parallel_joint import rgb_loss

METHODS={"bias":("bias",False),"free":("free",False),"coupled":("coupled",False),
         "free_supervised":("free",True),"coupled_supervised":("coupled",True)}


def apply_fields(image,log_gain,bias,airlight,mode):
    size=image.shape[-2:]
    gain=F.interpolate(log_gain.float(),size,mode="bilinear",align_corners=False).exp()
    offset=F.interpolate(bias.float(),size,mode="bilinear",align_corners=False)
    if mode=="bias":unclipped=image.float()+offset
    elif mode=="free":unclipped=image.float()+(gain-1)*image.float()+offset
    elif mode=="coupled":unclipped=image.float()+(gain-1)*(image.float()-airlight.float())
    else:raise ValueError("Unknown color-field parameterization")
    return unclipped.clamp(0,1),unclipped


class CoefficientJoint(PartialSharing):
    def __init__(self,seed=42,method="bias",pretrained=True):
        if method not in METHODS:raise ValueError("Unknown V9 method")
        super().__init__(seed,"branch12",pretrained)
        self.method=method;self.color_mode,self.coefficient_supervision=METHODS[method]
        for name in ("factor_adapter","stats_project","rgb_from_semantic","semantic_from_rgb","aux_error","aux_t","aux_airlight","rgb_refine"):
            delattr(self,name)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+9_000_000)
            self.field_head=nn.Conv2d(32,4,3,padding=1)
            self.airlight_head=nn.Linear(64,3)
            nn.init.zeros_(self.field_head.weight);nn.init.zeros_(self.field_head.bias)
            nn.init.zeros_(self.airlight_head.weight);nn.init.constant_(self.airlight_head.bias,math.log(.9/.1))

    def forward(self,image):
        native_size=image.shape[-2:]
        canonical=F.interpolate(image,(384,384),mode="bilinear",align_corners=False,antialias=True) if native_size!=(384,384) else image
        source=self.encode(self.encoder,(canonical-self.mean)/self.std)
        photo=self.encode(self.rgb_encoder,(canonical-self.mean)/self.std)
        logits=self.segmenter(source["16"])
        logits=F.interpolate(logits,native_size,mode="bilinear",align_corners=False)
        latent=self.rgb_project(photo["16"])
        airlight=self.airlight_head(latent.float().mean((2,3))).sigmoid()[:,:,None,None]
        mid=self.mid_project(photo["6"])
        value=self.rgb_mid(torch.cat((F.interpolate(latent,mid.shape[-2:],mode="bilinear",align_corners=False),mid),1))
        low=self.low_project(photo["3"])
        value=self.rgb_low(torch.cat((F.interpolate(value,low.shape[-2:],mode="bilinear",align_corners=False),low),1))
        fields=F.adaptive_avg_pool2d(self.field_head(value).float(),(24,24))
        log_gain=math.log(6)*fields[:,:1].tanh();bias=4*fields[:,1:].tanh()
        restored,unclipped=apply_fields(image,log_gain,bias,airlight,self.color_mode)
        aux=dict(semantic=self.aux_semantic(source["16"]),log_gain=log_gain,bias=bias,airlight=airlight,
            clipped_fraction=((unclipped<0)|(unclipped>1)).float().mean())
        return restored,logits,aux


def training_loss(model,image,reference,mask,n,transmission,airlight):
    restored,logits,aux=model(image)
    semantic=partial_segmentation_loss(logits,mask)
    coarse=F.interpolate(aux["semantic"].float(),mask.shape[-2:],mode="bilinear",align_corners=False)
    semantic=semantic+.1*partial_segmentation_loss(coarse,mask)
    photo=rgb_loss(restored,reference)
    coefficient=photo*0
    if model.coefficient_supervision:
        t=F.interpolate(transmission,aux["log_gain"].shape[-2:],mode="bilinear",align_corners=False)
        coefficient=.1*F.l1_loss(aux["log_gain"][:n],torch.zeros_like(aux["log_gain"][:n]))
        coefficient=coefficient+.1*F.l1_loss(aux["log_gain"][2*n:],-t.log())+.1*F.l1_loss(aux["airlight"][2*n:],airlight)
        if model.color_mode=="free":
            # A stronger free-transform control gets the exact derived bias too.
            coefficient=coefficient+.025*F.l1_loss(aux["bias"][2*n:],(1-t.reciprocal())*airlight)
    loss=semantic+photo+coefficient
    return loss,dict(total=float(loss.detach()),semantic=float(semantic.detach()),rgb=float(photo.detach()),
                     coefficients=float(coefficient.detach()),clipped_fraction=float(aux["clipped_fraction"].detach()))


def load_model(path,device="cuda"):
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if packet.get("format")!="coefficient-joint-v9":raise ValueError("Wrong coefficient checkpoint")
    model=CoefficientJoint(packet["seed"],packet["method"],False)
    model.load_state_dict(packet["model_state"],strict=True)
    return model.to(device).eval(),packet

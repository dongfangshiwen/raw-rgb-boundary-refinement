"""Task-separated, semantic-conditioned photometry experiments.

Frozen semantic guidance, confidence consistency and affine field consistency
have prior art. These are registered hypotheses, not three proven inventions.
"""
from dataclasses import dataclass
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.data.corridor_data import REPO
from boundary_refinement.models.joint.geometry_visibility import GeometryVisibilityJoint
from boundary_refinement.models.joint.parallel_joint import conv_block, rgb_loss
from boundary_refinement.training.losses import partial_segmentation_loss

PRETRAINED = REPO / "pretrained/mit-b0-imagenet"


@dataclass(frozen=True)
class Spec:
    backbone: str = "segformer"
    semantic: str = "plain"
    photo: str = "lite"
    closure: str = "none"
    context: str = "aligned"
    joint: bool = False


SPECS = {
    "joint": Spec(joint=True),
    "heavy": Spec(photo="heavy"),
    "plain": Spec(),
    "visible": Spec(semantic="visible"),
    "full": Spec(semantic="visible", closure="fields"),
    "ordinary": Spec(semantic="ordinary"),
    "shuffled_semantic": Spec(semantic="shuffled"),
    "shuffled_context": Spec(context="shuffled"),
    "rgb_consistency": Spec(semantic="visible", closure="rgb"),
    "wrong_fields": Spec(semantic="visible", closure="wrong"),
    "mobile": Spec(backbone="mobile"),
}


def plain_conv(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride=stride,padding=1),nn.SiLU())


class TaskCompatibleJoint(nn.Module):
    def __init__(self,seed=42,method="plain",pretrained=True):
        super().__init__();self.seed=seed;self.method=method;self.spec=SPECS[method];self.phase="joint" if self.spec.joint else "semantic"
        # Reuse only the frozen historical factory; do not edit historical sources.
        base=GeometryVisibilityJoint(seed,"base",pretrained)
        self.register_buffer("mean",base.mean.clone());self.register_buffer("std",base.std.clone())
        if self.spec.backbone=="mobile":
            self.encoder=base.encoder
            for name in ("seg_deep","seg_mid_project","seg_mid","seg_low_project","seg_low","support_head","visibility_head","aux_semantic"):
                setattr(self,name,getattr(base,name))
            channels=(24,40,960)
        else:
            from transformers import SegformerConfig,SegformerModel
            from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
            cfg=SegformerConfig.from_pretrained(PRETRAINED,local_files_only=True)
            cfg.num_labels=2
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+11_100_000)
                if pretrained:
                    self.encoder,info=SegformerModel.from_pretrained(PRETRAINED,config=cfg,local_files_only=True,output_loading_info=True)
                    if info["missing_keys"] or info["mismatched_keys"] or any(not k.startswith("classifier.") for k in info["unexpected_keys"]):
                        raise ValueError("Unexpected MiT encoder loading mismatch: "+str(info))
                else:self.encoder=SegformerModel(cfg)
                self.seg_head=SegformerDecodeHead(cfg)
                self.seg_head.post_init()
                self.aux_semantic=nn.Conv2d(cfg.hidden_sizes[-1],2,1)
            channels=(cfg.hidden_sizes[0],cfg.hidden_sizes[1],cfg.hidden_sizes[-1])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+11_200_000)
            if self.spec.photo=="heavy":
                self.photo_encoder=copy.deepcopy(base.encoder)
                self.photo_deep=conv_block(960,64);self.photo_mid_project=nn.Conv2d(40,16,1);self.photo_low_project=nn.Conv2d(24,16,1)
                self.photo_mid=conv_block(80,64);self.photo_low=conv_block(80,32)
            else:
                self.raw_stem=nn.Sequential(plain_conv(3,24,2),plain_conv(24,32,2),plain_conv(32,32))
                self.raw_mid=nn.Sequential(plain_conv(32,48,2),plain_conv(48,48))
                self.raw_deep=nn.Sequential(plain_conv(48,64,2),plain_conv(64,64))
                self.context_low=nn.Conv2d(channels[0],16,1);self.context_mid=nn.Conv2d(channels[1],16,1);self.context_deep=nn.Conv2d(channels[2],32,1)
                self.photo_deep=conv_block(96,64);self.photo_mid=conv_block(128,64);self.photo_low=conv_block(112,32)
            # Same raw photometry bypass and same field head in EVERY V11 RGB method.
            self.field_head=nn.Conv2d(41,4,3,padding=1);self.airlight_head=nn.Linear(70,3)
            nn.init.zeros_(self.field_head.weight);nn.init.zeros_(self.field_head.bias)
            nn.init.zeros_(self.airlight_head.weight);nn.init.constant_(self.airlight_head.bias,math.log(9))
        self.set_phase(self.phase)

    def semantic_parameters(self):
        return [p for name,p in self.named_parameters() if self.is_semantic_name(name)]

    @staticmethod
    def is_semantic_name(name):
        return name.startswith(("encoder.","seg_head.","seg_deep.","seg_mid_project.","seg_mid.","seg_low_project.","seg_low.","support_head.","visibility_head.","aux_semantic."))

    def photo_parameters(self):
        return [p for name,p in self.named_parameters() if not self.is_semantic_name(name)]

    def set_phase(self,phase):
        self.phase=phase
        for name,p in self.named_parameters():p.requires_grad_(phase=="joint" or self.is_semantic_name(name)==(phase=="semantic"))
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        # Encoder MNV running stats are always fixed; the NEW SegFormer head BN
        # follows the official head during semantic/joint learning only.
        for name,layer in self.named_modules():
            if isinstance(layer,nn.modules.batchnorm._BatchNorm) and (name.startswith(("encoder.","photo_encoder.")) or self.phase=="photo"):
                layer.eval()
        if self.phase=="photo":
            self.encoder.eval()
            if hasattr(self,"seg_head"):self.seg_head.eval()
        return self

    @staticmethod
    def mobile_features(encoder,image):
        result={}
        for key,layer in encoder.items():
            image=layer(image)
            if key in ("3","6","16"):result[key]=image
        return (result["3"],result["6"],result["16"])

    def semantic(self,canonical):
        normalized=(canonical-self.mean)/self.std
        if self.spec.backbone=="mobile":
            low,mid,deep=self.mobile_features(self.encoder,normalized)
            value=self.seg_deep(deep);m=self.seg_mid_project(mid)
            value=self.seg_mid(torch.cat((F.interpolate(value,m.shape[-2:],mode="bilinear",align_corners=False),m),1));l=self.seg_low_project(low)
            value=self.seg_low(torch.cat((F.interpolate(value,l.shape[-2:],mode="bilinear",align_corners=False),l),1))
            delta=self.support_head(value).float()+self.visibility_head(value).float()-4
            logits=torch.cat((torch.zeros_like(delta),delta),1);features=(low,mid,deep)
        else:
            values=self.encoder(normalized,output_hidden_states=True,return_dict=True).hidden_states
            logits=self.seg_head(values);features=(values[0],values[1],values[-1])
        return logits,features,self.aux_semantic(features[-1])

    def photograph(self,canonical,features):
        stats=torch.cat((canonical.float().mean((2,3)),(canonical.float().var((2,3),unbiased=False)+1e-6).sqrt()),1)
        if self.spec.photo=="heavy":
            low,mid,deep=self.mobile_features(self.photo_encoder,(canonical-self.mean)/self.std)
            deep=self.photo_deep(deep);m=self.photo_mid_project(mid)
            value=self.photo_mid(torch.cat((F.interpolate(deep,m.shape[-2:],mode="bilinear",align_corners=False),m),1));l=self.photo_low_project(low)
            value=self.photo_low(torch.cat((F.interpolate(value,l.shape[-2:],mode="bilinear",align_corners=False),l),1))
        else:
            low=self.raw_stem(canonical);mid=self.raw_mid(low);rawdeep=self.raw_deep(mid)
            ctx=list(features)
            if self.spec.context=="shuffled":ctx=[torch.roll(f,(f.shape[-2]//2,f.shape[-1]//2),(-2,-1)) for f in ctx]
            cl,cm,cd=(layer(f) for layer,f in zip((self.context_low,self.context_mid,self.context_deep),ctx))
            deep=self.photo_deep(torch.cat((rawdeep,F.interpolate(cd,rawdeep.shape[-2:],mode="bilinear",align_corners=False)),1))
            value=self.photo_mid(torch.cat((mid,F.interpolate(cm,mid.shape[-2:],mode="bilinear",align_corners=False),F.interpolate(deep,mid.shape[-2:],mode="bilinear",align_corners=False)),1))
            value=self.photo_low(torch.cat((low,F.interpolate(cl,low.shape[-2:],mode="bilinear",align_corners=False),F.interpolate(value,low.shape[-2:],mode="bilinear",align_corners=False)),1))
        raw=F.interpolate(canonical.float(),value.shape[-2:],mode="bilinear",align_corners=False)
        fields=F.adaptive_avg_pool2d(self.field_head(torch.cat((value,raw,stats[:,:,None,None].expand(-1,-1,*value.shape[-2:])),1)),(24,24)).float()
        airlight=self.airlight_head(torch.cat((deep.float().mean((2,3)),stats),1)).sigmoid()[:,:,None,None]
        return math.log(6)*fields[:,:1].tanh(),4*fields[:,1:].tanh(),airlight

    def forward(self,image):
        native=image.shape[-2:];canonical=image if native==(384,384) else F.interpolate(image,(384,384),mode="bilinear",align_corners=False,antialias=True)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.phase!="photo"):
            logits,features,coarse=self.semantic(canonical)
        if self.phase=="semantic":
            restored=image;g=torch.zeros(len(image),1,24,24,device=image.device);b=g.expand(-1,3,-1,-1);a=torch.full((len(image),3,1,1),.9,device=image.device)
        else:
            g,b,a=self.photograph(canonical,features)
            gain=F.interpolate(g,native,mode="bilinear",align_corners=False).exp();bias=F.interpolate(b,native,mode="bilinear",align_corners=False)
            restored=(image.float()*gain+bias).clamp(0,1)
        return restored,F.interpolate(logits.float(),native,mode="bilinear",align_corners=False),dict(semantic=coarse,log_gain=g,bias=b,airlight=a)


def binary_consistency(logits,batch,mode):
    if mode=="plain":return logits.sum()*0
    n=batch["scene_count"];size=logits.shape[-2:];known=batch["support_mask"]!=255
    outside=~batch["occlusion"]
    teacher=logits[:n].detach().softmax(1);target=batch["support_mask"]
    confidence,category=teacher.max(1)
    trusted=known if mode=="ordinary" else known&(confidence>=.7)&(category==target)
    if mode=="shuffled":teacher=torch.roll(teacher,(size[0]//2,size[1]//2),(-2,-1))
    terms=[]
    for view in (2,3):
        loss=F.kl_div(logits[view*n:(view+1)*n].log_softmax(1),teacher,reduction="none").sum(1)
        valid=trusted&(outside if view==3 else torch.ones_like(outside))
        if mode=="ordinary":terms.append(loss[valid].mean() if bool(valid.any()) else loss.sum()*0)
        else:
            classes=[loss[valid&(target==label)].mean() for label in (0,1) if bool((valid&(target==label)).any())]
            terms.append(sum(classes)/len(classes) if classes else loss.sum()*0)
    return .1*sum(terms)/len(terms)


def semantic_loss(logits,aux,batch,mode):
    mask=batch["visible_mask"]
    coarse=F.interpolate(aux["semantic"].float(),mask.shape[-2:],mode="bilinear",align_corners=False)
    supervised=partial_segmentation_loss(logits,mask)+.1*partial_segmentation_loss(coarse,mask)
    consistency=binary_consistency(logits,batch,mode)
    return supervised+consistency,dict(supervised=float(supervised.detach()),consistency=float(consistency.detach()))


def photo_loss(restored,aux,batch,mode):
    n=batch["scene_count"];g=aux["log_gain"];b=aux["bias"];a=aux["airlight"]
    t=F.interpolate(batch["transmission"],g.shape[-2:],mode="bilinear",align_corners=False)
    coefficient=.1*g[:n].abs().mean()
    for view in (2,3):
        q=slice(view*n,(view+1)*n)
        coefficient=coefficient+.05*F.l1_loss(g[q],-t.log())+.05*F.l1_loss(a[q],batch["airlight"])+.0125*F.l1_loss(b[q],(1-t.reciprocal())*batch["airlight"])
    loss=rgb_loss(restored,batch["reference"])+coefficient;closure=loss*0
    if mode in ("fields","wrong"):
        tc=t if mode=="fields" else t.roll(1,0)
        ac=batch["airlight"] if mode=="fields" else batch["airlight"].roll(1,0)
        closure=F.l1_loss(g[2*n:3*n]+tc.log(),g[:n].detach())
        closure=closure+F.l1_loss(b[2*n:3*n]+g[2*n:3*n].exp()*(1-tc)*ac,b[:n].detach())
        # Only coefficient cells whose surrounding 32 canonical pixels are free
        # from pasted pixels are used for occlusion invariance.
        padded=F.max_pool2d(batch["occlusion"][:,None].float(),65,1,32)
        valid=F.adaptive_max_pool2d(padded,g.shape[-2:])<.5
        for field in (g,b):
            error=(field[3*n:4*n]-field[2*n:3*n].detach()).abs();v=valid.expand_as(error)
            closure=closure+(error[v].mean() if bool(v.any()) else error.sum()*0)
        closure=.05*closure
    elif mode=="rgb":
        error=(restored[2*n:3*n]-restored[:n].detach()).abs().mean()
        valid=(~batch["occlusion"])[:,None].expand(-1,3,-1,-1)
        error2=(restored[3*n:4*n]-restored[2*n:3*n].detach()).abs()
        closure=.05*(error+(error2[valid].mean() if bool(valid.any()) else error2.sum()*0))
    return loss+closure,dict(rgb=float(loss.detach()),closure=float(closure.detach()))


def load_model(path,device="cuda"):
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if packet.get("format")!="task-compatible-v11":raise ValueError("Not V11 weights")
    model=TaskCompatibleJoint(packet["seed"],packet["method"],False);model.load_state_dict(packet["model_state"],strict=True)
    model.set_phase(packet["phase"]);return model.to(device).eval(),packet

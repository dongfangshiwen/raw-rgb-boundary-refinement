"""Unproven high-resolution/structural hypotheses, with image-only dual output.

U-Nets, local normalized correlation, self-similarity and hard-negative mining
are established ideas. This module implements precise combinations to TEST;
it does not establish three original contributions. Historical sources stay frozen.
"""
from boundary_refinement.paths import WORKSPACE
import ast
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.models.joint.projective_joint import SIZE
from boundary_refinement.models.joint.tangent_detail import TangentDetailJoint, segmentation_loss
from boundary_refinement.models.joint.parallel_joint import rgb_loss

AUTHOR = (WORKSPACE / "benchmark_data" / 'published-models-v14/author-nafnet')
OFFSETS = tuple((dy*r,dx*r) for r in (1,2,4,8,12,16) for dy,dx in ((0,1),(1,0),(1,1),(1,-1)))


@dataclass(frozen=True)
class Spec:
    local: str = 'none'
    mining: str = 'none'
    photo: str = 'pyramid'


SPECS = {
    'hires': Spec(),
    'local_raw': Spec(local='raw'),
    'norm_patch': Spec(local='normalized'),
    'local_corr': Spec(local='correlation'),
    'corr_shifted': Spec(local='shifted'),
    'mining': Spec(local='correlation',mining='structure'),
    'ohem': Spec(local='correlation',mining='confidence'),
    'mining_shifted': Spec(local='correlation',mining='shifted'),
    'nafnet_pair': Spec(photo='nafnet'),
    'dehazeformer_pair': Spec(photo='dehazeformer'),
}


def pool(x, kernel=9):
    radius=kernel//2
    return F.avg_pool2d(F.pad(x,(radius,)*4,mode='replicate'),kernel,1)


def shift(x,dy,dx):
    radius=max(abs(dy),abs(dx))
    if radius==0:return x
    padded=F.pad(x,(radius,)*4,mode='replicate')
    h,w=x.shape[-2:]
    return padded[...,radius+dy:radius+dy+h,radius+dx:radius+dx+w]


def local_descriptor(image,mode='correlation',size=None):
    """24 local patch lags; approximate affine-fog invariance only.

    Constant positive affine transformations cancel in centered NCC except the
    numerical floor. Spatially varying transmission, clipping and noise break
    this assumption. Off-image samples replicate borders, never wrap around.
    """
    with torch.autocast(image.device.type,enabled=False):
        x=image.float()
        if size is None:size=(max(1,x.shape[-2]//2),max(1,x.shape[-1]//2))
        x=F.interpolate(x,size,mode='bilinear',align_corners=False,antialias=True)
        gray=(x*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        centered=gray-gray.mean((2,3),keepdim=True)
        mean=pool(centered);variance=(pool(centered.square())-mean.square()).clamp_min(0)
        ridge=centered.square().mean((2,3),keepdim=True)*.001+1e-8
        channels=[]
        for dy,dx in OFFSETS:
            other=shift(centered,dy,dx)
            if mode=='raw':value=shift(gray,dy,dx)
            elif mode=='normalized':value=((other-mean)/(variance+ridge).sqrt()).clamp(-5,5)/5
            else:
                other_mean=pool(other);other_variance=(pool(other.square())-other_mean.square()).clamp_min(0)
                covariance=pool(centered*other)-mean*other_mean
                value=(covariance/((variance+ridge)*(other_variance+ridge)).sqrt()).clamp(-1,1)
            channels.append(value)
        result=torch.cat(channels,1)
        if mode=='shifted':result=torch.roll(result,(size[0]//2,size[1]//2),(-2,-1))
        return result


def block(a,b,stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride=stride,padding=1),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1,groups=b),nn.SiLU(),nn.Conv2d(b,b,1))


def resize(x,size):
    return F.interpolate(x,size,mode='bilinear',align_corners=False)


class PyramidRGB(nn.Module):
    """Dense native-grid RGB decoder, replacing the 24x24 affine field model."""
    def __init__(self):
        super().__init__()
        self.raw0=block(3,8);self.raw1=block(8,16,2);self.raw2=block(16,24,2);self.raw3=block(24,32,2)
        self.context0=nn.Conv2d(32,8,1);self.context1=nn.Conv2d(64,16,1);self.context2=nn.Conv2d(256,16,1)
        self.decode2=block(48,32);self.decode1=block(80,24);self.decode0=block(40,16)
        self.refine=block(24,8);self.output=nn.Conv2d(8,3,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,image,features):
        low,mid,deep=features
        r0=self.raw0(image);r1=self.raw1(r0);r2=self.raw2(r1);r3=self.raw3(r2)
        x=self.decode2(torch.cat((r3,resize(self.context2(deep),r3.shape[-2:])),1))
        # 24 raw + 32 decoded + 16 mid + 8 low context, at native quarter resolution.
        x=self.decode1(torch.cat((r2,resize(x,r2.shape[-2:]),resize(self.context1(mid),r2.shape[-2:]),
                                 resize(self.context0(low),r2.shape[-2:])),1))
        # 16 raw + 24 decoded, at native half resolution.
        x=self.decode0(torch.cat((r1,resize(x,r1.shape[-2:])),1))
        x=self.refine(torch.cat((r0,resize(x,r0.shape[-2:])),1))
        return (image.float()+self.output(x).float()).clamp(0,1)


def author_nafnet():
    """Compile only inspected author classes, bypassing unrelated BasicSR imports.

    Author block/network and normalization class bodies are unchanged. The local
    testing-conversion class and framework-wide utility imports are not used.
    """
    namespace=dict(torch=torch,nn=nn,F=F)
    for filename,names in (('arch_util.py',{'LayerNormFunction','LayerNorm2d'}),
                           ('NAFNet_arch.py',{'SimpleGate','NAFBlock','NAFNet'})):
        tree=ast.parse((AUTHOR/filename).read_text(encoding='utf-8'))
        selected=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
        if {node.name for node in selected}!=names:raise ValueError('Author source structure changed')
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(AUTHOR/filename),'exec'),namespace)
    return namespace['NAFNet'](width=16,middle_blk_num=1,enc_blk_nums=[1,1,1,28],dec_blk_nums=[1,1,1,1])


class StructureJoint(TangentDetailJoint):
    def __init__(self,seed=42,method='hires',pretrained=True):
        if method not in SPECS:raise ValueError(method)
        super().__init__(seed,'baseline',pretrained)
        self.method=method;self.design=SPECS[method]
        self.remove_photo()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+14_100_000)
            if self.design.photo=='pyramid':self.rgb_decoder=PyramidRGB()
            elif self.design.photo=='nafnet':self.rgb_decoder=author_nafnet()
            else:
                from boundary_refinement.baselines.published_runtime import dehazeformer_t
                self.rgb_decoder=dehazeformer_t()
        if self.design.local!='none':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+14_200_000)
                self.local_context=nn.Conv2d(256,32,1)
                self.local_stem=block(56,32)
                self.local_output=nn.Conv2d(32,2,1)
                nn.init.zeros_(self.local_output.weight);nn.init.zeros_(self.local_output.bias)
        self.set_phase('joint')

    @staticmethod
    def is_semantic_name(name):
        return TangentDetailJoint.is_semantic_name(name) or name.startswith(('local_context.','local_stem.','local_output.'))

    def forward(self,image):
        native=image.shape[-2:]
        canonical=image if native==(SIZE,SIZE) else F.interpolate(image,(SIZE,SIZE),mode='bilinear',align_corners=False,antialias=True)
        features=self.encoder((canonical-self.mean)/self.std,output_hidden_states=True,return_dict=True).hidden_states
        parts=[]
        for value,projection in zip(features,self.seg_head.linear_c):
            z=projection(value).permute(0,2,1).reshape(value.shape[0],-1,*value.shape[-2:])
            parts.append(resize(z,features[0].shape[-2:]))
        fused=self.seg_head.linear_fuse(torch.cat(parts[::-1],1))
        fused=self.seg_head.activation(self.seg_head.batch_norm(fused))
        logits=self.seg_head.classifier(self.seg_head.dropout(fused)).float()
        correction=None
        if self.design.local!='none':
            descriptor=local_descriptor(canonical,self.design.local)
            local=self.local_stem(torch.cat((descriptor,resize(self.local_context(fused),descriptor.shape[-2:])),1))
            correction=self.local_output(local).float()
        if self.design.photo=='pyramid':rgb=self.rgb_decoder(image,(features[0],features[1],features[-1]))
        elif self.design.photo=='nafnet':rgb=self.rgb_decoder(image).float().clamp(0,1)
        else:rgb=self.rgb_decoder(image*2-1).float().clamp(-1,1)*.5+.5
        logits=resize(logits,native)
        if correction is not None:logits=logits+resize(correction,native)
        return rgb,logits,dict(semantic=self.aux_semantic(features[-1]))


def mining_loss(logits,target,reference,mode):
    """Public-complete masks only. Unknown pixels can never become negatives.

    This is a matched hard-negative loss family, not a novel OHEM claim. Positive
    and negative counts are identical in the structure/confidence controls.
    Reference RGB is used for train-only selection and never model inference.
    """
    target=F.interpolate(target[:,None].float(),(64,64),mode='nearest')[:,0].long()
    delta=resize(logits.float(),(64,64))[:,1]-resize(logits.float(),(64,64))[:,0]
    with torch.no_grad():
        descriptor=local_descriptor(reference,'correlation',size=(64,64)) if mode!='confidence' else None
        if mode=='shifted':descriptor=torch.roll(descriptor,(32,32),(-2,-1))
    terms=[]
    for i in range(len(target)):
        fg=target[i]==1;bg=target[i]==0
        if not bool(fg.any()) or not bool(bg.any()):continue
        if mode=='confidence':score=delta[i].detach()
        else:
            feature=F.normalize(descriptor[i],dim=0,eps=1e-6)
            prototype=F.normalize(feature[:,fg].mean(1),dim=0,eps=1e-6)
            score=(feature*prototype[:,None,None]).sum(0)
        negatives=delta[i][bg];count=min(len(negatives),max(32,min(256,len(negatives)//8)))
        indices=score[bg].topk(count).indices
        terms.append((F.softplus(-delta[i][fg]).mean()+F.softplus(negatives[indices]).mean())/2)
    return .1*torch.stack(terms).mean() if terms else logits.sum()*0


def losses(model,rgb,logits,aux,batch):
    target=batch['visible_mask'];s=segmentation_loss(logits,target)
    s=s+.1*segmentation_loss(resize(aux['semantic'].float(),target.shape[-2:]),target)
    hard=s*0
    if model.design.mining!='none' and batch.get('public_complete',False):
        hard=mining_loss(logits,target,batch['reference'],model.design.mining)
    s=s+hard;r=rgb_loss(rgb,batch['reference'])
    return s,r,dict(semantic=float(s.detach()),RGB=float(r.detach()),mining=float(hard.detach()))


def build_model(seed,method,pretrained=True):
    if not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch required')
    model=StructureJoint(seed,method,pretrained).cuda()
    # Set AFTER every CPU constructor and device transfer: no fork_rng CUDA leak.
    torch.cuda.manual_seed_all(seed+13_020_000)
    return model


def load_model(path,device='cuda'):
    packet=torch.load(path,map_location='cpu',weights_only=True)
    if packet.get('format')!='structure-joint-v14':raise ValueError('Not V14 weights')
    model=StructureJoint(packet['seed'],packet['method'],False)
    model.load_state_dict(packet['model_state'],strict=True)
    return model.to(device).eval(),packet

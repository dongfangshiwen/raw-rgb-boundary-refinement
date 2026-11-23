"""Decision-space-constrained semantic guidance of an author NAFNet decoder.

The new claim to TEST is utility of classifier-aware feature-gradient routing.
Orthogonal/null-space gradient methods and semantic restoration guidance are
prior art. A feature-space property is NOT an Adam/encoder-update guarantee.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from scripts.task_compatible_v11 import PRETRAINED
from scripts.structure_joint_v14 import author_nafnet
from scripts.published_runtime_v12 import dehazeformer_t


@dataclass(frozen=True)
class Spec:
    context:bool=True
    route:str='plain'
    shifted:bool=False
    parameter_pcgrad:bool=False
    author:str='nafnet'


SPECS={
    'independent':Spec(context=False),
    'detached':Spec(route='detach'),
    'shared':Spec(),
    'decision_null':Spec(route='null'),
    'random_null':Spec(route='random'),
    'margin_null':Spec(route='margin'),
    'shifted_null':Spec(route='null',shifted=True),
    'pcgrad':Spec(parameter_pcgrad=True),
    'dehazeformer_pair':Spec(context=False,author='dehazeformer'),
}


def decision_basis(weight):
    """Orthonormal rows span classifier log-odds directions, rank<=C-1."""
    w=weight.detach().float().flatten(1);w=w-w.mean(0,keepdim=True)
    _,s,v=torch.linalg.svd(w,full_matrices=False)
    rank=int((s>max(float(s.max())*1e-5,1e-7)).sum())
    return v[:rank]


def remove_rowspace(gradient,basis):
    g=gradient.float()
    if len(basis)==0:return g
    return g-torch.einsum('kc,bkhw->bchw',basis,torch.einsum('kc,bchw->bkhw',basis,g))


class NullRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,basis):
        ctx.save_for_backward(basis);return value.view_as(value)

    @staticmethod
    def backward(ctx,gradient):
        return remove_rowspace(gradient,ctx.saved_tensors[0]),None


class MarginRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,direction):
        ctx.save_for_backward(direction);return value.view_as(value)

    @staticmethod
    def backward(ctx,gradient):
        direction=ctx.saved_tensors[0];g=gradient.float()
        return g-(g*direction).sum(1,keepdim=True)*direction,None


class DecisionJoint(nn.Module):
    def __init__(self,seed=42,method='decision_null',classes=11,pretrained=True):
        super().__init__()
        if method not in SPECS or classes not in (2,11):raise ValueError('Explicit method/ontology required')
        from transformers import SegformerConfig,SegformerModel
        from transformers.models.segformer.modeling_segformer import SegformerDecodeHead
        self.seed=seed;self.method=method;self.classes=classes;self.spec=SPECS[method]
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        cfg=SegformerConfig.from_pretrained(PRETRAINED,local_files_only=True);cfg.num_labels=classes
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+15_100_000)
            if pretrained:
                self.encoder,info=SegformerModel.from_pretrained(PRETRAINED,config=cfg,local_files_only=True,output_loading_info=True)
                if info['missing_keys'] or info['mismatched_keys'] or any(not k.startswith('classifier.') for k in info['unexpected_keys']):raise ValueError('Unexpected ImageNet loading keys')
            else:self.encoder=SegformerModel(cfg)
            self.seg_head=SegformerDecodeHead(cfg);self.seg_head.post_init()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+15_200_000)
            self.rgb_net=author_nafnet() if self.spec.author=='nafnet' else dehazeformer_t()
        if self.spec.context:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+15_300_000)
                self.photo_context=nn.ModuleList([nn.Conv2d(256,c,1) for c in (16,32,64,128)])
                for layer in self.photo_context:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+15_400_000)
            q,_=torch.linalg.qr(torch.randn(256,classes-1),mode='reduced')
        self.register_buffer('random_basis',q.T.contiguous())

    @staticmethod
    def semantic_name(name):return name.startswith(('encoder.','seg_head.'))

    def semantic_parameters(self):return [p for n,p in self.named_parameters() if self.semantic_name(n)]

    def photo_parameters(self):return [p for n,p in self.named_parameters() if not self.semantic_name(n)]

    def shared_parameters(self):
        # Classifier weights define the semantic decision but are NOT RGB inputs.
        return [p for n,p in self.named_parameters() if self.semantic_name(n) and not n.startswith('seg_head.classifier.') and p.requires_grad]

    def semantic(self,image):
        values=self.encoder((image-self.mean)/self.std,output_hidden_states=True,return_dict=True).hidden_states
        parts=[]
        for value,projection in zip(values,self.seg_head.linear_c):
            z=projection(value).permute(0,2,1).reshape(value.shape[0],-1,*value.shape[-2:])
            parts.append(F.interpolate(z,values[0].shape[-2:],mode='bilinear',align_corners=False))
        fused=self.seg_head.linear_fuse(torch.cat(parts[::-1],1))
        fused=self.seg_head.dropout(self.seg_head.activation(self.seg_head.batch_norm(fused))).float()
        # The classifier and the explicit routing boundary both use FP32. This
        # makes the local log-odds identity meaningful under BF16 trunk training.
        with torch.autocast(image.device.type,enabled=False):logits=self.seg_head.classifier(fused)
        return fused,logits

    def routed(self,fused,logits):
        mode=self.spec.route
        if mode=='detach':return fused.detach()
        if not self.training or not torch.is_grad_enabled() or mode=='plain':return fused
        with torch.autocast(fused.device.type,enabled=False):
            if mode in ('null','random'):
                basis=decision_basis(self.seg_head.classifier.weight) if mode=='null' else self.random_basis
                return NullRoute.apply(fused,basis)
            if mode=='margin':
                index=logits.detach().topk(2,dim=1).indices
                w=self.seg_head.classifier.weight.detach().float().flatten(1)
                direction=(w[index[:,0]]-w[index[:,1]]).permute(0,3,1,2)
                direction=F.normalize(direction,dim=1,eps=1e-8)
                return MarginRoute.apply(fused,direction)
        raise ValueError('Unknown routing mode')

    def restoration(self,image,context):
        net=self.rgb_net
        if not self.spec.context:
            return net(image).float().clamp(0,1) if self.spec.author=='nafnet' else net(image*2-1).float().clamp(-1,1)*.5+.5
        # Inspected author forward, with four explicit additive semantic inputs.
        # Author block/normalization bodies and all author parameters unchanged.
        height,width=image.shape[-2:];inp=net.check_image_size(image);x=net.intro(inp);skips=[]
        for encoder,down,projection in zip(net.encoders,net.downs,self.photo_context):
            value=F.interpolate(projection(context),x.shape[-2:],mode='bilinear',align_corners=False)
            x=encoder(x+value);skips.append(x);x=down(x)
        x=net.middle_blks(x)
        for decoder,up,skip in zip(net.decoders,net.ups,skips[::-1]):x=decoder(up(x)+skip)
        return (net.ending(x)+inp)[:,:,:height,:width].float().clamp(0,1)

    def forward(self,image):
        fused,logits=self.semantic(image);context=self.routed(fused,logits)
        if self.spec.shifted:context=torch.roll(context,(context.shape[-2]//2,context.shape[-1]//2),(-2,-1))
        rgb=self.restoration(image,context)
        return rgb,F.interpolate(logits,image.shape[-2:],mode='bilinear',align_corners=False),dict(feature=fused,low_logits=logits)


def build_model(seed,method,classes=11,pretrained=True):
    if not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch required')
    model=DecisionJoint(seed,method,classes,pretrained).cuda();torch.cuda.manual_seed_all(seed+15_020_000)
    return model


def load_model(path,device='cuda'):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['format']!='decision-joint-v15':raise ValueError('Wrong checkpoint format')
    model=DecisionJoint(value['seed'],value['method'],value['classes'],False)
    model.load_state_dict(value['model_state'],strict=True)
    return model.to(device).eval(),value

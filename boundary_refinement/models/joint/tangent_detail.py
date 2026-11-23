"""Paired-render gradient routing and colour-feasible detail: unproven hypotheses.

The paired directions are finite secants, not analytically computed manifold
tangents or a uniquely identified physical haze subspace. Routing is an explicit
surrogate gradient/preconditioner, NOT the derivative of
an identity function. Orthogonality holds only in activation coordinates, not
after the encoder Jacobian, Adam or clipping. Existing gradient surgery and
colour-preserving mappings are prior art, not originality claims here.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.models.joint.projective_joint import ProjectiveJoint,SIZE
from boundary_refinement.models.joint.parallel_joint import rgb_loss


@dataclass(frozen=True)
class Spec:
    route: str='none'
    detail: str='none'
    consistency: bool=False
    pcgrad: bool=False
    detach_rgb: bool=False


SPECS={
    'baseline':Spec(),
    'tangent':Spec(route='paired'),
    'permuted':Spec(route='permuted'),
    'detail':Spec(detail='feasible'),
    'neutral_clip':Spec(detail='neutral_clip'),
    'rgb_detail':Spec(detail='rgb'),
    'full':Spec(route='paired',detail='feasible'),
    'consistency':Spec(consistency=True),
    'pcgrad':Spec(pcgrad=True),
    'detach_rgb':Spec(detach_rgb=True),
}


@torch.no_grad()
def tangent_basis(feature,permutation=None):
    if len(feature)%4:raise ValueError('Routing requires four ordered paired views in training')
    n=len(feature)//4;f=feature.detach().float()
    d1=f[n:2*n]-f[:n];d2=f[2*n:3*n]-f[:n]
    norm1=d1.square().sum(1,keepdim=True).sqrt()
    u=torch.where(norm1>1e-5,d1/norm1.clamp_min(1e-5),torch.zeros_like(d1))
    residual=d2-(d2*u).sum(1,keepdim=True)*u
    norm2=residual.square().sum(1,keepdim=True).sqrt()
    # Avoid treating FP32 roundoff in nearly collinear directions as rank two.
    reliable=norm2>torch.maximum(torch.full_like(norm2,1e-5),1e-4*d2.square().sum(1,keepdim=True).sqrt())
    v=torch.where(reliable,residual/norm2.clamp_min(1e-5),torch.zeros_like(residual))
    if permutation is not None:u=u[:,permutation];v=v[:,permutation]
    return u,v


def project_gradient(gradient,u,v):
    n=len(u)
    if len(gradient)!=4*n:raise ValueError('Gradient/view count mismatch')
    shape=(4,n,*gradient.shape[1:]);g=gradient.float().reshape(shape)
    return (((g*u[None]).sum(2,keepdim=True)*u[None])+((g*v[None]).sum(2,keepdim=True)*v[None])).reshape_as(gradient)


class RoutedIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,u,v,semantic):
        ctx.save_for_backward(u,v);ctx.semantic=semantic
        return value.view_as(value)

    @staticmethod
    def backward(ctx,gradient):
        u,v=ctx.saved_tensors;parallel=project_gradient(gradient,u,v)
        result=gradient.float()-parallel if ctx.semantic else parallel
        return result.to(gradient.dtype),None,None,None


def feasible_neutral(coarse,raw):
    """Zero is an exact identity with a live subgradient; preserve RGB differences.

    Real-arithmetic range guarantee; ordinary FP rounding tolerance applies.
    This preserves opponent differences of coarse, not ground-truth Lab hue.
    """
    r=raw.float().mean(1,keepdim=True).tanh()
    lower=torch.maximum(torch.full_like(r,-.2),-coarse.float().amin(1,keepdim=True))
    upper=torch.minimum(torch.full_like(r,.2),1-coarse.float().amax(1,keepdim=True))
    delta=.5*(upper-lower)*r+.5*(upper+lower)*r.abs()
    return coarse.float()+delta


class DetailHead(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.stem=nn.Conv2d(9,width,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=d,dilation=d,groups=width),nn.SiLU(),nn.Conv2d(width,width,1)) for d in (1,2,4)])
        self.output=nn.Conv2d(width,3,1,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,image,coarse):
        x=F.silu(self.stem(torch.cat((image.float(),coarse.float(),image.float()-coarse.float()),1)))
        for block in self.blocks:x=x+block(x)
        raw=self.output(x).float()
        # Replicate boundary prevents spurious zero-padding detail near edges.
        return raw-F.avg_pool2d(F.pad(raw,(2,2,2,2),mode='replicate'),5,stride=1)


def segmentation_loss(logits,target):
    valid=target!=255;k=logits.shape[1]
    if not bool(valid.any()):raise ValueError('No supervised pixels')
    if bool(((target[valid]<0)|(target[valid]>=k)).any()):raise ValueError('Label outside task ontology')
    ce=F.cross_entropy(logits,target,ignore_index=255,reduction='sum')/valid.sum()
    p=logits.float().softmax(1)*valid[:,None]
    y=F.one_hot(target.clamp(0,k-1),k).permute(0,3,1,2).float()*valid[:,None]
    return ce+1-((2*(p*y).sum((0,2,3))+1e-6)/((p+y).sum((0,2,3))+1e-6)).mean()


class TangentDetailJoint(ProjectiveJoint):
    def __init__(self,seed=42,method='baseline',pretrained=True,num_classes=2):
        if method not in SPECS:raise ValueError(method)
        super().__init__(seed,'dense',pretrained)
        self.method=method;self.experiment=SPECS[method];self.num_classes=num_classes
        if num_classes!=2:
            if num_classes<2:raise ValueError('At least two semantic classes')
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+13_010_000)
                self.seg_head.classifier=nn.Conv2d(256,num_classes,1)
                nn.init.normal_(self.seg_head.classifier.weight,std=.02);nn.init.zeros_(self.seg_head.classifier.bias)
                self.aux_semantic=nn.Conv2d(256,num_classes,1)
        if self.experiment.detail!='none':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+13_020_000);self.detail_head=DetailHead()
        # Training-control randomness cannot consume augmentation or CUDA RNG.
        generator=torch.Generator().manual_seed(seed+13_030_000)
        for i,c in enumerate((32,64,160,256)):
            self.register_buffer('tangent_permutation_'+str(i),torch.randperm(c,generator=generator),persistent=False)
        self.set_phase('joint')

    def split_features(self,features):
        if not self.training:return list(features),list(features),None
        semantic=[];photo=[];consistency=[]
        for i,f in enumerate(features):
            if self.experiment.route!='none':
                perm=getattr(self,'tangent_permutation_'+str(i)) if self.experiment.route=='permuted' else None
                u,v=tangent_basis(f,perm)
                semantic.append(RoutedIdentity.apply(f,u,v,True));photo.append(RoutedIdentity.apply(f,u,v,False))
            else:semantic.append(f);photo.append(f.detach() if self.experiment.detach_rgb else f)
            if self.experiment.consistency:
                n=len(f)//4
                if 4*n!=len(f):raise ValueError('Consistency requires paired views')
                z=F.normalize(f.float(),dim=1);teacher=z[:n].detach()
                consistency.append(((z[n:2*n]-teacher).square().sum(1).mean()+(z[2*n:3*n]-teacher).square().sum(1).mean())/2)
        return semantic,photo,torch.stack(consistency).mean() if consistency else None

    def forward(self,image):
        native=image.shape[-2:]
        canonical=image if native==(SIZE,SIZE) else F.interpolate(image,(SIZE,SIZE),mode='bilinear',align_corners=False,antialias=True)
        features=self.encoder((canonical-self.mean)/self.std,output_hidden_states=True,return_dict=True).hidden_states
        sem,photo,consistency=self.split_features(features)
        parts=[]
        for value,projection in zip(sem,self.seg_head.linear_c):
            z=projection(value).permute(0,2,1).reshape(value.shape[0],-1,*value.shape[-2:])
            parts.append(F.interpolate(z,sem[0].shape[-2:],mode='bilinear',align_corners=False))
        fused=self.seg_head.linear_fuse(torch.cat(parts[::-1],1));fused=self.seg_head.activation(self.seg_head.batch_norm(fused))
        logits=self.seg_head.classifier(self.seg_head.dropout(fused))
        g,b,a=self.photograph(canonical,(photo[0],photo[1],photo[-1]))
        coarse=(image.float()*F.interpolate(g,native,mode='bilinear',align_corners=False).exp()+F.interpolate(b,native,mode='bilinear',align_corners=False)).clamp(0,1)
        rgb=coarse
        if self.experiment.detail!='none':
            raw=self.detail_head(image,coarse)
            if self.experiment.detail=='feasible':rgb=feasible_neutral(coarse,raw)
            elif self.experiment.detail=='neutral_clip':rgb=(coarse+.2*raw.mean(1,keepdim=True).tanh()).clamp(0,1)
            else:rgb=(coarse+.2*raw.tanh()).clamp(0,1)
        aux=dict(semantic=self.aux_semantic(sem[-1]),feature_consistency=consistency)
        return rgb,F.interpolate(logits.float(),native,mode='bilinear',align_corners=False),aux


def losses(model,rgb,logits,aux,batch):
    target=batch['visible_mask'];s=segmentation_loss(logits,target)
    coarse=F.interpolate(aux['semantic'].float(),target.shape[-2:],mode='bilinear',align_corners=False)
    s=s+.1*segmentation_loss(coarse,target)
    consistency=aux['feature_consistency']
    regularizer=0.1*consistency if consistency is not None else s*0
    s=s+regularizer;r=rgb_loss(rgb,batch['reference'])
    return s,r,dict(semantic=float(s.detach()),RGB=float(r.detach()),feature_consistency=float(regularizer.detach()))


def pcgrad_backward(model,semantic,rgb,weight=.25):
    """Two-task PCGrad on shared encoder, sum private gradients (microbatch4).

    Both projections use the ORIGINAL other-task gradient, as in Yu2020.
    Two tasks have only one counterpart, hence no random-order ambiguity.
    """
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
    a=torch.autograd.grad(semantic*weight,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(rgb*weight,params,allow_unused=True)
    shared=[i for i,(name,_) in enumerate(named) if name.startswith('encoder.') and a[i] is not None and b[i] is not None]
    if not shared:raise ValueError('PCGrad expects a connected shared encoder')
    dot=sum((a[i].float()*b[i].float()).sum() for i in shared)
    na=sum(a[i].float().square().sum() for i in shared).clamp_min(1e-30)
    nb=sum(b[i].float().square().sum() for i in shared).clamp_min(1e-30)
    alpha=dot.clamp_max(0)/nb;beta=dot.clamp_max(0)/na
    shared_set=set(shared)
    for i,(_,p) in enumerate(named):
        if a[i] is None and b[i] is None:continue
        if i in shared_set:g=a[i]+b[i]-alpha*b[i]-beta*a[i]
        else:g=(a[i] if b[i] is None else b[i] if a[i] is None else a[i]+b[i])
        if p.grad is None:p.grad=g.detach().clone()
        else:p.grad.add_(g.detach())
    return dict(conflicting=bool(dot<0),cosine=float(dot/(na*nb).sqrt()))


def load_model(path,device='cuda'):
    packet=torch.load(path,map_location='cpu',weights_only=True)
    if packet.get('format')!='tangent-detail-v13b':raise ValueError('Not V13b weights')
    model=TangentDetailJoint(packet['seed'],packet['method'],False,packet['num_classes'])
    model.load_state_dict(packet['model_state'],strict=True)
    return model.to(device).eval(),packet

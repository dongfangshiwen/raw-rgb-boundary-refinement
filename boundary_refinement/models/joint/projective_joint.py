"""Projective visible-corridor hypotheses, not pre-established novelty claims."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.models.joint.task_compatible import TaskCompatibleJoint
from boundary_refinement.models.joint.parallel_joint import rgb_loss
from boundary_refinement.training.losses import partial_segmentation_loss

SIZE=256


@dataclass(frozen=True)
class Variant:
    query:str='none'
    support_supervision:bool=False
    evidence:str='raw'
    paper:str=''
    pretrained:bool=True


VARIANTS={
    'dense':Variant(),
    'free_query':Variant(query='free'),
    'projective':Variant(query='projective'),
    'visibility':Variant(query='projective',support_supervision=True),
    'full':Variant(query='projective',support_supervision=True,evidence='restored'),
    'affine':Variant(query='affine'),
    'detached':Variant(query='projective',support_supervision=True,evidence='detached'),
    'shuffled':Variant(query='projective',support_supervision=True,evidence='shuffled'),
    'mask2former':Variant(paper='mask2former'),
    'dehazeformer_pair':Variant(paper='dehazeformer_pair'),
    'dehazeseg':Variant(paper='dehazeseg',pretrained=False),
    'scratch_dense':Variant(pretrained=False),
    'scratch_full':Variant(query='projective',support_supervision=True,evidence='restored',pretrained=False),
}


def coordinates(height,width,device,dtype=torch.float32):
    # align_corners=False pixel centres, also used for all inverse sampling.
    y=(torch.arange(height,device=device,dtype=dtype)+.5)*2/height-1
    x=(torch.arange(width,device=device,dtype=dtype)+.5)*2/width-1
    yy,xx=torch.meshgrid(y,x,indexing='ij');return torch.stack((xx,yy),-1)


def pose_parameters(raw,affine=False):
    center=.95*raw[...,:2].tanh();scale=.05+.95*raw[...,2:4].sigmoid()
    angle=math.pi*raw[...,4].tanh();p=raw[...,5:7].tanh()
    # L1 radius .85 permits pronounced perspective while keeping every
    # canonical-corner denominator at least .15, hence no projective fold.
    perspective=.85*p/p.abs().sum(-1,keepdim=True).clamp_min(1)
    if affine:perspective=perspective*0
    return center,scale,angle,perspective


def project_points(points,pose):
    center,scale,angle,perspective=pose
    cosine,sine=angle.cos(),angle.sin();u,v=points.unbind(-1)
    denominator=1+perspective[...,0,None,None]*u+perspective[...,1,None,None]*v
    x=scale[...,0,None,None]*u;y=scale[...,1,None,None]*v
    return torch.stack((center[...,0,None,None]+(cosine[...,None,None]*x-sine[...,None,None]*y)/denominator,
        center[...,1,None,None]+(sine[...,None,None]*x+cosine[...,None,None]*y)/denominator),-1)


def inverse_points(points,pose):
    center,scale,angle,perspective=pose;x=points[...,0]-center[...,0,None,None];y=points[...,1]-center[...,1,None,None]
    cosine,sine=angle.cos()[...,None,None],angle.sin()[...,None,None]
    a=(cosine*x+sine*y)/scale[...,0,None,None];b=(-sine*x+cosine*y)/scale[...,1,None,None]
    denominator=1-perspective[...,0,None,None]*a-perspective[...,1,None,None]*b
    valid=denominator>0
    safe=denominator.clamp_min(.02)
    return torch.stack((a/safe,b/safe),-1),valid


class CorridorQueries(nn.Module):
    def __init__(self,mode='projective',queries=8):
        super().__init__();self.mode=mode;self.queries=queries
        self.pixel=nn.Conv2d(256,64,1);self.memory=nn.Conv2d(64,128,1);self.position=nn.Linear(2,128)
        self.query=nn.Embedding(queries,128)
        layer=nn.TransformerDecoderLayer(128,4,512,dropout=0.,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(layer,2,norm=nn.LayerNorm(128))
        self.pose=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,7))
        nn.init.normal_(self.pose[-1].weight,std=.001);nn.init.zeros_(self.pose[-1].bias)
        template=torch.zeros(queries,7)
        for i in range(queries):
            template[i,0]=math.atanh((-.6+.4*(i%4))/.95);template[i,1]=math.atanh((-.2+.7*(i//4))/.95)
        template[:,2]=math.log((.35-.05)/(1.-.35));template[:,3]=math.log((.25-.05)/(1.-.25))
        self.register_buffer('template',template)
        self.classifier=nn.Linear(128,1);nn.init.constant_(self.classifier.bias,-1)
        self.free_mask=nn.Linear(128,64)
        self.roi=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.visibility=nn.Conv2d(16,1,1);self.evidence_class=nn.Linear(16,1)
        nn.init.constant_(self.visibility.bias,2);nn.init.zeros_(self.evidence_class.weight);nn.init.zeros_(self.evidence_class.bias)
        self.blend=nn.Parameter(torch.tensor(math.log(.2/.8)))

    def forward(self,feature,base_logits,evidence):
        b=feature.shape[0];q=self.queries;pixel=self.pixel(feature);h,w=pixel.shape[-2:]
        memory=self.memory(F.adaptive_avg_pool2d(pixel,(16,16))).flatten(2).transpose(1,2)
        position=self.position(coordinates(16,16,pixel.device).reshape(1,256,2))
        query=self.decoder(self.query.weight[None].expand(b,-1,-1),memory+position)
        raw_pose=self.pose(query).float()+self.template[None];pose=pose_parameters(raw_pose,self.mode=='affine')
        grid=project_points(coordinates(16,16,pixel.device)[None,None],pose)
        crop=F.grid_sample(evidence.float(),grid.reshape(b,q*16,16,2),align_corners=False,padding_mode='border')
        crop=crop.reshape(b,3,q,16,16).permute(0,2,1,3,4).reshape(b*q,3,16,16)
        # Local contrast normalization is established prior art, not a separate claim.
        crop=(crop-crop.mean((2,3),keepdim=True))/(crop.var((2,3),unbiased=False,keepdim=True)+.01).sqrt()
        roi=self.roi(crop.clamp(-3,3));gate=(self.classifier(query).float().squeeze(-1)+self.evidence_class(roi.mean((2,3))).float().reshape(b,q)).sigmoid()
        inverse,valid=inverse_points(coordinates(h,w,pixel.device)[None,None],pose)
        if self.mode=='free':
            support=torch.einsum('bqc,bchw->bqhw',self.free_mask(query),pixel).float().div(8).sigmoid()
        else:
            distance=(1-inverse.abs())/.035
            support=distance[...,0].sigmoid()*distance[...,1].sigmoid()*valid
        v=F.grid_sample(self.visibility(roi).float(),inverse.reshape(b*q,h,w,2),align_corners=False,padding_mode='border').reshape(b,q,h,w).sigmoid()
        weight=(gate[:,:,None,None]*support).clamp(0,1-1e-6)
        support_union=-torch.expm1(torch.log1p(-weight).sum(1,keepdim=True))
        visible=-torch.expm1(torch.log1p(-(weight*v).clamp(max=1-1e-6)).sum(1,keepdim=True))
        fixed=weight.detach();conditional=(fixed*v).sum(1,keepdim=True)/(fixed.sum(1,keepdim=True)+1e-6)
        base=base_logits.float().softmax(1)[:,1:2];blend=self.blend.sigmoid()
        probability=((1-blend)*base+blend*visible).clamp(1e-5,1-1e-5)
        gap=torch.logit(probability);logits=torch.cat((torch.zeros_like(gap),gap),1)
        return logits,dict(support=support_union,conditional_visibility=conditional,poses=raw_pose,blend=blend)


class ProjectiveJoint(TaskCompatibleJoint):
    def __init__(self,seed=42,method='full',pretrained=True):
        self.variant=VARIANTS[method]
        super().__init__(seed,'joint',pretrained and self.variant.pretrained);self.method=method
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+12_100_000)
            if self.variant.query!='none':self.corridor_head=CorridorQueries(self.variant.query)
            if self.variant.paper:
                from boundary_refinement.baselines.published_runtime import Mask2FormerHead,dehazeformer_t,dehazeseg_backbone,AuthorFrequencyNeck
                if self.variant.paper=='mask2former':
                    self.corridor_head=Mask2FormerHead([32,64,160,256]);del self.seg_head
                elif self.variant.paper=='dehazeformer_pair':
                    self.restore_author=dehazeformer_t();self.remove_photo()
                elif self.variant.paper=='dehazeseg':
                    self.restore_author=dehazeseg_backbone();self.frequency_author=AuthorFrequencyNeck()
                    self.corridor_head=Mask2FormerHead([192,384,768,768]);self.remove_photo()
                    del self.encoder,self.seg_head,self.aux_semantic;self.aux_semantic=nn.Conv2d(96,2,1)
        self.set_phase('joint')

    def remove_photo(self):
        for name in ('raw_stem','raw_mid','raw_deep','context_low','context_mid','context_deep','photo_deep','photo_mid','photo_low','field_head','airlight_head'):
            if hasattr(self,name):delattr(self,name)

    @staticmethod
    def is_semantic_name(name):return TaskCompatibleJoint.is_semantic_name(name) or name.startswith(('corridor_head.','frequency_author.'))

    def encode(self,canonical):
        features=self.encoder((canonical-self.mean)/self.std,output_hidden_states=True,return_dict=True).hidden_states
        coarse=self.aux_semantic(features[-1]);levels=(features[0],features[1],features[-1])
        if self.variant.paper=='mask2former':
            logits,auxiliary=self.corridor_head(features);return logits,levels,dict(semantic=coarse,paper_aux=auxiliary),None
        # Execute the installed SegFormer decoder once; expose its fused map.
        parts=[]
        for value,projection in zip(features,self.seg_head.linear_c):
            z=projection(value).permute(0,2,1).reshape(value.shape[0],-1,*value.shape[-2:])
            parts.append(F.interpolate(z,features[0].shape[-2:],mode='bilinear',align_corners=False))
        fused=self.seg_head.linear_fuse(torch.cat(parts[::-1],1));fused=self.seg_head.activation(self.seg_head.batch_norm(fused))
        fused=self.seg_head.dropout(fused);logits=self.seg_head.classifier(fused)
        return logits,levels,dict(semantic=coarse),fused

    def forward(self,image):
        native=image.shape[-2:];canonical=image if native==(SIZE,SIZE) else F.interpolate(image,(SIZE,SIZE),mode='bilinear',align_corners=False,antialias=True)
        if self.variant.paper=='dehazeseg':
            value=self.restore_author((canonical-self.mean)/self.std);rgb=(value[0]*self.std+self.mean).clamp(0,1)
            logits,intermediate=self.corridor_head(self.frequency_author(value[1:]));aux=dict(semantic=self.aux_semantic(value[3]),paper_aux=intermediate)
            if native!=(SIZE,SIZE):rgb=(self.restore_author.forward_dehaze((image-self.mean)/self.std)*self.std+self.mean).clamp(0,1)
        else:
            logits,features,aux,fused=self.encode(canonical)
            if self.variant.paper=='dehazeformer_pair':
                rgb=(self.restore_author(image*2-1).clamp(-1,1)*.5+.5)
                restored=rgb if native==(SIZE,SIZE) else F.interpolate(rgb,(SIZE,SIZE),mode='bilinear',align_corners=False,antialias=True)
            else:
                g,b,a=self.photograph(canonical,features)
                rgb=(image.float()*F.interpolate(g,native,mode='bilinear',align_corners=False).exp()+F.interpolate(b,native,mode='bilinear',align_corners=False)).clamp(0,1)
                restored=(canonical.float()*F.interpolate(g,(SIZE,SIZE),mode='bilinear',align_corners=False).exp()+F.interpolate(b,(SIZE,SIZE),mode='bilinear',align_corners=False)).clamp(0,1)
            if self.variant.query!='none':
                evidence=canonical if self.variant.evidence=='raw' else restored
                if self.variant.evidence=='detached':evidence=evidence.detach()
                if self.variant.evidence=='shuffled':evidence=torch.roll(evidence,(SIZE//2,SIZE//2),(-2,-1))
                logits,q=self.corridor_head(fused,logits,evidence);aux.update(q)
        return rgb,F.interpolate(logits.float(),native,mode='bilinear',align_corners=False),aux


def probability_loss(probability,target):
    p=probability.clamp(1e-5,1-1e-5);return partial_segmentation_loss(torch.cat((torch.log1p(-p),p.log()),1),target)


def losses(model,restored,logits,aux,batch):
    target=batch['visible_mask'];s=partial_segmentation_loss(logits,target)
    coarse=F.interpolate(aux['semantic'].float(),target.shape[-2:],mode='bilinear',align_corners=False)
    s=s+.1*partial_segmentation_loss(coarse,target);support=s*0;visibility=s*0
    if model.variant.support_supervision:
        n=batch['scene_count'];mask=batch['support_mask'];truth=torch.cat((mask,mask,mask,mask))
        p=F.interpolate(aux['support'],target.shape[-2:],mode='bilinear',align_corners=False)
        support=.1*probability_loss(p,truth)
        v=F.interpolate(aux['conditional_visibility'],target.shape[-2:],mode='bilinear',align_corners=False)
        expected=torch.ones_like(v);expected[3*n:]=(~batch['occlusion'])[:,None]
        valid=(truth==1)[:,None]
        with torch.autocast('cuda',enabled=False):
            visibility=.05*F.binary_cross_entropy(v.float().clamp(1e-5,1-1e-5)[valid],expected.float()[valid]) if bool(valid.any()) else v.sum()*0
        s=s+support+visibility
    if aux.get('paper_aux'):
        intermediate=[F.interpolate(x.float(),target.shape[-2:],mode='bilinear',align_corners=False) for x in aux['paper_aux']]
        s=s+.1*torch.stack([partial_segmentation_loss(x,target) for x in intermediate]).mean()
    r=rgb_loss(restored,batch['reference'])
    return s,r,dict(semantic=float(s.detach()),RGB=float(r.detach()),support=float(support.detach()),visibility=float(visibility.detach()))


def load_model(path,device='cuda'):
    packet=torch.load(path,map_location='cpu',weights_only=True)
    if packet.get('format')!='projective-joint-v12':raise ValueError('Not V12 selected weights')
    model=ProjectiveJoint(packet['seed'],packet['method'],False);model.load_state_dict(packet['model_state'],strict=True)
    return model.to(device).eval(),packet

"""CamVid11 source-only paired synthetic-fog experiment, no test loading."""
import hashlib

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from boundary_refinement.data.provenance import REPO,read,digest
from boundary_refinement.data.data_alignment import Geometry,align_image,align_mask,decode_mask,valid_pixels

DATA=REPO/'protocols/data-alignment-v15/manifest.json'
CONDITIONS=('camera','uniform035','spatial')


def prepare_data():
    manifest=read(DATA);result={}
    for part in ('train','val'):
        rows=[r for r in manifest['semantic_records'] if r['source']=='camvid' and r['partition']==part]
        images=[];masks=[];valid=[]
        for row in rows:
            if digest(row['image'])!=row['image_SHA256'] or digest(row['mask'])!=row['mask_SHA256']:raise ValueError('CamVid source changed')
            g=Geometry(**row['geometry'])
            with Image.open(row['image']) as im:images.append(torch.from_numpy(align_image(im,g)).permute(2,0,1))
            with Image.open(row['mask']) as im:masks.append(torch.from_numpy(align_mask(decode_mask(np.asarray(im),'camvid11'),g)))
            valid.append(torch.from_numpy(valid_pixels(g)))
        result[part]=dict(raw=torch.stack(images),mask=torch.stack(masks),valid=torch.stack(valid),ids=[r['id'] for r in rows])
    if len(result['train']['ids'])!=367 or len(result['val']['ids'])!=101:raise ValueError('Author source splits changed')
    result['validation_GT_counts']=torch.bincount(result['val']['mask'].long().flatten(),minlength=256)[:11].tolist()
    return result


def fog(image,transmission,airlight):return (transmission*image+(1-transmission)*airlight).clamp(0,1)


def make_batch(data,indices,generator):
    raw=data['raw'][indices].cuda().float()/255;target=data['mask'][indices].cuda().long();valid=data['valid'][indices].cuda()
    n=len(raw);flips=(torch.rand(n,generator=generator)>.5).cuda()
    raw=torch.where(flips[:,None,None,None],raw.flip(-1),raw)
    target=torch.where(flips[:,None,None],target.flip(-1),target);valid=torch.where(flips[:,None,None],valid.flip(-1),valid)
    airlight=(.75+.25*torch.rand(n,3,1,1,generator=generator)).cuda()
    t=(.25+.65*torch.rand(n,1,1,1,generator=generator)).cuda()
    field=F.interpolate(torch.rand(n,1,5,7,generator=generator).cuda(),raw.shape[-2:],mode='bicubic',align_corners=False)
    spatial=(t+.45*(field-.5)).clamp(.15,.95)
    image=torch.cat((raw,fog(raw,t,airlight),fog(raw,spatial,airlight)))
    return dict(image=image,reference=raw.repeat(3,1,1,1),target=target.repeat(3,1,1),valid=valid.repeat(3,1,1),scenes=n)


def microbatch(batch,index):
    n=batch['scenes'];indices=[index,index+n,index+2*n]
    return {k:v[indices] for k,v in batch.items() if isinstance(v,torch.Tensor)}


def fixed_fog(image,mode):
    if mode=='camera':return image
    if mode=='uniform035':return fog(image,.35,.9)
    if mode=='spatial':
        h,w=image.shape[-2:];y=torch.linspace(0,1,h,device=image.device)[None,None,:,None];x=torch.linspace(0,1,w,device=image.device)[None,None,None,:]
        t=.25+.45*(.5+.5*torch.sin(2*torch.pi*x)*torch.cos(torch.pi*y))
        a=torch.tensor([.88,.94,1.],device=image.device)[None,:,None,None]
        return fog(image,t,a)
    raise ValueError(mode)


def semantic_loss(logits,target,classes=11):
    known=target!=255
    ce=F.cross_entropy(logits,target,ignore_index=255,reduction='none')
    if not bool(known.any()):return logits.sum()*0
    safe=target.masked_fill(~known,0);one=F.one_hot(safe,classes).permute(0,3,1,2).float()*known[:,None]
    prob=logits.softmax(1)*known[:,None];dimensions=(0,2,3)
    dice=1-((2*(prob*one).sum(dimensions)+1)/(prob.sum(dimensions)+one.sum(dimensions)+1)).mean()
    return ce[known].mean()+dice


def photo_loss(rgb,reference,valid):
    v=valid[:,None].float()
    def mean(error,mask):return (error*mask).sum()/((mask.sum()*error.shape[1]).clamp_min(1))
    l1=mean((rgb-reference).abs(),v)
    p=torch.cat((rgb[:,0:1]-rgb[:,1:2],rgb[:,2:3]-rgb[:,1:2]),1)
    q=torch.cat((reference[:,0:1]-reference[:,1:2],reference[:,2:3]-reference[:,1:2]),1)
    chroma=2*mean((p-q).abs(),v)
    edge=mean(((rgb[:,:,1:]-rgb[:,:,:-1])-(reference[:,:,1:]-reference[:,:,:-1])).abs(),v[:,:,1:]*v[:,:,:-1])
    edge+=mean(((rgb[:,:,:,1:]-rgb[:,:,:,:-1])-(reference[:,:,:,1:]-reference[:,:,:,:-1])).abs(),v[:,:,:,1:]*v[:,:,:,:-1])
    return l1+.1*chroma+.05*edge


def batch_digest(batch):
    h=hashlib.sha256()
    for key in ('image','reference','target','valid'):
        h.update(key.encode());h.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

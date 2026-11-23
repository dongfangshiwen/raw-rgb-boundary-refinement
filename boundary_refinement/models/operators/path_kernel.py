"""A four-donor decoder on the manuscript lattice with an intervening-color control."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from boundary_refinement.models.operators.range_kernel import RANGES, RangeDecoder

MODES=("gaussian","ranges","endpoint","path","shifted","permuted")
SEEDS=(101,211,307)

def excess_variation(samples):
    """Travel beyond the RMS RGB chord. The quantity vanishes on a straight monotone color ramp."""
    lengths=torch.linalg.vector_norm(samples[...,1:,:]-samples[...,:-1,:],dim=-1).sum(-1)
    chord=torch.linalg.vector_norm(samples[...,-1,:]-samples[...,0,:],dim=-1)
    return (lengths-chord).clamp_min(0)/math.sqrt(3)

class Stencil:
    def __init__(self,z,guide):
        _,self.c,h,w=z.shape
        self.h,self.w=h,w
        shape=(4*h,4*w)
        if guide.shape[-2:]!=shape:
            guide=F.interpolate(guide,size=shape,mode="bilinear",align_corners=False,antialias=True)
        self.image=guide.clamp(0,1).mul(255).round()/255
        self.shifted=self.image.roll((17,37),(-2,-1))
        self.mean=F.avg_pool2d(self.image,4,4)[0].permute(1,2,0).reshape(-1,3)
        self.z=z[0].permute(1,2,0).reshape(-1,self.c)
        lab=z.argmax(1,keepdim=True).float()
        active=F.max_pool2d(lab,3,1,1)!=-F.max_pool2d(-lab,3,1,1)
        self.active=active[0,0].repeat_interleave(4,0).repeat_interleave(4,1).flatten()

    def path(self,im,y,x,yd,xd):
        h,w=im.shape[-2:]
        t=torch.linspace(0,1,6,device=y.device)
        def coords(start,end,size):
            q=start[:,None,None]+(end[:,:,None]-start[:,None,None])*t
            return 2*q/(size-1)-1
        grid=torch.stack((coords(x,xd,w),coords(y,yd,h)),-1).reshape(1,len(y),24,2)
        samples=F.grid_sample(im,grid,mode="bilinear",padding_mode="border",align_corners=True)
        samples=samples[0].permute(1,2,0).reshape(len(y),4,6,3)
        return -torch.expm1(-excess_variation(samples)/.08)

    def features(self,ids):
        y=(ids//(4*self.w)).float();x=(ids%(4*self.w)).float()
        cy=(y+.5)/4-.5;cx=(x+.5)/4-.5
        y0,x0=cy.floor(),cx.floor()
        dy,dx=cy-y0,cx-x0
        b=torch.stack(((1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx),-1)
        ys=torch.stack((y0,y0,y0+1,y0+1),-1).clamp(0,self.h-1)
        xs=torch.stack((x0,x0+1,x0,x0+1),-1).clamp(0,self.w-1)
        donor=(ys.long()*self.w+xs.long())
        raw=self.image[0,:,y.long(),x.long()].T
        dist=(raw[:,None]-self.mean[donor]).square().mean(-1)
        yd,xd=4*ys+1.5,4*xs+1.5
        return b,dist,self.path(self.image,y,x,yd,xd),self.path(self.shifted,y,x,yd,xd),self.z[donor]

class Decoder(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode
        self.core=RangeDecoder("bilinear_"+("bounded4" if mode=="ranges" else "gaussian"))
        if mode not in ("gaussian","ranges"):
            self.additional=nn.Parameter(torch.tensor(math.log(.4/(6-.4))))

    def weights(self,features):
        base,dist,path,shifted,_=features
        exponent=.25+1.5*self.core.spatial.sigmoid()
        if self.mode=="ranges":
            bands=dist.new_tensor(RANGES)
            cost=(-torch.expm1(-.5*dist[...,None]/bands.square()))@(6*self.core.mix.sigmoid())
        else:
            width=.005+.495*self.core.width.sigmoid()
            cost=.5*dist/width.square()
            if self.mode not in ("gaussian","ranges"):
                if self.mode=="endpoint":cue=-torch.expm1(-.5*dist/.10**2)
                elif self.mode=="path":cue=path
                elif self.mode=="shifted":cue=shifted
                else:cue=path.roll(1,-1)
                cost=cost+6*self.additional.sigmoid()*cue
        return (base.log()*exponent-cost).softmax(-1)

    def forward(self,features):
        return (self.weights(features).unsqueeze(-2)@features[-1]).squeeze(-2)

    def coefficients(self):
        values={"spatial_exponent":float((.25+1.5*self.core.spatial.sigmoid()).detach())}
        if self.mode=="ranges":values["range_strengths"]=(6*self.core.mix.sigmoid()).detach().cpu().tolist()
        else:values["sigma"]=float((.005+.495*self.core.width.sigmoid()).detach())
        if hasattr(self,"additional"):values["extra_strength"]=float((6*self.additional.sigmoid()).detach())
        return values

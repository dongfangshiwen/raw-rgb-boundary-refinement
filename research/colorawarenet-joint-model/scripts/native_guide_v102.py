"""Use original camera pixels to interpolate a frozen coarse semantic score grid."""
import math

import torch
from torch import nn
import torch.nn.functional as F


class PixelGrid:
    def __init__(self, z, small_rgb, native_rgb):
        self.h, self.w = z.shape[-2:]
        self.H, self.W = native_rgb.shape[-2:]
        self.scores = z[0].permute(1,2,0).reshape(-1,z.shape[1])
        self.classes = self.scores.argmax(-1)
        # All arms use exactly the same donor colors; only the query resolution changes.
        work = (small_rgb*255).round().clamp(0,255)/255
        self.colors = F.adaptive_avg_pool2d(work,(self.h,self.w))[0].permute(1,2,0).reshape(-1,3)
        self.raw = native_rgb[0].permute(1,2,0).reshape(-1,3)
        self.low = F.interpolate(work,(self.H,self.W),mode="bilinear",align_corners=False)[0].permute(1,2,0).reshape(-1,3)
        labels=self.classes.reshape(1,1,self.h,self.w).float()
        self.gate=(F.max_pool2d(labels,3,1,1)!=-F.max_pool2d(-labels,3,1,1))[0,0]
        self.y=(torch.arange(self.H,device=z.device,dtype=torch.float32)+.5)*(self.h/self.H)-.5
        self.x=(torch.arange(self.W,device=z.device,dtype=torch.float32)+.5)*(self.w/self.W)-.5
        self.cy=(self.y+.5).floor().long().clamp(0,self.h-1)
        self.cx=(self.x+.5).floor().long().clamp(0,self.w-1)

    def active(self):
        return self.gate[self.cy[:,None],self.cx[None,:]].flatten()

    def four(self, ids, lowpass=False, include_distance=False):
        rows,cols=ids//self.W,ids%self.W
        y,x=self.y[rows],self.x[cols]
        y0,x0=y.floor().long(),x.floor().long()
        a,b=y-y0,x-x0
        prior=torch.stack(((1-a)*(1-b),(1-a)*b,a*(1-b),a*b),-1)
        top,bottom=y0.clamp(0,self.h-1)*self.w,(y0+1).clamp(0,self.h-1)*self.w
        left,right=x0.clamp(0,self.w-1),(x0+1).clamp(0,self.w-1)
        donors=torch.stack((top+left,top+right,bottom+left,bottom+right),-1)
        color=(self.low if lowpass else self.raw)[ids]
        dist=(color[:,None]-self.colors[donors]).square().mean(-1)
        bands=color.new_tensor((.02,.05,.10,.20))
        cost=(1-torch.exp(-.5*dist[...,None]/bands.square())).sum(-1)*(6/(1+math.exp(.5)))
        log_prior=prior.clamp_min(1e-15).log().masked_fill(prior==0,-torch.inf)
        order=dist.masked_fill(prior==0,torch.inf)
        best,top_index=order.min(-1)
        second=order.topk(2,largest=False).values[:,1]
        cls=self.classes[donors]
        different=order.masked_fill(cls==cls.gather(1,top_index[:,None]),torch.inf).amin(-1)
        def gap(other):
            finite=other.isfinite()
            other=torch.where(finite,other,best)
            return (other-best)/(other+best+.05**2)
        result=log_prior,cost,torch.stack((gap(second),gap(different)),-1),self.scores[donors]
        return (*result,dist) if include_distance else result

    def jbu(self,ids):
        row,col=ids//self.W,ids%self.W
        cy,cx=self.cy[row],self.cx[col]
        axis=torch.arange(-1,2,device=ids.device)
        delta=torch.cartesian_prod(axis,axis)
        y=cy[:,None]+delta[:,0]
        x=cx[:,None]+delta[:,1]
        donor=y.clamp(0,self.h-1)*self.w+x.clamp(0,self.w-1)
        spatial=(y-self.y[row,None]).square()+(x-self.x[col,None]).square()
        color=(self.raw[ids,None]-self.colors[donor]).square().mean(-1)
        prob=(-.5*(spatial/.65**2+color/.05**2)).softmax(-1)
        return torch.bmm(prob[:,None],self.scores[donor]).squeeze(1)


class Strength(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("scalar","rgb_gap","class_gap"):
            raise ValueError(mode)
        self.mode=mode
        self.bias=nn.Parameter(torch.zeros(()))
        if mode!="scalar":
            self.slope=nn.Parameter(torch.zeros(()))

    def forward(self, packed):
        log_prior,cost,gaps,donors=packed
        term=self.bias
        if self.mode!="scalar":
            term=term+self.slope*(2*gaps[:,0 if self.mode=="rgb_gap" else 1]-1)
        return interpolate(log_prior,cost,donors,2*term.sigmoid().reshape(-1,1))


def interpolate(log_prior,cost,donors,strength=1.):
    weights=(log_prior-strength*cost).softmax(-1)
    return torch.bmm(weights[:,None],donors).squeeze(1)

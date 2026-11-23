"""sRGB/D65 CIELAB and CIEDE2000 (kL=kC=kH=1), evaluated in float64.

Formula/test reference: Sharma, Wu & Dalal, Color Research and Application
30(1), 21-30, 2005. https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
The image score is the mean pixel DeltaE00; pair/image means are averaged later.
"""
import torch


@torch.no_grad()
def srgb_to_lab(rgb):
    rgb=rgb.clamp(0,1).to(torch.float64).movedim(1,-1)
    linear=torch.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055).pow(2.4))
    matrix=rgb.new_tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])
    xyz=linear@matrix.T
    xyz=xyz/rgb.new_tensor([.95047,1.,1.08883])
    delta=6/29
    f=torch.where(xyz>delta**3,xyz.clamp_min(0).pow(1/3),xyz/(3*delta**2)+4/29)
    x,y,z=f.unbind(-1)
    return torch.stack((116*y-16,500*(x-y),200*(y-z)),dim=-1)


@torch.no_grad()
def ciede2000_lab(first,second):
    """Broadcasting Lab tensors with the last dimension equal to three."""
    first,second=first.to(torch.float64),second.to(torch.float64)
    l1,a1,b1=first.unbind(-1);l2,a2,b2=second.unbind(-1)
    c1,c2=torch.sqrt(a1*a1+b1*b1),torch.sqrt(a2*a2+b2*b2)
    mean_c=(c1+c2)/2
    g=.5*(1-torch.sqrt(mean_c.pow(7)/(mean_c.pow(7)+25**7)))
    ap1,ap2=(1+g)*a1,(1+g)*a2
    cp1,cp2=torch.sqrt(ap1*ap1+b1*b1),torch.sqrt(ap2*ap2+b2*b2)
    h1=torch.rad2deg(torch.atan2(b1,ap1)).remainder(360)
    h2=torch.rad2deg(torch.atan2(b2,ap2)).remainder(360)
    dl,dc=l2-l1,cp2-cp1
    dh=h2-h1
    # The reference includes pairs exactly on the 180-degree discontinuity.
    dh=torch.where(dh>180+1e-12,dh-360,torch.where(dh < -180-1e-12,dh+360,dh))
    zero_chroma=(cp1*cp2)==0
    dh=torch.where(zero_chroma,torch.zeros_like(dh),dh)
    dH=2*torch.sqrt(cp1*cp2)*torch.sin(torch.deg2rad(dh/2))
    ml,mc=(l1+l2)/2,(cp1+cp2)/2
    hs=h1+h2
    mh=torch.where((h1-h2).abs()<=180+1e-12,hs/2,torch.where(hs<360,(hs+360)/2,(hs-360)/2))
    mh=torch.where(zero_chroma,hs,mh)
    radians=torch.deg2rad
    t=1-.17*torch.cos(radians(mh-30))+.24*torch.cos(radians(2*mh))+.32*torch.cos(radians(3*mh+6))-.20*torch.cos(radians(4*mh-63))
    theta=30*torch.exp(-((mh-275)/25).square())
    rc=2*torch.sqrt(mc.pow(7)/(mc.pow(7)+25**7))
    sl=1+.015*(ml-50).square()/torch.sqrt(20+(ml-50).square())
    sc=1+.045*mc;sh=1+.015*mc*t
    rt=-torch.sin(radians(2*theta))*rc
    vl,vc,vh=dl/sl,dc/sc,dH/sh
    return (vl.square()+vc.square()+vh.square()+rt*vc*vh).clamp_min(0).sqrt()


@torch.no_grad()
def ciede2000_rgb(first,second):
    return float(ciede2000_lab(srgb_to_lab(first),srgb_to_lab(second)).mean())

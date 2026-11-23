"""Explicit image geometry and label schemas; originals remain immutable."""
from dataclasses import dataclass,asdict
import hashlib
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Geometry:
    width:int
    height:int
    scaled_width:int
    scaled_height:int
    left:int
    top:int
    canvas_width:int
    canvas_height:int


def geometry(width,height,long_side=384,multiple=32):
    if min(width,height,long_side,multiple)<=0:raise ValueError('Positive dimensions required')
    scale=long_side/max(width,height)
    w=max(1,round(width*scale));h=max(1,round(height*scale))
    cw=math.ceil(w/multiple)*multiple;ch=math.ceil(h/multiple)*multiple
    return Geometry(width,height,w,h,(cw-w)//2,(ch-h)//2,cw,ch)


def decode_mask(array,schema):
    value=np.asarray(array)
    if value.ndim!=2:raise ValueError('Mask must be a single label plane')
    labels=set(np.unique(value).tolist())
    if schema=='own_partial':
        if not labels<={0,1,255}:raise ValueError('Expected own0/1/255')
        return value.astype(np.uint8,copy=True)
    if schema=='crosswalk_binary255':
        if not labels<={0,255}:raise ValueError('Expected author binary0/255')
        return (value>0).astype(np.uint8)
    if schema=='fpv_jpeg':
        if value.dtype!=np.uint8:raise ValueError('Expected grayscale8bit JPEG')
        return (value>127).astype(np.uint8)
    if schema=='camvid11':
        if not labels<=set(range(12)):raise ValueError('Expected CamVid0..11')
        result=value.astype(np.uint8,copy=True);result[result==11]=255
        return result
    raise ValueError('No implicit label conversion: '+schema)


def align_image(image,spec):
    value=np.asarray(image.convert('RGB').resize((spec.scaled_width,spec.scaled_height),Image.Resampling.BILINEAR)).copy()
    bottom=spec.canvas_height-spec.scaled_height-spec.top;right=spec.canvas_width-spec.scaled_width-spec.left
    return np.pad(value,((spec.top,bottom),(spec.left,right),(0,0)),mode='edge')


def align_mask(mask,spec):
    value=np.asarray(Image.fromarray(mask).resize((spec.scaled_width,spec.scaled_height),Image.Resampling.NEAREST)).copy()
    result=np.full((spec.canvas_height,spec.canvas_width),255,np.uint8)
    result[spec.top:spec.top+spec.scaled_height,spec.left:spec.left+spec.scaled_width]=value
    return result


def valid_pixels(spec):
    value=np.zeros((spec.canvas_height,spec.canvas_width),bool)
    value[spec.top:spec.top+spec.scaled_height,spec.left:spec.left+spec.scaled_width]=True
    return value


def rgb_signature(image):
    value=image.convert('RGB')
    return hashlib.sha256(f'{value.width}x{value.height}:RGB:'.encode()+value.tobytes()).hexdigest()


def correspondence(clear,hazy):
    """Image-only translation diagnostic, not permission to warp or infer depth."""
    arrays=[]
    for image in (clear,hazy):
        a=np.asarray(image.convert('L').resize((256,144),Image.Resampling.BILINEAR),dtype=np.float64)/255
        # Finite-difference channels reject constant radiometric offsets.
        gx=np.pad(np.diff(a,axis=1),((0,0),(0,1)));gy=np.pad(np.diff(a,axis=0),((0,1),(0,0)))
        arrays.append(np.stack((gx,gy)))
    a,b=arrays;results=[]
    for dy in range(-3,4):
        for dx in range(-3,4):
            left=a[:,5:-5,5:-5];right=b[:,5+dy:144-5+dy,5+dx:256-5+dx]
            correlation=float((left*right).sum()/max(np.sqrt((left*left).sum()*(right*right).sum()),1e-12))
            results.append((correlation,dy,dx))
    best=max(results);zero=next(v[0] for v in results if v[1:]==(0,0))
    return dict(best_shift_256=[best[2],best[1]],zero_edge_cosine=zero,best_edge_cosine=best[0],
        improvement_over_identity=best[0]-zero,review_translation=bool(best[1:]!=(0,0) and best[0]-zero>.03),
        scope='Coarse image-only edge translation screen, not calibrated registration; no image/mask warping performed.')

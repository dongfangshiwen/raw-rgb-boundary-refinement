"""Image-only native-size restored RGB and indexed semantic mask."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from boundary_refinement.data.provenance import write,digest,now
from boundary_refinement.models.joint.decision_joint import load_model

PALETTE=np.array([[70,130,180],[180,70,70],[220,190,100],[128,64,128],[190,130,190],
    [70,150,60],[220,220,30],[120,100,80],[30,70,180],[220,50,70],[40,190,190]],np.uint8)


def palette(classes):return PALETTE if classes==11 else np.array([[0,0,0],[0,230,90]],np.uint8)


@torch.inference_mode()
def predict_native(model,image):
    height,width=image.shape[-2:];dh=(-height)%32;dw=(-width)%32;top=dh//2;left=dw//2
    padded=F.pad(image,(left,dw-left,top,dh-top),mode='replicate')
    rgb,logits,_=model(padded)
    return rgb[:,:,top:top+height,left:left+width],logits[:,:,top:top+height,left:left+width]


def outputs(rgb,logits,array,classes):
    restored=(rgb[0].clamp(0,1)*255).round().byte().permute(1,2,0).cpu().numpy()
    labels=logits.argmax(1)[0].byte().cpu().numpy();colours=palette(classes)[labels]
    overlay=(array*.5+colours*.5).round().astype(np.uint8)
    return restored,labels,overlay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDArequired')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sha=digest(args.checkpoint);model,packet=load_model(args.checkpoint)
    with Image.open(args.image) as im:array=np.asarray(im.convert('RGB')).copy()
    image=torch.from_numpy(array).permute(2,0,1)[None].cuda().float()/255
    rgb,logits=predict_native(model,image)
    if rgb.shape!=image.shape or logits.shape!=(1,packet['classes'],*image.shape[-2:]) or not bool(torch.isfinite(rgb).all()&torch.isfinite(logits).all()):raise ValueError('Native shape/finiteness')
    restored,labels,overlay=outputs(rgb,logits,array,packet['classes']);args.output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(restored).save(args.output/'restored.png');Image.fromarray(overlay).save(args.output/'overlay.png')
    mask=Image.fromarray(labels).convert('P');colours=palette(packet['classes']).flatten().tolist();mask.putpalette(colours+[0]*(768-len(colours)));mask.save(args.output/'mask.png')
    if digest(args.checkpoint)!=sha:raise ValueError('Selected weightchanged')
    write(args.output/'inference.json',dict(method=packet['method'],seed=packet['seed'],classes=packet['classes'],selected_epoch=packet['epoch'],
        checkpoint=str(args.checkpoint.resolve()),checkpoint_SHA256=sha,input=str(args.image.resolve()),input_SHA256=digest(args.image),
        native_size=[array.shape[1],array.shape[0]],mask_encoding='Indexedclass IDs0..classes-1; palette visualization. No GTvoidclass emitted. RGB-onlyinput.',
        geometry='NativeRGB, symmetricreplicatepadtonext32,fulljointmodelonce,unpadbothoutputs; no squareresize/tiling. DemonstratedCamVid480x360; largerimagesrequireavailableGPUmemory.',
        output_SHA256={n:digest(args.output/n) for n in ('restored.png','mask.png','overlay.png')},finished_utc=now()))
    print(dict(method=packet['method'],output=str(args.output.resolve())))


if __name__=='__main__':main()

"""Replot the two archived segmentation crops on the original SemanticSTF camera frame."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

PALETTE=np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
                  (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
                  (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),
                  (0,80,100),(0,0,230),(119,11,32)],np.uint8)
BASE=Path(__file__).resolve().parent

def render(camera,output):
    with np.load(BASE/"figure14-measured-source.npz",allow_pickle=False) as f:
        xy,labels=f["xy"],f["reference"]
        x,y,w,h=map(int,f["crop"])
        native=[f["released"],f["paper"]]
        ident=f["id"].item()
    manifest=json.loads((BASE.parent/"records/stf-manifest.json").read_text("utf8"))
    row=next(r for r in manifest["rows"] if r["id"]==ident)
    if hashlib.sha256(camera.read_bytes()).hexdigest()!=row["sha256"][0]:
        raise ValueError("The camera-file hash differs from the original source frame.")
    with Image.open(camera) as im:rgb=np.array(im.convert("RGB"))
    crop=rgb[y:y+h,x:x+w].astype(float)/255
    font="Times New Roman" if any(v.name=="Times New Roman" for v in font_manager.fontManager.ttflist) else "DejaVu Serif"
    plt.rcParams.update({"font.family":font,"font.size":10})
    fig,axes=plt.subplots(2,2,figsize=(6.25,3.63))
    fig.subplots_adjust(left=.002,right=.998,bottom=.008,top=.931,wspace=.047,hspace=.24)
    axs=axes.ravel()
    for ax,title in zip(axs,("a   Camera view","b   Measured reference points","c   Released Mask2Former","d   RGB-guided decoding")):
        ax.set_title(title,loc="left",fontweight="bold",pad=2.5,fontsize=10,color="#172f38")
        ax.set_axis_off()
    axs[0].imshow(rgb,extent=(0,rgb.shape[1],rgb.shape[0],0))
    axs[0].add_patch(Rectangle((x,y),w,h,edgecolor="#ffd15f",linewidth=1.15,facecolor="none"))
    axs[1].imshow(.63*crop)
    axs[1].scatter(xy[:,0],xy[:,1],c=PALETTE[labels]/255.,s=3.4,linewidths=.23,edgecolors="#ffffff")
    for ax,k in zip(axs[2:],native):ax.imshow(.7*PALETTE[k]/255.+.3*crop)
    for ax in axs[1:]:ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
    fig.savefig(output,dpi=330,facecolor="white",bbox_inches="tight",pad_inches=.025)
    plt.close(fig)

if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("camera",type=Path,help="Original 2018-10-29_14-30-29_02390.png from SemanticSTF.")
    args.add_argument("output",type=Path,help="Output PNG.")
    a=args.parse_args()
    render(a.camera,a.output)

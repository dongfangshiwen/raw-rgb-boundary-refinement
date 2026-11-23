"""Plot measured validation effects and fixed-index model outputs for Appendix A."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def read(f):return json.loads((HERE/f).read_text(encoding="utf-8"))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False})
def loadimg(p):
    with Image.open(p) as im:return np.array(im)
def effects():
    results=read("semantic-results.json")["bddval"]
    controls=[("released","Released decoder"),("fixed_jbu","Fixed Gaussian JBU"),("gaussian_gaussian","Fitted Gaussian JBU"),
              ("bilinear_bounded1","One bounded range"),("bilinear_gaussian","One Gaussian range"),("fixed_four","Fixed four ranges")]
    fig,ax=plt.subplots(figsize=(6.35,2.8))
    for target,color,offset,name in [("mask2former","#027c78",.12,"Mask2Former"),("segformer","#605296",-.12,"SegFormer")]:
        y=np.arange(len(controls))[::-1]+offset
        r=results[target]["all"]["contrasts"]
        est=np.array([r["bilinear_bounded4 minus "+k]["boundary"]["pp"] for k,_ in controls])
        b=np.array([r["bilinear_bounded4 minus "+k]["boundary"]["ci95"] for k,_ in controls])
        ax.errorbar(est,y,xerr=(est-b[:,0],b[:,1]-est),fmt="o" if offset>0 else "s",color=color,
                    markersize=4,capsize=2,elinewidth=1.2,lw=0,label=name)
    ax.axvline(0,color="#727b81",lw=.9,ls="--");ax.set_axisbelow(True);ax.grid(axis="x",alpha=.17)
    ax.set_yticks(np.arange(len(controls))[::-1],[v for _,v in controls]);ax.tick_params(axis="y",length=0)
    ax.set_xlabel("Boundary mIoU difference (percentage points)")
    ax.legend(ncol=2,frameon=False,loc="lower right",bbox_to_anchor=(1,1.015),
              borderaxespad=0,fontsize=8)
    ax.set_xlim(-.3,1.55);ax.set_ylim(-.48,len(controls)-.30)
    fig.subplots_adjust(left=.28,bottom=.20,right=.98,top=.87)
    fig.savefig(HERE/"matched-decoder-effects.png",dpi=300,bbox_inches="tight",pad_inches=.06);plt.close(fig)
def photo():
    rows=read("photo-individual.json")
    fig,axes=plt.subplots(2,4,figsize=(6.5,2.75))
    for i,(ds,label) in enumerate([("bedde","BeDDE"),("revide_final","REVIDE")]):
        first=next(r for r in rows if r["ds"]==ds)
        for j,(tag,title,arm) in enumerate([("input","Input","input"),("mct","MCT","public_mct"),("scalar","Scalar blend","scalar"),("reference","Clear reference",None)]):
            ax=axes[i,j];a=loadimg(HERE/f"photo-{ds}-{tag}.png");ax.imshow(a,aspect="equal");ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():s.set_visible(False)
            if i==0:ax.set_title(title,fontweight="bold",fontsize=8,pad=3)
            if j==0:ax.set_ylabel(label,fontweight="bold",fontsize=8)
            if arm:
                val=next(r for r in rows if r["ds"]==ds and r["key"]==first["key"] and r["arm"]==arm)["psnr"]
                ax.set_xlabel(f"{val:.2f} dB",fontsize=7,labelpad=1)
    fig.subplots_adjust(left=.07,right=.995,bottom=.10,top=.91,hspace=.27,wspace=.055)
    fig.savefig(HERE/"paired-photo-examples.jpg",dpi=300,bbox_inches="tight",pad_inches=.025);plt.close(fig)
def dual():
    row=read("bdd-manifest.json")["rows"][0]
    orig=loadimg(ROOT/row["rgb_path"]);rest=loadimg(HERE/"example-mask2former/restored.png")
    rgb=loadimg(HERE/"example-mask2former/mask_color.png")
    gt=loadimg(ROOT/row["mask_path"])
    palette=np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],
        [250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],
        [0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],np.uint8)
    truth=np.full_like(orig,235);truth[gt<19]=palette[gt[gt<19]]
    fig,axes=plt.subplots(2,2,figsize=(6.3,3.85))
    for ax,img,label in zip(axes.ravel(),[orig,rest,rgb,truth],["Original exposure","Restored RGB","Predicted labels","Reference labels"]):
        ax.imshow(img);ax.set_axis_off();ax.set_title(label,fontsize=8.5,fontweight="bold",pad=2)
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.95,wspace=.025,hspace=.14)
    fig.savefig(HERE/"captured-output-panels.jpg",dpi=300,bbox_inches="tight",pad_inches=.03);plt.close(fig)
if __name__=="__main__":effects();photo();dual()

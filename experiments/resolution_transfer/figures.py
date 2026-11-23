"""Draw confidence intervals and measured fog examples from archived predictions."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from experiments.resolution_transfer.cohort import OUT, ROOT, digest, read, write
from experiments.recording_gate import gate
from experiments.resolution_transfer.sampling_bridge import native_labels

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})
NAVY = "#153743"
TEAL = "#00776f"
PURPLE = "#72538e"
MEASURE = "#ffc747"
TARGETS = ("mask2former", "segformer")
NAMES = ("Mask2Former", "SegFormer-B1")
COLORS = np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
                   (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
                   (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
                   (0,60,100),(0,80,100),(0,0,230),(119,11,32)], dtype=np.uint8)


def forest():
    dense = read(OUT / "results.json")["panels"]
    sparse = read(OUT / "stf-results.json")["results"]
    bridge = read(OUT / "sampling-bridge-results.json")["datasets"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), sharex=True)
    fig.subplots_adjust(left=.045, right=.985, top=.73, bottom=.26, wspace=.13)
    triple = (("A    Dense camera boundaries", "BDD100K  ·  19 classes"),
              ("B    Sparse camera-mask sample", "Same BDD100K images  ·  13 classes"),
              ("C    Real-fog LiDAR projections", "SemanticSTF  ·  13 classes"))
    for n, ax in enumerate(axes):
        ax.axvline(0, color="#9ba8ab", lw=1.1)
        ax.grid(axis="x", color="#e6ebea", lw=.6)
        for i, target in enumerate(TARGETS):
            if n == 0:
                data = dense[target]["new_bdd"]["scales"]["768"]["boundary"]
            elif n == 1:
                data = bridge["bdd"][target]["sampled_near12"]
            else:
                data = sparse[target]["all"]["endpoints"]["near12"]
            est = data["contrasts"]["paper - released"]
            lo, hi = est["ci95"]
            y = 1 - i
            color = (TEAL, PURPLE)[i]
            ax.errorbar(est["pp"], y, xerr=[[est["pp"]-lo],[hi-est["pp"]]], fmt="o",
                        markersize=6.5, lw=2, capsize=3, color=color)
            ax.text(.03, (y+.35)/2.1 + .12, NAMES[i], transform=ax.transAxes,
                    fontsize=11, color=color)
            ax.annotate(f'{est["pp"]:+.2f}', (est["pp"], y), xytext=(0,-13),
                        textcoords="offset points", ha="center", va="top", color=color, fontsize=10)
        a,b = triple[n]
        ax.set_title(a, loc="left", weight="bold", fontsize=13, color=NAVY, pad=26)
        ax.text(0,1.045,b, transform=ax.transAxes, fontsize=11, color="#596b72")
        ax.set_xlim(-1.2, 2.05)
        ax.set_ylim(-.55,1.55)
        ax.set_yticks([])
        ax.set_xticks([-1, 0, 1, 2], ["−1","0","+1","+2"])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9ba8ab")
    fig.text(.044,.94,"Boundary evidence across reference types",size=19,weight="bold",color=NAVY)
    fig.text(.044,.105,"Manuscript decoder minus released backbone, in percentage points. Bars show paired 95% intervals.",
             size=12, color=NAVY)
    fig.text(.044,.045,"A uses a six-pixel camera-mask band. B and C use nearby measured points of different classes. Each panel has its own reference.",
             size=10, color="#52666e")
    fig.savefig(OUT / "boundary-evidence.png",dpi=180)
    plt.close(fig)


def select_crop(xy, near, shape):
    height, width = shape
    cw, ch = 480, 304
    pts = xy[near]
    def starts(size, window):
        return sorted(set(range(0, size-window+1, 32)) | {size-window})
    best = (-1, 0, 0)
    for top in starts(height, ch):
        for left in starts(width, cw):
            k = int(((pts[:, 0]>=left) & (pts[:, 0]<left+cw) & (pts[:, 1]>=top) & (pts[:, 1]<top+ch)).sum())
            if k > best[0]:
                best = (k, left, top)
    return best[1],best[2],cw,ch


@torch.inference_mode()
def fog_figure():
    gate.ev.setup.configure_cuda()
    model=gate.ev.Target("mask2former")
    paper=gate.ev.original.load()
    fixed=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    preview=read(OUT/"stf-preview.json")["rows"]
    rows={r["id"]:r for r in read(OUT/"stf-manifest.json")["rows"]}
    fig, axes = plt.subplots(len(preview),5,figsize=(14,8.1))
    fig.subplots_adjust(left=.087, right=.992, bottom=.107, top=.862, wspace=.045,hspace=.16)
    headers=["Full scene","Human point labels","Released Mask2Former","Manuscript decoder","Outcomes at measured points"]
    for ax, head in zip(axes[0],headers):
        ax.set_title(head,fontsize=12.5,color=NAVY,weight="bold",pad=8)
    summary=[]
    for j, ref in enumerate(preview):
        r=rows[ref["id"]]
        with Image.open(ROOT/r["paths"][0]) as im:
            rgb=np.array(im.convert("RGB"))
        with np.load(OUT/"measured-points/mask2former"/(r["id"]+".npz")) as f:
            xy=f["xy"].astype(int)
            gt=f["truth"]
            near=f["band"]<=1
            archived=f["pred"][0]
            ids=f["arms"].tolist()
        left,top,w,h=select_crop(xy,near,rgb.shape[:2])
        crop=np.s_[top:top+h,left:left+w]
        include=(xy[:,0]>=left)&(xy[:,0]<left+w)&(xy[:,1]>=top)&(xy[:,1]<top+h)
        pos=xy-np.array((left,top));m=include&near
        both={}
        for name,seed,pred in native_labels(rgb,"mask2former",model,paper,fixed):
            if name=="released" or (name=="paper" and seed==0):
                np.testing.assert_array_equal(pred[xy[:,1],xy[:,0]],archived[ids.index(name),0])
                both[name]=pred
            if len(both)==2:
                break
        ax=axes[j]
        ax[0].imshow(rgb[::3,::3],extent=(0,rgb.shape[1],rgb.shape[0],0))
        ax[0].add_patch(Rectangle((left,top),w,h,fill=False,edgecolor=MEASURE,lw=1.8))
        ax[0].text(-.055,.5,r["day"]+"\n"+r["weather"].replace("_"," "),ha="right",va="center",
                   transform=ax[0].transAxes,size=11,color=NAVY)
        sky=rgb[crop].astype(float)/255.
        ax[1].imshow(.57*sky)
        ax[1].scatter(pos[include,0],pos[include,1],s=4.5,c=COLORS[gt[include]]/255.,
                      edgecolors="white",linewidths=.15)
        for a,name in ((ax[2],"released"),(ax[3],"paper")):
            display=.68*COLORS[both[name][crop]]/255.+.32*sky
            a.imshow(display)
        g=np.dot(sky,[.299,.587,.114])
        ax[4].imshow(g,cmap="gray",vmin=0,vmax=1.4)
        old=archived[ids.index("released"),0];new=archived[ids.index("paper"),0]
        corrected=m&(new==gt)&(old!=gt)
        harmed=m&(old==gt)&(new!=gt)
        under=m&~corrected&~harmed
        ax[4].scatter(pos[under,0],pos[under,1],marker=".",s=3.5,c="#afbec4",linewidths=0)
        for indices,marker,c in ((corrected,"o","#00e0ab"),(harmed,"X","#ff924f")):
            ax[4].scatter(pos[indices,0],pos[indices,1],s=22,marker=marker,c=c,edgecolors="#101d21",linewidths=.45)
        for a in ax:
            a.set_axis_off()
        for a in ax[1:]:
            a.set_xlim(-.5,w-.5);a.set_ylim(h-.5,-.5)
        summary.append(dict(id=r["id"],crop_xywh=[left,top,w,h],reference_points=int(include.sum()),
                            near_points=int(m.sum()),corrected=int(corrected.sum()),newly_incorrect=int(harmed.sum())))
    fig.text(.019,.963,"Camera predictions on four real-fog recording days",weight="bold",color=NAVY,size=19)
    fig.text(.019,.917,"Long edge 768 pixels  ·  Mask2Former  ·  manuscript seed 101  ·  same crop and label palette for both predictions",
             size=12,color="#52666e")
    handles=[Line2D([],[],marker="o",linestyle="",markerfacecolor="#00e0ab",markeredgecolor=NAVY,label="Newly correct"),
             Line2D([],[],marker="X",linestyle="",markerfacecolor="#ff924f",markeredgecolor=NAVY,label="Newly incorrect"),
             Line2D([],[],marker=".",linestyle="",color="#84959c",label="Other measured near points")]
    fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(.013,.029),ncol=3,frameon=False,fontsize=11,
               handletextpad=.4,columnspacing=1.4)
    fig.text(.019,.013,"Gold boxes enclose 480 × 304 crops chosen by reference-point density. Only the measured points carry human labels.",
             size=10.5,color="#52666e")
    fig.savefig(OUT/"fog-prediction-examples.png",dpi=185)
    plt.close(fig)
    write(OUT/"figure-audit.json",dict(frames=summary,seed=101,backbone="mask2former",
          selection="Four dates selected by hash before scoring. Per-date reference-points-only crop, row-major tie break.",
          scored_pixel_predictions_match_archive=True,interval_plot_sha256=digest(OUT/"boundary-evidence.png"),
          examples_sha256=digest(OUT/"fog-prediction-examples.png")))


if __name__ == "__main__":
    forest()
    fog_figure()

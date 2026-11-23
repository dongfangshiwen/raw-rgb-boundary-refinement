"""Plot paired cluster estimates and display the preselected real model predictions."""
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts.weatherproof_release_v103 import OUT,read

PALETTE=np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),
                  (250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),
                  (255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)],np.uint8)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.labelcolor":"#283a45","xtick.color":"#33454e","ytick.color":"#33454e",
                     "figure.facecolor":"white","savefig.facecolor":"white"})
FIG=OUT/"figures"
def interval_plot():
    a=read(OUT/"results.json")["strata"];dense=read(OUT/"dense-roads/results.json")
    segments=[("BDD drivable area",a["bdd_tagged"]["primary"]["boundary"])]
    if dense["groups"]:segments.insert(0,("BDD dense semantics",dense["primary"]["boundary"]))
    segments.append(("WeatherProof, 3 shared classes",a["wp_fogmix"]["primary"]["boundary"]))
    colors=("#1a6666","#785186","#a65b32")
    refs=("paper","segfix","gaussian_segfix")
    names=("Decoder alone","SegFix alone","Gaussian + SegFix")
    fig,axes=plt.subplots(1,len(segments),figsize=(4*len(segments),3.65),layout="constrained",sharey=True)
    for ax,(name,k) in zip(axes,segments):
        ax.axvline(0,color="#72838d",lw=.9)
        cs=k["targets"]["mask2former"]["contrasts"]
        spans=[]
        for j,(ref,col) in enumerate(zip(refs,colors)):
            r=cs["paper_segfix minus "+ref];y=2-j;x=r["pp"];ci=r["ci95"]
            if ci:
                ax.plot(ci,[y,y],color=col,lw=2,solid_capstyle="round");spans.extend(ci)
            spans.append(x);ax.scatter(x,y,s=44,color=col,zorder=3,edgecolors="white",linewidth=.7)
            ax.annotate(f"{x:+.2f}",(x,y),xytext=(0,8),textcoords="offset points",ha="center",color=col,fontsize=9)
        ax.set_title(f"{name}\n{k['informative_groups']} groups with scored boundaries",fontsize=10,loc="left",pad=11,
                     color="#263a46",fontweight="semibold")
        limit=max(abs(v) for v in spans)*1.12+.035
        ax.set_xlim(-limit,limit);ax.set_ylim(-.5,2.55);ax.set_yticks([2,1,0],names)
        ax.tick_params(axis="y",length=0);ax.set_xlabel("Change in boundary mIoU (pp)")
        ax.spines["left"].set_visible(False);ax.grid(axis="x",alpha=.13)
    fig.suptitle("Adding SegFix after color decoding on frozen Mask2Former",
                 fontsize=14,fontweight="semibold",color="#203843")
    fig.savefig(FIG/"paired-boundary-contrasts.png",dpi=200,bbox_inches="tight");plt.close(fig)

def road_example():
    cohort="dense-road" if (OUT/"examples/dense-road/mask2former").exists() else "bdd"
    d=OUT/"examples"/cohort/"mask2former"
    a=lambda name:np.asarray(Image.open(d/(name+".png")))
    im,y=a("input"),a("truth-ids");p,m=a("paper-city-ids"),a("paper_segfix-city-ids")
    binary=cohort=="bdd"
    if binary:p,m=[(x==0).astype(np.uint8) for x in (p,m)]
    palette=PALETTE if not binary else np.array([[80,85,88],[33,140,136]],np.uint8)
    def color(x):
        out=np.broadcast_to([218,221,222],(*x.shape,3)).copy().astype(np.uint8)
        valid=x<len(palette);out[valid]=palette[x[valid]];return out
    transition=np.zeros(y.shape,bool)
    for a1,b1 in ((np.s_[1:,:],np.s_[:-1,:]),(np.s_[:,1:],np.s_[:,:-1])):
        edge=(y[a1]<19)&(y[b1]<19)&(y[a1]!=y[b1]);transition[a1]|=edge;transition[b1]|=edge
    h,w=y.shape;ch,cw=h//3,w//3
    # Select the crop by reference contour density alone.
    coords=[(int(yy),int(xx)) for yy in np.linspace(0,h-ch,9) for xx in np.linspace(0,w-cw,9)]
    yy,xx=max(coords,key=lambda pt:transition[pt[0]:pt[0]+ch,pt[1]:pt[1]+cw].sum())
    sl=np.s_[yy:yy+ch,xx:xx+cw]
    gray=np.asarray(Image.fromarray(im).convert("L")); overlay=np.stack([gray]*3,-1)*.55+80
    mask=(p!=m)&(y<19)&maximum_filter(transition,size=13)
    rescued=mask&(y==m);harmed=mask&(y==p);other=mask&~(rescued|harmed)
    overlay[other]=[159,120,194];overlay[rescued]=[0,155,108];overlay[harmed]=[227,78,40]
    fig,axes=plt.subplots(2,3,figsize=(12.1,5.75),layout="constrained")
    contents=[(im,"Raw keyframe"),(color(y),"Dataset annotation"),(color(p),"Decoder prediction"),
              (im[sl],"Raw crop"),(color(m)[sl],"Decoder + SegFix, crop"),(overlay[sl].astype(np.uint8),"Changed boundary pixels, crop")]
    for ax,(image,name) in zip(axes.flat,contents):
        ax.imshow(image,interpolation="nearest",aspect="equal");ax.set_title(name,loc="left",fontsize=10);ax.axis("off")
    from matplotlib.patches import Rectangle
    for ax in axes[0]:ax.add_patch(Rectangle((xx,yy),cw,ch,fill=False,lw=1.15,edgecolor="white"))
    source=read(d/"source.json")
    fig.suptitle(f"Frozen Mask2Former predictions  |  BDD100K keyframe {source['id']}",fontsize=13,fontweight="semibold")
    fig.supxlabel("Green = corrected by the move     Orange = spoiled by the move     Violet = changed, both wrong     Light gray in the reference = ignored",fontsize=9)
    fig.savefig(FIG/"road-prediction-audit.png",dpi=180,bbox_inches="tight");plt.close(fig)

def photo():
    d=OUT/"examples/wp/photographs"
    entry=read(d/"metrics.json")
    extra=read(OUT/"manuscript-rgb-individual.json")["rows"]
    entry["psnr"]["manuscript"]=next(r for r in extra if r["id"]==entry["id"])["seeds"]["32003"]["psnr"]
    modes=("input","mct","manuscript","scalar","pgl","clear")
    names=("Adverse input","MCT","Manuscript RGB","Calibrated MCT","PGL-Net-T","Paired clear reference")
    fig,axes=plt.subplots(1,6,figsize=(13.8,2.8),layout="constrained")
    for ax,k,name in zip(axes,modes,names):
        ax.imshow(np.asarray(Image.open(d/(k+".png"))))
        ax.set_title(name+("\n%.2f dB"%entry["psnr"][k] if k!="clear" else "\n"),fontsize=9,loc="left")
        ax.set_axis_off()
    fig.suptitle("Paired real-weather RGB predictions  |  "+entry["id"],fontsize=11.5,fontweight="semibold")
    fig.savefig(FIG/"weatherproof-paired-rgb.png",dpi=190,bbox_inches="tight");plt.close(fig)

def run():
    FIG.mkdir(parents=True,exist_ok=True)
    interval_plot();road_example();photo()
if __name__=="__main__":run()

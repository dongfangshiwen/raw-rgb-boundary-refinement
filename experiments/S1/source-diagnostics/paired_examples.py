"""Plots the first manifest-ordered day and night cases at t=.40 using the calibrated guides."""
import sys,json,gzip
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[2] / "research" / "model_code"))
from scripts import joint_tone_prototype_v99 as j

@torch.inference_mode()
def run():
    j.configure_cuda()
    engine=j.base.DualScaleFeatures()
    nets,raw,heads,coeff=j.weighted.load(),j.base.anchors(),j.calib.heads(),j.calib.load()
    s,sc=j.SEEDS[0],j.road.SEEDS[0]
    with gzip.open(j.ROOT/"joint-seg.json.gz","rt",encoding="utf-8") as f:
        archive={r["id"]:r for r in json.load(f) if r["ds"]=="MUSES_t40" and r["rgb"]=="curve" and r["sem"]=="fixed" and r["seed"]==s}
    selected={}
    for ds,key,group,day,rgb,truth in j.held.scene_cases():
        if ds!="new_muses_clear" or day in selected:
            continue
        gt=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
        fog=j.to_rgb(j.srgb(.4*j.linear_rgb(gt)+.6*.85))
        v,win,_=engine(fog)
        fine=j.resize(torch.from_numpy(fog).cuda().permute(2,0,1)[None].float()/255.,win.shape[-2:],antialias=True)
        u,a,w=j.calib.endpoints(j.road.geometry(fine,engine.core.rgb),engine.core.bank,heads,sc)
        pred=j.calib.mixture(u,a,w,"curve",coeff["curve",sc])
        z=j.raw_anchor(raw["raw_detail",s],v)
        guides=j.base.render_guides(fog,pred,z.shape[-2:])
        delta=j.base.prediction("dual_free",nets["dual_core",s],z,j.weighted.prepare(guides,z,"dual_core"))
        mask=(j.resize(z,truth.shape)+j.resize(delta,truth.shape)).argmax(1)[0]
        assert j.confusion_matrix(mask,truth,19).tolist()==archive[key]["counts"],key
        selected[day]=(key,group,[fog,j.to_rgb(pred),rgb,j.paint(mask.cpu().byte().numpy()),j.paint(truth.cpu().byte().numpy())])
        if len(selected)==2: break
    assert len(selected)==2
    fig,axs=plt.subplots(2,5,figsize=(9.5,2.56))
    keys=sorted(selected,key=lambda x:("night" in x.lower(),x))
    for i,tag in enumerate(keys):
        for col,(pic,label) in enumerate(zip(selected[tag][2],("Veiled input","Calibrated RGB","Clear reference","Predicted labels","Reference labels"))):
            ax=axs[i,col]; ax.imshow(pic,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
            for sp in ax.spines.values():sp.set_visible(False)
            if i==0: ax.set_title(label,fontsize=9.5,fontweight="bold",pad=3.5)
            if col==0:ax.set_ylabel(tag.title(),fontsize=9.5,fontweight="bold")
    fig.subplots_adjust(left=.037,right=.998,bottom=.008,top=.87,hspace=.1,wspace=.027)
    fig.savefig(OUT/"fig5-visibility.png",dpi=330,bbox_inches="tight",pad_inches=.045)
    (OUT/"paired-examples.json").write_text(json.dumps({k:dict(id=v[0],group=v[1],confusion_verified=True) for k,v in selected.items()},indent=2),encoding="utf-8")
    print({k:v[:2] for k,v in selected.items()})

if __name__ == "__main__":
    run()

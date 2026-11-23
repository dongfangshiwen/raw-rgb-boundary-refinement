"""Paired re-evaluation of the source-selected Dehaze-Seg-S comparator; preserve the first transfer."""
from collections import defaultdict
from hashlib import sha256

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from scripts import idd_fog_evaluate_v100 as test
from scripts import idd_peer_setfit_v100 as repair
from scripts import joint_tone_prototype_v99 as j
from scripts.idd_fog_data_v100 import DATA,MAP,OUT,fixed
from scripts.idd_fog_reference_v100 import iou_counts
from scripts.projective_ledger_v12 import digest,read,write


D=repair.DIR
SEEDS=repair.old.SEEDS


def compute_summary(rows):
    bank=defaultdict(list)
    for r in rows:
        bank[r["group"]].append(r["v"])
    groups=sorted(bank)
    data=np.stack([np.sum(bank[g],axis=0) for g in groups])
    n=len(data)
    draws=np.vstack((np.ones(n),np.random.default_rng(100481).multinomial(n,np.full(n,1/n),5000)))
    a=(draws@data.reshape(n,-1).astype(float)).reshape(len(draws),*data.shape[1:])
    io=np.divide(100*a[...,0,:],a[...,1,:],out=np.zeros_like(a[...,0,:]),where=a[...,1,:]>0)
    values=io.mean((-1,-2))
    result={}
    for k,m in enumerate(("full","boundary")):
        own,first,corrected=values[:,k].T
        result[m]=dict(weighted=float(own[0]),initial_peer=float(first[0]),corrected_peer=float(corrected[0]),
                       ours_minus_corrected=float(own[0]-corrected[0]),
                       paired_ci95=np.quantile((own-corrected)[1:],[.025,.975]).tolist(),
                       corrected_seeds=io[0,k,2].mean(-1).tolist())
    return dict(images=len(rows),drives=n,scores=result)


@torch.inference_mode()
def run():
    repair.configure_cuda()
    models={s:repair.load(s) for s in SEEDS}
    first=models[SEEDS[0]]
    fixed(D/"score-protocol.json",dict(source_selection=digest(D/"protocol.json"),evaluator=digest(__file__),
        weights={str(s):digest(D/f"queries-{s}.pth") for s in SEEDS},external=digest(OUT/"external-index.json"),
        synthetic=digest(OUT/"synthetic-index.json"),paired="Same 512 input and native GT as the original joint-system "
                     "comparison. Full-coverage diagnostic; the IDD cohort has already been evaluated. RGB branch stays "
                     "frozen; verify both the exported bytes and 26 paired-image RGB MSEs."))
    source_rows={r["id"]:r for r in read(DATA/"evaluation.json")["rows"]}
    examples={p.name for p in (OUT/"examples").iterdir()}
    batches=[]
    def evaluate(tag,index,rgb,truth,clean=None):
        key=index["id"]
        path=D/"counts"/tag/(key.replace("/","_")+".npz")
        path.parent.mkdir(parents=True,exist_ok=True)
        with np.load(index["file"]) as old:
            arms=old["arms"].tolist()
            previous=np.stack((old["full"],old["edge"]))
            old_rgb=old["photo"]
        if not path.exists():
            counts=test.Counts(truth)
            x,size=repair.old.peer_input(rgb)
            full=[];edge=[]
            ex=(key.replace("/","_") in examples and tag=="external")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                im,*feat=first.backbone((x-first.mean)/first.std)
                restored=(im.float()*first.std+first.mean)[...,:size[0],:size[1]].clamp(0,1)
                for s in SEEDS:
                    z=repair.old.peer_forward(models[s],feat,size)
                    labels=j.resize(z,truth.shape).argmax(1)[0].byte()
                    c,e=counts.score(labels);full.append(c);edge.append(e)
                    if ex and s==SEEDS[0]:
                        loc=D/"examples"/key.replace("/","_");loc.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(labels.cpu().numpy()).save(loc/"corrected-labels.png")
                        px=j.to_rgb(j.resize(restored,truth.shape))
                        if not np.array_equal(px,np.array(Image.open(OUT/"examples"/loc.name/"published-rgb.png"))):
                            raise ValueError("The source-side repair changed RGB bytes")
            if clean is not None:
                ref=j.resize(clean,size,antialias=True)
                mse=float((restored-ref).square().mean())
                if not np.allclose(old_rgb[2],mse,atol=1e-8,rtol=1e-6):
                    raise ValueError("RGB baseline changed during semantic training")
            with path.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,full=np.stack(full),edge=np.stack(edge))
            path.with_suffix(".part").replace(path)
        with np.load(path) as f:
            new=np.stack([f["full"],f["edge"]])
        oldcounts=previous[:,arms.index("published512")]
        if not np.array_equal(new.sum(-1),oldcounts.sum(-1)):
            raise ValueError("Unequal IDD class support after repair")
        values=np.stack((previous[:,arms.index("weighted512")],oldcounts,new),axis=1)
        cohort_classes=sorted(MAP.values()) if tag=="external" else list(range(19))
        batches.append(dict(id=key,group=index["group"],cohorts=index["cohorts"],v=iou_counts(values,cohort_classes)))
    for i,r in enumerate(read(OUT/"external-index.json")["rows"]):
        row=source_rows[r["id"]]
        with Image.open(DATA/row["rgb"]) as im:
            rgb=np.array(im.convert("RGB"))
        evaluate("external",r,rgb,test.native_truth(row))
        if (i+1)%80==0:
            print("repaired native predictions",i+1,flush=True)
    syn={r["id"]:r for r in read(OUT/"synthetic-index.json")["rows"]}
    for _,key,g,_,rgb,truth in j.held.scene_cases():
        if key+"_t70" not in syn:
            continue
        clean=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
        for t in (.70,.40):
            fog=j.to_rgb(j.srgb(t*j.linear_rgb(clean)+(1.-t)*.85))
            evaluate("synthetic",syn[key+f"_t{round(t*100)}"],fog,truth,clean)
    if len(batches)!=len(source_rows)+len(syn):
        raise ValueError("Missing native predictions")
    summary={c:compute_summary([r for r in batches if c in r["cohorts"]]) for c in ("fog","mixed","synthetic","t70","t40")}
    source={str(s):read((D/f"queries-{s}.pth").with_suffix(".json")) for s in SEEDS}
    write(D/"paired-summary.json",dict(cohorts=summary,
          source={s:dict(initial_coverage=r["previous_source"]["classes_predicted"],initial_miou=r["previous_source"]["miou"],
                         corrected_coverage=r["source_scores"][str(r["selected"])]["classes_predicted"],
                         corrected_miou=r["selected_miou"],selected_step=r["selected"]) for s,r in source.items()}))
    gallery()
    print("repaired peer",summary["fog"],flush=True)


def gallery():
    dirs=sorted((OUT/"examples").iterdir())
    headers=["Fog / adverse input","Our RGB","Dehaze-Seg-S RGB","Our labels","Dehaze-Seg-S labels","Reference labels"]
    width,height=254,190
    canvas=Image.new("RGB",(width*6,77+(height+36)*len(dirs)+51),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    small=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    draw.text((6,4),"IDD-AW  |  Same image and 512-pixel input long side",fill="#243744",font=font)
    for i,title in enumerate(headers):
        draw.text((i*width+5,43),title,fill="#243744",font=small)
    for k,p in enumerate(dirs):
        top=75+k*(height+36)
        paths=[p/"input.png",p/"restored512.png",p/"published-rgb.png",p/"weighted512.png",D/"examples"/p.name/"corrected-labels.png",p/"reference.png"]
        for i,path in enumerate(paths):
            with Image.open(path) as im:
                if i>=3:
                    ids=np.array(im);a=j.paint(ids)
                    if i==5:
                        a[ids==255]=[105,105,105]
                    im=Image.fromarray(a)
                thumb=im.resize((width-3,height),Image.Resampling.NEAREST if i>=3 else Image.Resampling.LANCZOS)
                canvas.paste(thumb,(width*i,top))
        draw.text((5,top+height+2),p.name.replace("_"," / "),fill="#243744",font=small)
    draw.text((6,canvas.height-40),"Ground truth uses the 17 shared classes. Gray = excluded. Source-adapted Dehaze-Seg-S. "
              "The dataset provides no paired clear RGB for these frames.",fill="#243744",font=small)
    canvas.save(D/"predictions.jpg",quality=93)


if __name__=="__main__":
    run()

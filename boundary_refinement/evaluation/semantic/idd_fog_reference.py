"""Check whether the complete result also beats the frozen LiDSeg backbone on precisely the same frames."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.evaluation.semantic import idd_fog_evaluate as test
from boundary_refinement.evaluation.paired import muses_transmission as j
from boundary_refinement.data.cohorts.idd_fog_data import DATA,MAP,OUT,fixed
from boundary_refinement.data.provenance import digest,read,write


def register():
    fixed(OUT/"author-reference-protocol.json",dict(script=digest(__file__),model=digest(OUT/"model-protocol.json"),
        purpose="Supplementary direct frozen backbone check. Original LiDSeg on raw image with no new adapter at 512 and "
                "at deployed fine size capped at 1536. Direct native bilinear interpolation of its logits. Same GT, native "
                "6-pixel boundary, fixed 17-class macro and paired drive bootstrap. Compare to the complete deployed or 512 "
                "weighted result. No segmentation labels used to choose a scale. For previously used MUSES clear scenes "
                "under the two registered veils also measure reference author MCT RGB at the same 512 working grid. "
                "This MCT + LiDSeg reference is two independent releases."))


def iou_counts(mat,classes):
    d=np.diagonal(mat,axis1=-2,axis2=-1)
    union=mat.sum(-1)+mat.sum(-2)-d
    return np.stack([d[...,classes],union[...,classes]],-2)


def contrasts(rows):
    groups=sorted({r["group"] for r in rows})
    parts=defaultdict(list)
    for r in rows:
        parts[r["group"]].append(r["counts"])
    data=np.stack([np.sum(parts[g],axis=0) for g in groups])
    n=len(data)
    draws=np.vstack([np.ones(n),np.random.default_rng(100481).multinomial(n,np.full(n,1/n),5000)])
    a=(draws@data.reshape(n,-1).astype(np.float64)).reshape(len(draws),*data.shape[1:])
    iou=np.divide(100*a[...,0,:],a[...,1,:],out=np.zeros_like(a[...,0,:]),where=a[...,1,:]>0).mean((-1,-2))
    values={}
    for k,metric in enumerate(("full","boundary")):
        for i,label in enumerate(("deployed","single512")):
            diff=iou[:,k,2*i]-iou[:,k,2*i+1]
            values[metric+"_"+label]=dict(weighted=float(iou[0,k,2*i]),author_raw=float(iou[0,k,2*i+1]),
                                           delta=float(diff[0]),ci95=np.quantile(diff[1:],[.025,.975]).tolist())
    return values


@torch.inference_mode()
def run():
    fit.configure_cuda();register()
    core=j.base.DualScaleFeatures()
    meta={r["id"]:r for r in read(DATA/"evaluation.json")["rows"]}
    per=[]
    src=OUT/"author-counts"
    src.mkdir(exist_ok=True)
    common=sorted(MAP.values())
    for i,r in enumerate(read(OUT/"external-index.json")["rows"]):
        row=meta[r["id"]];p=DATA/row["rgb"]
        file=src/(r["id"].replace("/","_")+".npz")
        if not file.exists():
            if digest(p)!=row["rgb_sha"]:
                raise ValueError("Changed RGB")
            truth=test.native_truth(row)
            with Image.open(p) as im:
                rgb=np.array(im.convert("RGB"))
            raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
            scoring=test.Counts(truth)
            counts=[]
            for size in (1536,512):
                h,w=rgb.shape[:2];s=min(1.,size/max(h,w))
                z=core.logits(j.resize(raw,(round(s*h),round(s*w)),antialias=True))
                labels=j.resize(z,(h,w)).argmax(1)[0]
                counts.append(np.stack(scoring.score(labels)))
            with file.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,counts=np.stack(counts))
            file.with_suffix(".part").replace(file)
        with np.load(file) as f:
            author=f["counts"]
        if digest(r["file"])!=r["sha"]:
            raise ValueError("Changed weighted result file")
        with np.load(r["file"]) as f:
            arms=f["arms"].tolist()
            joint=np.stack((f["full"],f["edge"]))
            combined=[]
            for k,label in enumerate(("weighted","weighted512")):
                c=joint[:,arms.index(label)]
                peer=np.repeat(author[k][:,None],3,axis=1)
                if not np.array_equal(c.sum(-1),peer.sum(-1)):
                    raise ValueError("Different labeled support")
                combined.extend([iou_counts(c,common),iou_counts(peer,common)])
            per.append(dict(group=r["group"],cohorts=r["cohorts"],counts=np.stack(combined,axis=1)))
        if (i+1)%60==0:
            print("raw author",i+1,flush=True)
    results={c:contrasts([r for r in per if c in r["cohorts"]]) for c in ("fog","mixed")}
    syn=read(OUT/"synthetic-index.json")["rows"]
    lookup={r["id"]:r for r in syn}
    by_scene=defaultdict(list)
    for ds,key,g,dt,rgb,truth in j.held.scene_cases():
        if ds=="new_muses_clear":
            by_scene[g].append((key,rgb))
    rgb_rows=[]
    for g,cases in by_scene.items():
        key,raw=min(cases,key=lambda r:sha256(("idd100/photo/"+r[0]).encode()).digest())
        target=torch.from_numpy(raw).cuda().permute(2,0,1)[None].float()/255.
        for t in (.70,.40):
            pk=key+f"_t{round(100*t)}"; idx=lookup[pk]
            hazy=j.to_rgb(j.srgb(t*j.linear_rgb(target)+(1.-t)*.85))
            x,size=fit.peer_input(hazy)
            x=x[...,:size[0],:size[1]]
            restored=j.road.author_rgb(core.core.rgb,x).clamp(0,1)
            ref=j.resize(target,size,antialias=True)
            mse=float((restored-ref).square().mean())
            with np.load(idx["file"]) as f:
                own=-10*np.log10(f["photo"][1,0])
            rgb_rows.append(dict(id=pk,drive=g,t=t,mct=float(-10*np.log10(mse)),weighted=float(own)))
    if len(rgb_rows)!=26:
        raise ValueError("Missing paired RGB controls")
    write(OUT/"author-reference.json",dict(scores=results,photo=rgb_rows,counts=len(per),
                                           protocol=digest(OUT/"author-reference-protocol.json")))
    print("raw reference complete",results["fog"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part",choices=("register","run"))
    if cli.parse_args().part=="run":
        run()
    else:
        register()

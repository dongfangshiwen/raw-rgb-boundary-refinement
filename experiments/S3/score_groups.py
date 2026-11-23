"""Standalone NumPy-only rebuild from per-keyframe confusion matrices."""
import argparse
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

DEFAULT=Path(__file__).resolve().parents[2]/"experiment_exports/matched-color-decoding-v104"
PAIRS={"paper":("released","bilinear","fixed_grid","gaussian","native_gaussian"),
       "fixed_grid":("bilinear","native_gaussian"),"gaussian":("fixed_grid","released"),
       "ranges":("gaussian","paper"),
       "path":("gaussian","endpoint","shifted","permuted","ranges","paper","released"),
       "paper_segfix":("paper","segfix","native_gaussian_segfix"),
       "path_segfix":("path","segfix","paper_segfix","native_gaussian_segfix")}
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def scores(rows,draws):
    shape=rows.shape[1:]
    pooled=(draws@rows.reshape(len(rows),-1)).reshape(len(draws),*shape)
    hit,union,truth=np.moveaxis(pooled,-1,0)
    ratio=np.divide(hit,union,out=np.zeros_like(hit),where=union>0)
    has=truth>0
    fixed=100*ratio.mean(-1)
    observed=100*(ratio*has).sum(-1)/has.sum(-1).clip(min=1)
    return fixed,observed

def metrics(data,names,arms,targets,comparisons=PAIRS,bootstrap_seed=10417):
    # group, target, arm, seed, class, [hits, unions, ground truth]
    valid=data[:,0,0,0,:,2].sum(-1)>0;data=data[valid].astype(float)
    names=[n for n,v in zip(names,valid) if v];n=len(data)
    if not n:return {"groups":0}
    y,obs=scores(data,np.ones((1,n)));ind=scores(data,np.eye(n))[0].mean(-1)
    lo=scores(data,np.ones((n,n))-np.eye(n))[0].mean(-1) if n>1 else None
    boot=None
    if n>=6:
        draws=np.random.default_rng(bootstrap_seed).multinomial(n,np.full(n,1/n),size=5000)
        boot=np.concatenate([scores(data,w)[0].mean(-1) for w in np.array_split(draws,50)])
    result={}
    for t,name in enumerate(targets):
        m=y[0,t].mean(-1);supported=obs[0,t].mean(-1);comp={}
        for new,controls in comparisons.items():
            a=arms.index(new)
            for c in controls:
                b=arms.index(c);d=ind[:,t,a]-ind[:,t,b]
                loo=lo[:,t,a]-lo[:,t,b] if lo is not None else None
                ci=np.quantile(boot[:,t,a]-boot[:,t,b],[.025,.975]).tolist() if boot is not None else None
                comp[new+" - "+c]=dict(pp=m[a]-m[b],ci95=ci,supported_pp=supported[a]-supported[b],
                    seed_pp=(y[0,t,a]-y[0,t,b]).tolist(),leave_one_out=[loo.min(),loo.max()] if loo is not None else None,
                    scene_win_equal_loss=[int((d>1e-12).sum()),int((abs(d)<=1e-12).sum()),int((d< -1e-12).sum())])
        result[name]=dict(scores=dict(zip(arms,m.tolist())),supported=dict(zip(arms,supported.tolist())),contrasts=comp)
    return dict(groups=n,names=names,class_support_pixels=data[:,0,0,0,:,2].sum(0).astype(int).tolist(),targets=result)

def run(root):
    panels={}
    for tag in ("new_random","new_fog","old_fog","idd","weatherproof"):
        card=read(root/(tag+"-evaluation.json"));ar=card["arms"];ts=card["targets"]
        files=sorted((root/"counts"/tag).glob("*.npz"));assert len(files)==card["frames"]
        if not files:panels[tag]={"frames":0,"groups":0,"ontologies":{}};continue
        bank=defaultdict(lambda:0);seen=set()
        for path in files:
            with np.load(path,allow_pickle=False) as f:
                assert f["lock"].item()==card["lock"] and f["id"].item() not in seen
                seen.add(f["id"].item()); group=f["group"].item()
                assert f["arms"].tolist()==ar and f["targets"].tolist()==ts and f["seeds"].tolist()==[101,211,307]
                cm=f["counts"]
                assert cm.shape==(2,2 if tag=="weatherproof" else 1,2,len(ar),3,19,19) and cm.dtype==np.int32
                gt=cm.sum(-1);assert (cm>=0).all() and (cm[:,:,1]<=cm[:,:,0]).all()
                assert (gt==gt[:1,:,:,:1,:1]).all()
                hits=cm.diagonal(axis1=-2,axis2=-1);union=gt+cm.sum(-2)-hits
                bank[group]+=np.stack((hits,union,gt),-1)
        assert len(bank)==card["groups"];names=sorted(bank); data=np.stack([bank[g] for g in names])
        sets={"primary":[k for k in range(19) if tag!="idd" or k not in (9,16)]}
        if tag=="weatherproof":sets={"primary":[0,2,10],"expanded":[0,2,8,10]}
        res={}
        for j,(ontology,cols) in enumerate(sets.items()):
            sub=np.take(data[:,:,j],cols,axis=-2)
            res[ontology]={name:metrics(sub[:,:,k],names,ar,ts) for k,name in enumerate(("full","boundary"))}
            res[ontology]["class_ids"]=cols
        panels[tag]={"frames":len(files),"groups":len(bank),"ontologies":res}
        print("Recomputed",tag,len(files),len(bank),flush=True)
    write(root/"results.json",dict(panels=panels,bootstrap_draws=5000,seed=10417,interval="paired video-prefix / drive / camera-root percentile; seed mean within each resample"))

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--root",type=Path,default=DEFAULT);run(a.parse_args().root)

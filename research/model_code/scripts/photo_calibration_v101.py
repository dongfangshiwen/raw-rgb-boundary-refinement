"""Calibrate RGB correction strength and cross-fit whole BeDDE cities; evaluate frozen on REVIDE."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts import external_tone_v99 as ex
from scripts.fit_range_v101 import OUT
from scripts.projective_ledger_v12 import digest,read
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write,configure_cuda


def features(raw,mct):
    y=(raw*raw.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
    return torch.stack((y.mean(),y.std(unbiased=False),(mct-raw).abs().mean()))

def apply_gate(raw,mct,coeff):
    f=features(raw,mct)
    c=lambda key:f.new_tensor(coeff[key])
    alpha=(c("bias")+((f-c("mean"))/c("std")*c("slopes")).sum()).sigmoid()
    return raw+alpha*(mct-raw),alpha

@torch.inference_mode()
def collect():
    mct,_=ex.road.load_source()
    rows=[]
    for ds,key,group,hazy,ref,roi in chain(ex.city_pairs(ex.city_manifest()),((*r,None) for r in ex.indoor.pairs())):
        h,w=hazy.shape[-2:]; ratio=min(1.,576/max(h,w));size=round(h*ratio),round(w*ratio)
        x,y=(ex.resize(a[None].cuda().float()/255,size,antialias=True) for a in (hazy,ref))
        mask=(torch.ones_like(y[0,0]).bool() if roi is None else
              F.interpolate(roi[None,None].cuda().float(),size,mode="nearest-exact")[0,0].bool())
        m=author_rgb(mct,x).clamp(0,1)
        d=m-x;e=x-y
        avg=lambda v:(v.mean(1)[0][mask]).double().mean()
        moments=torch.stack([avg(e.square()),avg(d*e),avg(d.square())])
        i,j=[v%4 for v in sha256((ds+key).encode()).digest()[:2]]
        patch=lambda v:v[...,i::4,j::4].detach().cpu().clone()
        row=dict(ds=ds,key=key,group=group,f=features(x,m).cpu(),mom=moments.cpu(),x=patch(x),m=patch(m),
                 y=patch(y),roi=patch(mask),valid=int(mask.sum()))
        if ds not in {r["ds"] for r in rows}:
            row["example"]=[v.cpu().clone() for v in (x,m,y)]
        rows.append(row)
        if len(rows)%64==0: print("RGB source and reference checked",len(rows),flush=True)
    assert Counter(r["ds"] for r in rows)==dict(bedde=208,revide_final=16)
    return rows

def fit(rows,conditional):
    counts=Counter(r["group"] for r in rows)
    w=torch.tensor([1/(len(counts)*counts[r["group"]]) for r in rows],device="cuda",dtype=torch.float64)
    feats=torch.stack([r["f"] for r in rows]).cuda().double()
    terms=torch.stack([r["mom"] for r in rows]).cuda()
    mu=(w[:,None]*feats).sum(0)
    sd=(w[:,None]*(feats-mu).square()).sum(0).sqrt().clamp_min(.001)
    z=(feats-mu)/sd
    coeff=torch.zeros(4 if conditional else 1,device="cuda",dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([coeff],lr=.035)
    for i in range(250):
        opt.zero_grad()
        alpha=(coeff[0]+(z @ coeff[1:] if conditional else 0)).sigmoid()
        mse=terms[:,0]+2*alpha*terms[:,1]+alpha.square()*terms[:,2]
        loss=(w*mse.clamp_min(1e-12).log()).sum()
        if conditional:loss=loss+.005*coeff[1:].square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coeff],50.,error_if_nonfinite=True);opt.step()
    return dict(bias=coeff[0].item(),slopes=coeff[1:].tolist() if conditional else [0.]*3,
                mean=mu.tolist(),std=sd.tolist(),training_cities=sorted(counts),updates=250)

def summarize(rows):
    res={}
    for ds in ("bedde","revide_final"):
        sub=[r for r in rows if r["ds"]==ds]
        groups=sorted({r["group"] for r in sub});methods=sorted({r["arm"] for r in sub})
        table=np.array([[[np.mean([r[metric] for r in sub if r["group"]==g and r["arm"]==a])
                         for metric in ("psnr","deltaE")] for a in methods] for g in groups])
        assert np.isfinite(table).all()
        avg=table.mean(0)
        rng=np.random.default_rng(100481)
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=5000)/len(groups)
        boot=(draws@table.reshape(len(groups),-1)).reshape(-1,len(methods),2)
        contrast={}
        for arm in ("conditional","scalar"):
            for comp in ("public_mct","public_pgl","paper_curve","input","scalar"):
                if arm==comp:continue
                a,b=methods.index(arm),methods.index(comp);delta=boot[:,a]-boot[:,b]
                contrast[arm+" minus "+comp]={"estimate":(avg[a]-avg[b]).tolist(),
                                                "ci95":np.quantile(delta,[.025,.975],axis=0).T.tolist()}
        res[ds]={"groups":len(groups),"images":len({r["key"] for r in sub}),"metric_order":["PSNR_dB","CIEDE2000"],
                 "scores":dict(zip(methods,avg.tolist())),"contrasts":contrast}
    return res

def run():
    configure_cuda()
    protocol=OUT/"photo-protocol.json"
    baseline_file=ex.c.ROOT/"external-individual.json"
    old=defaultdict(list)
    for r in read(baseline_file):
        if r["arm"] in ("public_mct","input","public_pgl","curve"):
            old[r["ds"],r["key"],r["arm"]].append(r)
    rows=collect()
    cities=sorted({r["group"] for r in rows if r["ds"]=="bedde"},key=lambda s:sha256(("rawguide101-photo/"+s).encode()).digest())
    assert len(cities)==23
    folds={c:i%5 for i,c in enumerate(cities)}
    models={}
    for fold in (*range(5),"all"):
        train=[r for r in rows if r["ds"]=="bedde" and folds[r["group"]]!=fold]
        for arm in ("scalar","conditional"):
            models[f"{fold}/{arm}"]=fit(train,arm=="conditional")
    atomic_write(OUT/"photo-weights.json",dict(protocol=digest(protocol),implementation=digest(__file__),folds=folds,models=models))
    results=[]
    for r in rows:
        ds,key,group=r["ds"],r["key"],r["group"]
        fold=folds[group] if ds=="bedde" else "all"
        x,m,y,mask=(r[k].cuda() for k in ("x","m","y","roi"))
        gt_lab=ex.srgb_to_lab(y)
        def record(arm,psnr,de,**extra):
            results.append(dict(ds=ds,key=key,group=group,arm=arm,psnr=psnr,deltaE=de,fold=fold,**extra))
        def stats(alpha):
            a,b,c=r["mom"].tolist()
            return float(-10*np.log10(max(1e-12,a+2*b*alpha+c*alpha**2))),float(
                ex.ciede2000_lab(ex.srgb_to_lab(x+alpha*(m-x)),gt_lab)[0][mask].mean())
        for name,alpha in (("input",0.),("public_mct",1.)):
            psnr,de=stats(alpha)
            ref=old[ds,key,name][0]
            assert ref["valid_pixels"]==r["valid"] and abs(psnr-ref["psnr"])<.002 and abs(de-ref["deltaE"])<.015
            record(name,psnr,de)
        for name in ("public_pgl","curve"):
            refs=old[ds,key,name]
            assert len(refs)==(3 if name=="curve" else 1)
            record("paper_curve" if name=="curve" else name,np.mean([a["psnr"] for a in refs]),np.mean([a["deltaE"] for a in refs]))
        for name in ("scalar","conditional"):
            net=models[f"{fold}/{name}"]
            if ds=="bedde":assert group not in net["training_cities"]
            v=(np.asarray(r["f"])-net["mean"])/net["std"]
            alpha=float(1/(1+np.exp(-(net["bias"]+np.dot(net["slopes"],v)))))
            record(name,*stats(alpha),alpha=alpha)
            if name=="conditional" and "example" in r:
                raw,base,ref=r["example"];pred,a=apply_gate(raw,base,net)
                assert abs(float(a)-alpha)<1e-4
                for tag,image in (("input",raw),("mct",base),("conditional",pred),("reference",ref)):
                    Image.fromarray(image[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()).save(OUT/f"photo-{ds}-{tag}.png")
    atomic_write(OUT/"photo-individual.json",results)
    atomic_write(OUT/"photo-results.json",dict(protocol=digest(protocol),weights=digest(OUT/"photo-weights.json"),
                 reference=digest(baseline_file),individual=digest(OUT/"photo-individual.json"),sets=summarize(results)))
    print("Completed 12 deterministic CUDA fits and 224 paired-image evaluations",flush=True)

if __name__=="__main__":
    run()

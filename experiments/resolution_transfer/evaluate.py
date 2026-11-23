"""Native-grid paired comparisons of fixed, manuscript and newly fitted RGB weights."""
import argparse
from pathlib import Path
import numpy as np
import torch

from experiments.resolution_transfer.cohort import OUT, PRIOR, ROOT, read, write, digest, stage_labels
from experiments.resolution_transfer.scale_training import METHODS, SIDES, SEEDS, image_at, range_model
from experiments.recording_gate import gate
from experiments.recording_gate import fresh_scores as prior_scores

ARMS=("released","coarse","fixed","paper",*METHODS,"shifted_shared")
TARGETS=("mask2former","segformer")
PAIRS={"fixed":("released","coarse"),"paper":("released","coarse","fixed"),
       "shared":("released","fixed","paper","low_only","high_only","shifted_shared"),
       "by_scale":("released","fixed","paper","shared","shuffled_scale")}


def panels():
    data=gate.source_rows()
    return {
        "new_bdd":[("random",r) for r in read(OUT/"new-test.json")["rows"]],
        "muses":[(s,r) for s in ("calibrate","held") for r in data[s]],
        "old_fog":[("visible_fog",r) for r in read(PRIOR/"bdd-evaluation.json")["rows"] if r["panel"]=="visible_fog"],
    }


def scoring_lock():
    card={"study":digest(OUT/"protocol.json"),"training":digest(OUT/"training.json"),"new_targets":digest(OUT/"new-test.json"),
          "native_evaluator":digest(__file__),"arms":ARMS,"seeds":SEEDS,"scales":SIDES,"targets":TARGETS,"contrasts":PAIRS,
          "confidence":"Within each split, draw as many independent group IDs with replacement as are present. For MUSES use 4+5 stratified recordings. 5000 paired draws conditional on the three source fits. Group bootstrap unavailable for fewer than six groups.",
          "negative_control":"Roll both raw RGB axes by half the working image for the mixed-resolution shared model; keep scores and donors in place."}
    # Normalize tuples for stable readback after a resumed run.
    import json
    file=OUT/"scoring-lock.json";write(file,json.loads(json.dumps(card)),True)
    return digest(file)


def save(file,**values):
    tmp=file.with_suffix(".part")
    with tmp.open("wb") as handle:np.savez_compressed(handle,**values)
    tmp.replace(file)


class Counts:
    def __init__(self,truth):
        self.shape=tuple(truth.shape)
        self.gt=truth.cpu().numpy().astype(np.int32).ravel()
        self.band=gate.boundary_band(truth).cpu().numpy().reshape(-1)
        self.valid=self.gt<19
        self.code=(self.band.astype(np.int32)*19+self.gt)*19

    def hist(self,pred,valid):
        raw=np.bincount(self.code[valid]+pred[valid],minlength=2*19*19).reshape(2,19,19)
        return np.stack((raw.sum(axis=0),raw[1])).astype(np.int32)

    def anchor(self,scores):
        self.labels=scores.argmax(1)[0].byte().cpu().numpy().ravel()
        self.base=self.hist(self.labels,self.valid)

    def score(self,scores):
        labels=scores.argmax(1)[0].byte().cpu().numpy().ravel()
        changes=(self.labels!=labels)&self.valid
        return self.base+self.hist(labels,changes)-self.hist(self.labels,changes)


@torch.inference_mode()
def one_scale(rgb,y,side,target,model,fits,paper,fixed,metric,check):
    image=image_at(rgb,side);z,official=model(image,y.shape)
    assert tuple(4*n for n in z.shape[-2:])==tuple(image.shape[-2:])
    coarse=gate.ev.resize(z,y.shape)
    if target=="segformer":assert torch.equal(coarse,official)
    metric.anchor(coarse)
    ary=np.empty((2,len(ARMS),len(SEEDS),19,19),np.int32)
    def record(arm,idx,scores):
        c=metric.score(scores)
        if check and idx in (None,0):
            reference=check.score(scores.argmax(1)[0])
            np.testing.assert_array_equal(c,np.stack(reference))
        if idx is None:ary[:,ARMS.index(arm)]=c[:,None]
        else:ary[:,ARMS.index(arm),idx]=c
    record("released",None,official);del official
    record("coarse",None,coarse)
    guide=image.mul(255).round().byte()
    st=gate.ev.prepare(z,guide)
    shift=torch.roll(guide,(image.shape[-2]//2,image.shape[-1]//2),(-2,-1))
    shifted=gate.ev.prepare(z,shift)
    old=gate.ev.legacy.guide_stencil(guide[[0,0]],z.argmax(1)[0])
    def native(d):return coarse+gate.ev.resize(d,y.shape)
    record("fixed",None,native(fixed.correction(z,st)))
    for k,seed in enumerate(SEEDS):
        record("paper",k,native(gate.ev.legacy.prediction("dual_free",paper["dual_core",seed],z,old)))
        for m in METHODS:
            net=fits[m,seed,side]
            record(m,k,native(net.correction(z,st)))
            if m=="shared":record("shifted_shared",k,native(net.correction(z,shifted)))
    assert (ary>=0).all() and (ary[1]<=ary[0]).all() and (ary.sum(-1)==ary[:,:1,:1].sum(-1)).all()
    return ary


def run():
    gate.ev.setup.configure_cuda()
    if not (OUT/"new-test.json").exists():stage_labels()
    lock=scoring_lock()
    fitted={(r["method"],r["seed"],s):range_model(r["effective"][str(s)]) for r in read(OUT/"training.json")["runs"] for s in SIDES}
    fixed=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    paper=gate.ev.original.load()
    for target in TARGETS:
        model=gate.ev.Target(target)
        for name,items in panels().items():
            directory=OUT/"counts"/target/name
            directory.mkdir(parents=True,exist_ok=True)
            for i,(split,row) in enumerate(items):
                p=directory/(row["id"]+".npz")
                if p.exists():
                    with np.load(p,allow_pickle=False) as f:assert f["lock"].item()==lock and f["id"].item()==row["id"]
                    continue
                rgb,y=gate.open_muses(row) if name=="muses" else prior_scores.load_case(row)
                counts=Counts(y)
                check=gate.ev.ev.Counts(y) if i==0 else None
                cm=np.stack([one_scale(rgb,y,s,target,model,fitted,paper,fixed,counts,check) for s in SIDES])
                save(p,counts=cm,id=row["id"],group=row["group"],split=split,arms=ARMS,seeds=SEEDS,scales=SIDES,lock=lock)
                if i==0 or (i+1)%32==0 or i+1==len(items):print(target,name,i+1,"/",len(items),flush=True)
        del model;torch.cuda.empty_cache()


def summary(cms,groups,splits):
    names=np.unique(groups)
    bank=np.stack([cms[groups==n].sum(0,dtype=np.int64) for n in names])
    truth=bank.sum(-1);hit=bank.diagonal(axis1=-2,axis2=-1)
    union=truth+bank.sum(-2)-hit
    sufficient=np.stack((hit,union,truth),axis=-1)
    def score(draws,s,k):
        a=sufficient[:,s,k]
        total=(draws@a.reshape(len(names),-1)).reshape(len(draws),*a.shape[1:])
        h,u,t=np.moveaxis(total,-1,0)
        iou=np.divide(h,u,out=np.zeros_like(h,dtype=float),where=u!=0)*100
        return iou.mean(-1),(iou*(t>0)).sum(-1)/(t>0).sum(-1).clip(min=1)
    draws=None
    if len(names)>=6:
        rng=np.random.default_rng(11321);draws=np.zeros((5000,len(names)),np.float64)
        strata=np.array([np.unique(splits[groups==n]).item() for n in names])
        for partition in sorted(set(strata)):
            ix=np.flatnonzero(strata==partition)
            draws[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),len(draws))
    scales={}
    for s,side in enumerate(SIDES):
        by_region={}
        for k,region in enumerate(("full","boundary")):
            point,sens=score(np.ones((1,len(names))),s,k)
            seed=point[0];p=seed.mean(-1)
            individual=score(np.eye(len(names)),s,k)[0].mean(-1)
            reps=None if draws is None else np.concatenate([score(d,s,k)[0].mean(-1) for d in np.array_split(draws,50)])
            comp={}
            for a,controls in PAIRS.items():
                x=ARMS.index(a)
                for b in controls:
                    q=ARMS.index(b);dist=individual[:,x]-individual[:,q]
                    ci=None if reps is None else np.quantile(reps[:,x]-reps[:,q],[.025,.975]).tolist()
                    comp[f"{a} - {b}"]=dict(pp=p[x]-p[q],ci95=ci,seed_pp=(seed[x]-seed[q]).tolist(),
                        positive_tie_negative=[int((dist>1e-10).sum()),int((abs(dist)<=1e-10).sum()),int((dist< -1e-10).sum())])
            by_region[region]=dict(scores=dict(zip(ARMS,p.tolist())),observed_scores=dict(zip(ARMS,sens[0].mean(-1).tolist())),
                reference_classes=int((truth[:,s,k,0,0].sum(0)>0).sum()),contrasts=comp)
        scales[str(side)]=by_region
    return {"frames":len(cms),"groups":len(names),"group_names":names.tolist(),"scales":scales}


def rescore():
    lock=scoring_lock();report={}
    for target in TARGETS:
        report[target]={}
        for name,items in panels().items():
            bank=[];groups=[];splits=[]
            for partition,r in items:
                file=OUT/"counts"/target/name/(r["id"]+".npz")
                with np.load(file,allow_pickle=False) as f:
                    assert f["lock"].item()==lock and f["id"].item()==r["id"] and f["group"].item()==r["group"]
                    assert f["arms"].tolist()==list(ARMS) and f["seeds"].tolist()==list(SEEDS) and f["scales"].tolist()==list(SIDES)
                    cm=f["counts"];assert cm.shape==(3,2,len(ARMS),3,19,19)
                assert (cm>=0).all() and (cm[:,1]<=cm[:,0]).all() and (cm.sum(-1)==cm[:,:,:1,:1].sum(-1)).all()
                bank.append(cm);groups.append(r["group"]);splits.append(partition)
            counts=np.stack(bank);g=np.array(groups);p=np.array(splits)
            result=summary(counts,g,p)
            if name=="muses":
                result["splits"]={s:summary(counts[p==s],g[p==s],p[p==s]) for s in np.unique(p)}
            report[target][name]=result
            print(target,name,"768 scale effect",result["scales"]["768"]["boundary"]["contrasts"]["by_scale - shared"],flush=True)
    write(OUT/"results.json",dict(lock=lock,training=digest(OUT/"training.json"),panels=report))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--rescore",action="store_true");args=ap.parse_args()
    if not args.rescore:run()
    rescore()

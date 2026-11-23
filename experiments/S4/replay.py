"""Recalculate the reported 768-pixel and projected-point contrasts from released counts.

Run in the extracted Supplementary Data S4 directory with Python 3 and NumPy.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
RECORD=ROOT/"records"
TARGETS=("mask2former","segformer")
CLASSES=(0,1,2,4,5,7,8,9,11,13,14,17,18)
SEEDS=(101,211,307)


def read(name):
    return json.loads((RECORD/name).read_text("utf-8"))


def check_input_hashes():
    for filename,expected in json.loads((ROOT/"sha256.json").read_text("utf-8")).items():
        actual=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError("Input checksum mismatch: "+filename)


def paired(grouped,labels,classes,seed):
    # Each entry stores two methods, three source seeds, and a 19x19 confusion matrix.
    names=sorted(k for k,v in grouped.items() if v[0,0].sum()>0)
    counts=np.stack([grouped[g] for g in names])
    gt=counts.sum(-1)
    hit=counts.diagonal(axis1=-1,axis2=-2)
    union=gt+counts.sum(-2)-hit
    moments=np.take(np.stack((hit,union),-1),classes,axis=-2).astype(float)
    def estimate(weights):
        v=(weights@moments.reshape(len(names),-1)).reshape(len(weights),*moments.shape[1:])
        h,u=np.moveaxis(v,-1,0)
        return 100*np.divide(h,u,out=np.zeros_like(h),where=u>0).mean((-1,-2))
    point=estimate(np.ones((1,len(names))))[0]
    draws=np.zeros((5000,len(names)))
    rng=np.random.default_rng(seed)
    strat=np.array([labels[n] for n in names])
    for group in sorted(set(strat)):
        ix=np.flatnonzero(strat==group)
        draws[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=len(draws))
    all_est=np.concatenate([estimate(batch) for batch in np.array_split(draws,40)])
    lo,hi=np.quantile(all_est[:,0]-all_est[:,1],[.025,.975])
    return np.array((point[0],point[1],point[0]-point[1],lo,hi))


def add(bank,key,value):
    if key not in bank:bank[key]=value.astype(np.int64)
    else:bank[key]+=value


def validate(name,value,record,arm,control):
    v=record["contrasts"][arm+" - "+control]
    expected=[record["scores"][arm],record["scores"][control],v["pp"],*v["ci95"]]
    np.testing.assert_allclose(value,expected,atol=1e-9,rtol=0,err_msg=name)
    print(f"{name:62s} {value[2]:+.3f} [{value[3]:+.3f}, {value[4]:+.3f}]")


def dense(target):
    reported=read("results.json")["panels"][target]
    checks=0
    for cohort in ("new_bdd","muses"):
        records=[]
        for file in sorted((ROOT/"counts"/target/cohort).glob("*.npz")):
            with np.load(file,allow_pickle=False) as v:
                assert v["seeds"].tolist()==list(SEEDS)
                records.append((v["group"].item(),v["split"].item(),v["arms"].tolist(),v["counts"][v["scales"].tolist().index(768)]))
        for region,k in (("boundary",1),("full",0)):
            pairs=[("paper","released")]
            if region=="boundary":
                pairs+=([("paper","fixed"),("shared","paper"),("shared","shifted_shared"),("by_scale","shared"),("by_scale","shuffled_scale")]
                        if cohort=="new_bdd" else [("shared","paper")])
            for a,b in pairs:
                bank={};strata={}
                for group,split,arms,cm in records:
                    add(bank,group,cm[k,[arms.index(a),arms.index(b)]])
                    strata[group]=split
                val=paired(bank,strata,list(range(19)),11321)
                validate(f"{target} {cohort} {region} {a}-{b}",val,reported[cohort]["scales"]["768"][region],a,b)
                checks+=1
    return checks


def measured(target):
    reported=read("stf-results.json")["results"][target]["all"]["endpoints"]
    banks=defaultdict(dict);strata={}
    for fn in sorted((ROOT/"measured-points"/target).glob("*.npz")):
        with np.load(fn,allow_pickle=False) as d:
            arms=d["arms"].tolist()
            pred=d["pred"][:,[arms.index("paper"),arms.index("released")]]
            gt=d["truth"];bands=d["band"];day=d["day"].item()
            shifts=d["offsets"].tolist();strata[day]="all"
        for shift,(dx,dy) in enumerate(shifts):
            for region,limit in (("all_points",3),("near6",0),("near12",1),("near24",2)):
                if shift>0 and region!="near12":continue
                key=region if shift==0 else f"{region}@{dx},{dy}"
                chosen=bands<=limit
                truth=gt[chosen].astype(int)
                cms=np.empty((2,3,19,19),np.int64)
                for a,s in np.ndindex(2,3):
                    cms[a,s]=np.bincount(19*truth+pred[shift,a,s,chosen],minlength=361).reshape(19,19)
                add(banks[key],day,cms)
    assert len(strata)==11
    for key,bank in banks.items():
        val=paired(bank,strata,CLASSES,11324)
        validate(f"{target} real fog {key} paper-released",val,reported[key],"paper","released")
    return len(banks)


def diagnostic(target):
    report=read("sampling-bridge-results.json")["datasets"]
    arms=["released","paper","native_paper","fixed","native_fixed"]
    checks=0
    for tag in ("bdd","stf"):
        rows=read("new-test.json" if tag=="bdd" else "stf-manifest.json")["rows"]
        regions=(("dense19","dense13","sampled_all","sampled_dense_band","sampled_near12") if tag=="bdd" else ("sampled_all","sampled_near12"))
        banks=defaultdict(dict);strata={}
        for row in rows:
            group=row["group"] if tag=="bdd" else row["day"]
            strata[group]="all"
            with np.load(ROOT/"bridge-counts"/target/tag/(row["id"]+".npz"),allow_pickle=False) as f:
                cm=f["counts"]
            for i,reg in enumerate(regions):
                for a,b in (("paper","released"),("native_paper","paper")):
                    add(banks[reg,a,b],group,cm[i,[arms.index(a),arms.index(b)]])
        for (reg,a,b),bank in banks.items():
            det=paired(bank,strata,list(range(19)) if reg=="dense19" else CLASSES,11329)
            validate(f"{target} {tag} {reg} {a}-{b}",det,report[tag][target][reg],a,b)
            checks+=1
    return checks


if __name__=="__main__":
    check_input_hashes()
    checked=0
    for backbone in TARGETS:
        checked+=dense(backbone)+measured(backbone)+diagnostic(backbone)
    from runpy import run_path
    kernels = run_path(str(ROOT / "kernel-ablation" / "replay.py"))["verify"](verbose=False)
    timings = run_path(str(ROOT / "inference-costs" / "replay.py"))["verify"]()
    fog = run_path(str(ROOT / "four-decoder-fog" / "replay.py"))["verify"]()
    print(f"Verified {checked} existing contrasts, {kernels} kernel contrasts, {timings} inference cost summaries and {fog} matched fog contrasts.")

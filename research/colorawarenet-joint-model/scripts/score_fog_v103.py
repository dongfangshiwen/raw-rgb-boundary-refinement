"""CPU-only reanalysis of per-frame predictions and paired photograph errors."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]/"experiment_exports/fog-boundary-validation-v103"
COMPARISONS={
    "paper":("released","gaussian","segfix"),
    "segfix":("released",),
    "gaussian_segfix":("segfix","gaussian"),
    "paper_segfix":("paper","segfix","gaussian_segfix","released"),
    "aligned":("paper_segfix","margin","shifted","paper","segfix","gaussian_segfix"),
    "margin":("paper_segfix","paper"),
    "shifted":("paper_segfix",),
}
TARGETS=("mask2former","segformer")
REGIONS=("full","boundary")
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
def digest(p):
    return sha256(p.read_bytes()).hexdigest()
def write(p,data):
    tmp=p.with_suffix(".part")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(p)
def inventories(root,card):
    rows=read(root/"bdd-manifest.json")["rows"]+read(root/"wp-manifest.json")["rows"]
    by_id={r["id"]:r for r in rows}
    bank={}
    for path in sorted((root/"counts").glob("*/*.npz")):
        with np.load(path,allow_pickle=False) as f:
            key=f["id"].item(); group=f["group"].item();cohort=path.parent.name
            assert key not in bank and f["lock"].item()==card["lock"]
            assert f["arms"].tolist()==card["arms"] and f["targets"].tolist()==list(TARGETS) and f["seeds"].tolist()==[101,211,307]
            if cohort=="fog":row={"id":key,"group":group,"cohort":"fog"}
            else:
                row=by_id.pop(key)
                assert row["group"]==group and row["cohort"]==cohort
            cm=f["counts"]; edits=f["edits"]
            assert cm.dtype==edits.dtype==np.int32 and (cm>=0).all() and (edits>=0).all()
            assert cm.shape==(2,(2 if cohort=="wp" else 1),2,len(card["arms"]),3,19,19)
            assert (cm[:,:,1]<=cm[:,:,0]).all()
            gt=cm.sum(-1)
            assert (gt==gt[:1,:,:,:1,:1]).all()
            hit=cm.diagonal(axis1=-2,axis2=-1)
            union=gt+cm.sum(-2)-hit
            correct=hit.sum(-1)
            assert np.array_equal(correct-correct[:,:,:,card["arms"].index("paper"):card["arms"].index("paper")+1],edits[...,0]-edits[...,1])
            bank[key]=(row,np.stack((hit,union,gt),-1).astype(np.int32),edits)
    assert not by_id and len(bank)==card["files"]
    return bank

def one_metric(sums,weights):
    # group, target, arm, seed, class, hit/union/GT -> draw, target, arm
    pooled=(weights@sums.reshape(len(sums),-1)).reshape(len(weights),*sums.shape[1:])
    hit,union,gt=np.moveaxis(pooled,-1,0)
    iou=np.divide(hit,union,out=np.zeros_like(hit,dtype=float),where=union>0)
    supported=gt>0
    dynamic=(iou*supported).sum(-1)/supported.sum(-1).clip(min=1)
    return (100*iou.mean(-1).mean(-1),100*dynamic.mean(-1),100*iou.mean(-2))

def describe(d,l):
    return dict(ci95=np.quantile(d,[.025,.975]).tolist() if d is not None else None,
                leave_one_out=[float(l.min()),float(l.max())] if l is not None else None)
def summary(groups,edits,cols,arms):
    group_ids=sorted(groups)
    data=np.stack([groups[g] for g in group_ids]).astype(float)
    corrections=np.stack([edits[g] for g in group_ids]).sum(0)
    answer={}
    rgb_match={}
    for k,reg in enumerate(REGIONS):
        sub=np.take(data[:,:,k],cols,axis=-2)
        mask=sub[:,0,0,0,:,2].sum(-1)>0
        sub=sub[mask]
        names=[g for g,v in zip(group_ids,mask) if v]
        n=len(sub)
        if not n:
            answer[reg]={"informative_groups":0};continue
        point,sensitive,per=one_metric(sub,np.ones((1,n)))
        which=np.ones((n,n))-np.eye(n)
        loo=one_metric(sub,which)[0] if n>1 else None
        boot=None
        if n>=6:
            draws=np.random.default_rng(10317).multinomial(n,np.full(n,1/n),size=5000)
            boot=np.concatenate([one_metric(sub,w)[0] for w in np.array_split(draws,40)])
        individual=one_metric(sub,np.eye(n))[0]
        metrics={}
        for t,target in enumerate(TARGETS):
            contrasts={}
            for a,list_b in COMPARISONS.items():
                i=arms.index(a)
                for b in list_b:
                    j=arms.index(b)
                    extra=describe(boot[:,t,i]-boot[:,t,j] if boot is not None else None,loo[:,t,i]-loo[:,t,j] if loo is not None else None)
                    contrasts[a+" minus "+b]=dict(pp=point[0,t,i]-point[0,t,j],supported_pp=sensitive[0,t,i]-sensitive[0,t,j],**extra)
            metrics[target]=dict(scores=dict(zip(arms,point[0,t].tolist())),supported_scores=dict(zip(arms,sensitive[0,t].tolist())),
                per_class=dict(zip(arms,per[0,t].tolist())),contrasts=contrasts,
                changes_vs_paper=dict(zip(arms,corrections[t,k].mean(1).tolist())))
        pixels=sub[:,0,0,0,:,2].sum(0).astype(int)
        answer[reg]=dict(informative_groups=n,class_ids=cols,ground_truth_pixels=pixels.tolist(),targets=metrics)
        rgb_match[reg]={"names":names,"sem":individual}
    return answer,rgb_match

def photo_results(rows,names,sem,arms):
    modes=("input","mct","scalar","pgl")
    grouped=defaultdict(list)
    for r in rows:grouped[r["group"]].append(r)
    vals=np.array([[np.mean([r["psnr"][a] for r in grouped[g]]) for a in modes] for g in names])
    n=len(names);mean=vals.mean(0)
    draws=np.random.default_rng(10317).multinomial(n,np.full(n,1/n),size=5000)/n if n>=6 else None
    cons={}
    for j in (0,1,3):
        d=vals[:,2]-vals[:,j]
        cons["scalar minus "+modes[j]]=dict(dB=float(d.mean()),ci95=np.quantile(draws@d,[.025,.975]).tolist() if draws is not None else None)
    ret=dict(groups=n,scores=dict(zip(modes,mean.tolist())),contrasts=cons)
    if sem is not None:
        joint={}
        for t,model in enumerate(TARGETS):
            for a in ("paper","paper_segfix"):
                for b,photo in (("released","input"),("segfix","mct")):
                    delta=sem[:,t,arms.index(a)]-sem[:,t,arms.index(b)]
                    photo_delta=vals[:,2]-vals[:,modes.index(photo)]
                    good=(delta>0)&(photo_delta>0)
                    joint[f"{model}/{a}+scalar vs {b}+{photo}"]=dict(
                        both_strictly_positive=int(good.sum()),rgb_only=int(((delta<=0)&(photo_delta>0)).sum()),
                        semantic_only=int(((delta>0)&(photo_delta<=0)).sum()),neither=int(((delta<=0)&(photo_delta<=0)).sum()),
                        per_group={g:[float(x),float(y)] for g,x,y in zip(names,delta,photo_delta)})
        ret["joint_counts"]=joint
    return ret

def run(root):
    card=read(root/"evaluation.json")
    for k,v in card["inputs"].items():assert digest(root/k)==v
    bank=inventories(root,card)
    photos=read(root/"rgb-individual.json")
    assert photos["manifest"]==digest(root/"wp-manifest.json")
    photomap={r["id"]:r for r in photos["rows"]}
    assert set(photomap)=={k for k,(r,*_) in bank.items() if r["cohort"]=="wp"}
    for r in photomap.values():
        assert all(abs(-10*np.log10(max(1e-12,r["mse"][a]))-v)<1e-9 for a,v in r["psnr"].items())
    visually=set(read(root/"bdd-visual-lock.json")["ids"])
    strata={"bdd_tagged":lambda r:r["cohort"]=="bdd","bdd_visible":lambda r:r["id"] in visually,
        "wp_validation":lambda r:r["cohort"]=="wp" and r["split"]=="test_scenes",
        "wp_fogmix":lambda r:r["cohort"]=="wp" and r["weather"] in ("fog","mixed"),
        "wp_fog":lambda r:r["cohort"]=="wp" and r["weather"]=="fog",
        "wp_mixed":lambda r:r["cohort"]=="wp" and r["weather"]=="mixed",
        "idd_seen_drives":lambda r:r["cohort"]=="fog"}
    all_results={}
    for tag,selection in strata.items():
        subset=[(r,v,e) for r,v,e in bank.values() if selection(r)]
        cols=([0,1] if tag.startswith("bdd") else ([0,2,10] if tag.startswith("wp") else [c for c in range(19) if c not in (9,16)]))
        options=[("primary",cols,0)]
        if tag.startswith("wp"):options.append(("include_tree",[0,2,10,8],1))
        if tag.startswith("bdd"):options.append(("road_foreground",[1],0))
        section={}
        nr=len({r["group"] for r,_,_ in subset})
        for ontology,classes,ix in options:
            groups={};edits={}
            for r,v,e in subset:
                g=r["group"]
                if g not in groups:
                    groups[g]=np.zeros_like(v[:,ix],dtype=np.int64)
                    edits[g]=np.zeros_like(e[:,ix],dtype=np.int64)
                groups[g]+=v[:,ix];edits[g]+=e[:,ix]
            score,paired=summary(groups,edits,classes,card["arms"])
            if tag.startswith("wp"):
                different=[photomap[r["id"]] for r,_,_ in subset]
                if ontology=="primary":
                    section["rgb_all"]=photo_results(different,sorted(groups),None,card["arms"])
                for reg,cmp in paired.items():
                    score[reg]["rgb_on_identical_groups"]=photo_results(different,cmp["names"],cmp["sem"],card["arms"])
            section[ontology]=score
        all_results[tag]=dict(frames=len(subset),groups=nr,**section)
        print("Rescored",tag,len(subset),nr,flush=True)
    write(root/"results.json",dict(lock=card["lock"],draws=5000,bootstrap_seed=10317,
        effect_unit="segmentation percentage points; RGB dB",change_order=["rescue","harm","changed_both_wrong"],
        strata=all_results))

def rescore_dense(root):
    card=read(root/"evaluation.json")
    rows=read(root/"manifest.json")["rows"]
    arms=card["arms"]
    groups={};edits={}
    assert card["files"]==len(rows)==len({r["group"] for r in rows})
    for row in rows:
        with np.load(root/"counts"/(row["id"]+".npz"),allow_pickle=False) as d:
            assert (d["id"].item(),d["group"].item(),d["lock"].item())==(row["id"],row["group"],card["lock"])
            assert d["arms"].tolist()==arms and d["seeds"].tolist()==[101,211,307] and d["targets"].tolist()==list(TARGETS)
            c=d["counts"]
            assert c.shape==(2,2,len(arms),3,19,19) and (c>=0).all() and (c[:,1]<=c[:,0]).all()
            gt=c.sum(-1)
            assert (gt==gt[:1,:,:1,:1]).all()
            hit=c.diagonal(axis1=-2,axis2=-1)
            groups[row["group"]]=np.stack((hit,gt+c.sum(-2)-hit,gt),axis=-1)
            delta=d["edits"];correct=hit.sum(-1);i=arms.index("paper")
            assert np.array_equal(correct-correct[:,:,i:i+1],delta[...,0]-delta[...,1])
            edits[row["group"]]=delta
    results=summary(groups,edits,list(range(19)),arms)[0] if rows else {}
    write(root/"results.json",dict(frames=len(rows),groups=len(groups),lock=card["lock"],primary=results))

def rescore_manuscript_rgb(root):
    extra=read(root/"manuscript-rgb-individual.json")
    assert extra["protocol"]==digest(root/"manuscript-rgb-protocol.json")
    rows={r["id"]:r for r in read(root/"rgb-individual.json")["rows"]}
    assert len(rows)==len(extra["rows"]) and set(rows)=={r["id"] for r in extra["rows"]}
    for r in extra["rows"]:
        q=rows[r["id"]]
        assert q["group"]==r["group"] and set(r["seeds"])=={"32003","32017","32029"}
        for seed,v in r["seeds"].items():
            expected=float(-10*np.log10(max(1e-12,v["mse"])))
            assert abs(expected-v["psnr"])<1e-8
        q["psnr"]["manuscript"]=float(np.mean([v["psnr"] for v in r["seeds"].values()]))
    reference=read(root/"results.json")
    choice={"wp_validation":lambda r:r["split"]=="test_scenes",
            "wp_fogmix":lambda r:r["weather"] in ("fog","mixed"),"wp_fog":lambda r:r["weather"]=="fog"}
    res={}
    for label,pick in choice.items():
        selected=[v for v in rows.values() if pick(v)]
        members=defaultdict(list)
        for v in selected:members[v["group"]].append(v)
        region=reference["strata"][label]["primary"]["boundary"]["rgb_on_identical_groups"]["joint_counts"]
        matching=next(iter(region.values()))["per_group"]
        part={}
        for name,names in (("all",sorted(members)),("matched_boundary",sorted(matching))):
            methods=("input","mct","pgl","scalar","manuscript")
            vals=np.array([[np.mean([v["psnr"][k] for v in members[g]]) for k in methods] for g in names])
            n=len(names);mean=vals.mean(0)
            draws=np.random.default_rng(10317).multinomial(n,np.full(n,1/n),size=5000)/n if n>=6 else None
            contrasts={}
            for j,k in enumerate(methods[:-1]):
                diff=vals[:,-1]-vals[:,j]
                contrasts["manuscript minus "+k]=dict(dB=float(diff.mean()),ci95=np.quantile(draws@diff,[.025,.975]).tolist() if draws is not None else None)
            part[name]=dict(groups=n,scores=dict(zip(methods,mean.tolist())),contrasts=contrasts)
            if name=="matched_boundary":
                counts={}
                for title,p in region.items():
                    base=title.split("+")[-1]
                    rgb=vals[:,-1]-vals[:,methods.index(base)]
                    sem=np.array([p["per_group"][g][0] for g in names])
                    counts[title.replace("+scalar","+manuscript")]=dict(both=int(((rgb>0)&(sem>0)).sum()),
                           rgb_only=int(((rgb>0)&(sem<=0)).sum()),semantic_only=int(((rgb<=0)&(sem>0)).sum()),
                           neither=int(((rgb<=0)&(sem<=0)).sum()),per_group={g:[float(a),float(b)] for g,a,b in zip(names,sem,rgb)})
                part[name]["joint_counts"]=counts
        res[label]=part
    write(root/"manuscript-rgb-results.json",dict(individual=digest(root/"manuscript-rgb-individual.json"),strata=res))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    selected=parser.parse_args().root
    run(selected)
    if (selected/"dense-roads/evaluation.json").is_file():rescore_dense(selected/"dense-roads")
    if (selected/"manuscript-rgb-individual.json").is_file():rescore_manuscript_rgb(selected)

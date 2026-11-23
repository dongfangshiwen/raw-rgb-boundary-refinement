"""CPU-only reanalysis of raw per-frame confusion matrices (NumPy only)."""
import argparse
from collections import Counter
import json
from hashlib import sha256
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CLASSES_IDD=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,18]
A="bilinear_bounded4"
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def checksum(p):return sha256(p.read_bytes()).hexdigest()

def verify(target,cohort):
    home=HERE/f"counts-{target}-{cohort}"
    rows=sorted(home.glob("*.npz"))
    full,members,groups,weather=[],[],[],[]
    for p in rows:
        with np.load(p,allow_pickle=False) as main,np.load(HERE/(home.name+"-peer")/p.name,allow_pickle=False) as peer:
            for key in ("id","group","weather","seeds","lock"):
                np.testing.assert_array_equal(main[key],peer[key])
            arms=main["arms"].tolist()+peer["arms"].tolist()
            packets=[main["counts"],peer["counts"]]
            if cohort in ("bddval","fogreserve"):
                with np.load(HERE/(home.name+"-axis")/p.name,allow_pickle=False) as axis:
                    for key in ("id","group","seeds"):np.testing.assert_array_equal(main[key],axis[key])
                    assert axis["lock"].item()==checksum(HERE/"axis-weights.json")
                    arms.extend(axis["arms"].tolist());packets.append(axis["counts"])
            mat=np.concatenate(packets,axis=1)
            assert mat.dtype==np.int32 and np.all(mat>=0) and np.all(mat[1]<=mat[0])
            assert np.all(mat.sum(-1)==mat[:,:1,:1].sum(-1))
            members.append(main["id"].item());groups.append(main["group"].item());weather.append(main["weather"].item())
            full.append(mat)
    assert len(set(members))==len(members) and len(rows)>0
    if cohort=="idd":
        assert len(members)==146 and len(set(groups))==24
        orig=HERE.parent/"segformer-transfer-review-2026-11-20"
        if target=="mask2former":orig=orig/"mask2former"
        orig=orig/"fog.npz"
        if orig.exists():
            with np.load(orig,allow_pickle=False) as prev:
                old_ids=prev["ids"].tolist();old_arms=prev["arms"].tolist()
                old=prev["counts"]
                for i,key in enumerate(members):
                    k=old_ids.index(key)
                    for name,other in (("paper_fit","learned_gated"),("fixed_four","default_gated"),
                                   ("released",("released_mask2former" if target=="mask2former" else "segformer_bilinear"))):
                        delta=np.abs(old[k,:,old_arms.index(other)]-full[i][:,arms.index(name)])
                        # Different but mathematically equivalent arithmetic can change a tie at a pixel.
                        if name=="released":assert not delta.any(),(key,name,delta.max())
                        else:assert delta.max()<=12,(key,name,delta.max())
    else:
        m=read(HERE/f"{cohort}-manifest.json")["rows"]
        assert set(members)=={r["id"] for r in m}
    return arms,np.stack(full),np.asarray(groups),np.asarray(weather)

def analyze(arms,mats,groups,classes,descriptive=False):
    drives=np.unique(groups);n=len(drives)
    by=np.stack([mats[groups==g].sum(0,dtype=np.int64) for g in drives])
    hit=by.diagonal(axis1=-2,axis2=-1)
    unions=by.sum(-1)+by.sum(-2)-hit
    stats=np.stack([hit[...,classes],unions[...,classes]],-1)
    packed=stats.reshape(n,-1).astype(np.float64)
    def aggregate(weights):
        tally=(weights@packed).reshape(len(weights),*stats.shape[1:])
        return 100*np.divide(tally[...,0],tally[...,1],out=np.zeros_like(tally[...,0]),where=tally[...,1]>0).mean(axis=(-1,-2))
    pp=aggregate(np.ones((1,n)))[0]
    result={"frames":len(mats),"groups":n,"scores":{method:{"full":pp[0,j],"boundary":pp[1,j]} for j,method in enumerate(arms)},
            "CI":"paired cluster percentile, 5000 replicate draws; fixed classes and source models"}
    pairs=[(A,b) for b in arms if b!=A]+[(a,"released") for a in ("paper_fit","fixed_four","fixed_jbu","anyup")]+[
        ("fixed_four","fixed_jbu"),("paper_fit","fixed_four"),("fixed_jbu","gaussian_gaussian")]
    if "class_axis" in arms:
        pairs.extend(("class_axis",b) for b in ("rgb_axis","luma_axis","fixed_four","fixed_jbu",A,"released"))
    ix=[(arms.index(a),arms.index(b)) for a,b in pairs]
    if n>1 and not descriptive:
        rng=np.random.default_rng(100481)
        draws=rng.multinomial(n,np.ones(n)/n,size=5000)
        boot=np.concatenate([aggregate(part) for part in np.array_split(draws,50)])
    results={}
    for (a,b),(i,j) in zip(pairs,ix):
        results[a+" minus "+b]={}
        for k,metric in enumerate(("full","boundary")):
            quant=np.quantile(boot[:,k,i]-boot[:,k,j],[.025,.975]).tolist() if n>1 and not descriptive else None
            results[a+" minus "+b][metric]={"pp":float(pp[k,i]-pp[k,j]),"ci95":quant}
    result["contrasts"]=results
    return result

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("cohorts",nargs="*",default=["bdd","idd","bddval","fogreserve"])
    args=cli.parse_args()
    outcomes={}
    for s in args.cohorts:
        outcomes[s]={}
        if s=="fogreserve" and not read(HERE/"fogreserve-manifest.json")["rows"]:
            outcomes[s]["status"]="No confirmed clips"
            continue
        for target in ("mask2former","segformer"):
            arms,mats,groups,weather=verify(target,s)
            labels=CLASSES_IDD if s=="idd" else list(range(19))
            frame_groups=groups
            if s=="bdd":
                groups=np.asarray([key.partition("-")[0] for key in groups])
            effects={"all":analyze(arms,mats,groups,labels,s=="fogreserve" and len(set(groups))<8)}
            if s=="bdd":
                effects["original_frame_resampling"]=analyze(arms,mats,frame_groups,labels)
                old={k.split("-")[0] for k in read(HERE/"bdd-selection.json")["excluded_previously_used_bdd_ids"]}
                fresh=~np.isin(groups,list(old))
                effects["historical_prefix_overlap_removed"]={"excluded_frames":int((~fresh).sum()),
                    "analysis":analyze(arms,mats[fresh],groups[fresh],labels) if not fresh.all() else "identical to all"}
                for attr in np.unique(weather):
                    selection=weather==attr
                    effects[str(attr)]=analyze(arms,mats[selection],groups[selection],labels)
            outcomes[s][target]=effects
    path=HERE/"semantic-results.json";dest=path.with_suffix(".tmp")
    dest.write_text(json.dumps(outcomes,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");dest.replace(path)
    print("Verified paired matrices and summarized",list(outcomes))
if __name__=="__main__":main()

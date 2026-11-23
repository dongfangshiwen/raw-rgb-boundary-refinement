"""CPU reanalysis of the two-output stress test, without model weights."""
from boundary_refinement.paths import WORKSPACE
from pathlib import Path
import argparse
import json
import numpy as np

HERE=(WORKSPACE / "experiment_exports" / "native-guide-segfix-2026-11-20")
BASELINES={
    "input + released":("input","released"),
    "MCT + SegFix":("mct","segfix"),
    "PGL + SegFix":("pgl","segfix"),
    "local Dehaze-Seg-S":("dehaze_seg_s_local","dehaze_seg_s_local"),
}

def read(p):
    return json.loads(p.read_text(encoding="utf-8"))

def run(manuscript=False):
    note=read(HERE/"paired-evaluation.json")
    names,sem=note["photo"],list(note["semantic"])
    if manuscript:
        extra=read(HERE/"paired-manuscript/evaluation.json")
        assert extra["parent"]==note["lock"]
        sem.append("paper_512")
    photos,counts,ids=[],[],set()
    registered={r["id"]:r["group"] for r in read(HERE/"manifest.json")["rows"] if r["cohort"]=="bdd"}
    assert len(set(registered.values()))==len(registered)==128
    for p in sorted((HERE/"paired-counts").glob("*.npz")):
        with np.load(p,allow_pickle=False) as data:
            key=data["id"].item()
            assert key not in ids and data["group"].item()==registered[key]
            assert data["lock"].item()==note["lock"] and data["seeds"].tolist()==note["seeds"]
            assert data["photo"].tolist()==names and data["semantic"].tolist()==note["semantic"]
            np.testing.assert_array_equal(data["transmission"],note["conditions"])
            photos.append(data["psnr"])
            mat=data["counts"]
            if manuscript:
                with np.load(HERE/"paired-manuscript"/p.name,allow_pickle=False) as ext:
                    assert ext["id"].item()==key and ext["group"].item()==registered[key] and ext["lock"].item()==extra["lock"]
                    mat=np.concatenate((mat,ext["counts"][:,None]),axis=1)
            assert (mat>=0).all() and (mat[:,:,1]<=mat[:,:,0]).all() and mat.dtype==np.int32
            assert (mat.sum(-1)==mat[:1,:1,:,:1].sum(-1)).all()
            counts.append(mat)
            ids.add(key)
    assert ids==set(registered)
    mat=np.stack(counts)
    photos=np.stack(photos)
    assert np.isfinite(photos).all()
    n=len(mat)
    hit=mat.diagonal(axis1=-2,axis2=-1)
    union=mat.sum(-2)+mat.sum(-1)-hit
    gt_present=mat.sum(-1)>0
    with np.errstate(divide="ignore",invalid="ignore"):
        frame_iou=np.divide(hit,union,out=np.zeros_like(hit,dtype=np.float64),where=union>0)
    perframe=100*(frame_iou*gt_present).sum(-1)/gt_present.sum(-1).clip(1)
    perframe=perframe.mean(-1)
    packed=np.stack((hit,union),-1).reshape(n,-1).astype(np.float64)
    shape=hit.shape[1:]
    rgb=photos.reshape(n,-1)
    def group_scores(weights):
        summed=(weights@packed).reshape(len(weights),*shape,2)
        h,u=summed[...,0],summed[...,1]
        m=100*np.divide(h,u,out=np.zeros_like(h),where=u>0).mean((-1,-2))
        p=(weights@rgb).reshape(len(weights),*photos.shape[1:])/n
        return p,m
    pp,mm=(a[0] for a in group_scores(np.ones((1,n))))
    draws=np.random.default_rng(10212).multinomial(n,np.full(n,1/n),size=5000)
    bs=[group_scores(w) for w in np.array_split(draws,100)]
    pb,mb=(np.concatenate([s[i] for s in bs]) for i in (0,1))
    ar,am=names.index("scalar"),sem.index("paper_512" if manuscript else "fixed_native")
    result={"frames":n,"conditions":{},"example":note["example"],"alpha":note["alpha"],
            "unit":"one BDD filename prefix per image, shared across the two conditions",
            "CI":"paired pointwise percentile, 5000 replicates, seed 10212"}
    def estimate(point,samples):
        return {"delta":float(point),"ci95":np.quantile(samples,[.025,.975]).tolist()}
    for level,t in enumerate(note["conditions"]):
        item={"rgb_psnr":dict(zip(names,pp[level].tolist())),
              "semantic":{a:dict(zip(("full","boundary"),mm[level,j].tolist())) for j,a in enumerate(sem)},
              "comparisons":{}}
        for label,(photo,method) in BASELINES.items():
            br,bm=names.index(photo),sem.index(method)
            deltas=photos[:,level,ar]-photos[:,level,br]
            delta_edges=perframe[:,level,am,1]-perframe[:,level,bm,1]
            both=(deltas>0)&(delta_edges>0)
            item["comparisons"][label]={
                "psnr":estimate(pp[level,ar]-pp[level,br],pb[:,level,ar]-pb[:,level,br]),
                "full":estimate(mm[level,am,0]-mm[level,bm,0],mb[:,level,am,0]-mb[:,level,bm,0]),
                "boundary":estimate(mm[level,am,1]-mm[level,bm,1],mb[:,level,am,1]-mb[:,level,bm,1]),
                "both_positive_images":int(both.sum()),
                "psnr_positive_images":int((deltas>0).sum()),
                "boundary_positive_images":int((delta_edges>0).sum())}
        other=sem.index("matched_gaussian_fixed")
        item["candidate_minus_gaussian"]={metric:estimate(mm[level,am,k]-mm[level,other,k],
            mb[:,level,am,k]-mb[:,level,other,k]) for k,metric in enumerate(("full","boundary"))}
        result["conditions"][str(t)]=item
    filename="paired-manuscript-results.json" if manuscript else "paired-results.json"
    (HERE/filename).write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Verified paired RGB and native semantic endpoints 128 x 2.")

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--manuscript",action="store_true")
    run(cli.parse_args().manuscript)

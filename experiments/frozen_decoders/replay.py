"""Recompute the matched IDD-AW fog results from frame-level native confusion matrices; NumPy only."""
from boundary_refinement.paths import WORKSPACE
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_RECORDS = WORKSPACE / "experiment_exports/JEI-four-decoder-check-2026-11-23"
ARMS = ["bilinear", "gaussian_jbu", "fixed_four", "source_fit"]
REGIONS = ["whole_image", "boundary"]
PAIRS = [(3,1),(2,1),(3,2),(3,0),(2,0),(1,0)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compute(root=DEFAULT_RECORDS):
    plan = json.loads((root/"protocol.json").read_text("utf8"))
    output = {"cohort":"146 previously studied IDD-AW fog frames from 24 drives",
              "protocol_sha256":sha(root/"protocol.json"),"targets":{}}
    for target in ("mask2former","segformer"):
        path = root/f"counts-{target}.npz"
        with np.load(path, allow_pickle=False) as pk:
            assert pk["lock"].item()==output["protocol_sha256"] and int(pk["complete"])==146
            assert pk["arms"].tolist()==ARMS and pk["seeds"].tolist()==plan["seeds"]
            assert pk["ids"].tolist()==[r["id"] for r in plan["rows"]]
            assert pk["groups"].tolist()==[r["group"] for r in plan["rows"]]
            cm=pk["counts"]; labels=pk["groups"]
        drives=np.unique(labels)
        assert len(drives)==24
        grouped=np.stack([cm[labels==g].sum(0,dtype=np.int64) for g in drives])
        np.testing.assert_array_equal(grouped.sum(-1),np.broadcast_to(grouped[:,:,:1,:1].sum(-1),grouped.sum(-1).shape))
        diag=grouped.diagonal(axis1=-2,axis2=-1)
        unions=grouped.sum(-2)+grouped.sum(-1)-diag
        key=np.take(np.stack((diag,unions),axis=-1),plan["classes"],axis=-2)
        mat=key.reshape(len(drives),-1).astype(float)
        def metric(w):
            vals=(w @ mat).reshape(len(w),*key.shape[1:])
            h,u=vals[...,0],vals[...,1]
            return 100*np.divide(h,u,out=np.zeros_like(h),where=u>0).mean(-1).mean(-1)
        point=metric(np.ones((1,len(drives))))[0]
        weights=np.random.default_rng(11221).multinomial(len(drives),np.full(len(drives),1/len(drives)),size=5000)
        boot=np.concatenate([metric(a) for a in np.array_split(weights,50)])
        areas={}
        for i,reg in enumerate(REGIONS):
            contrasts={}
            for left,right in PAIRS:
                v=boot[:,i,left]-boot[:,i,right]
                contrasts[f"{ARMS[left]}_minus_{ARMS[right]}"]={"pp":float(point[i,left]-point[i,right]),
                    "ci95":np.quantile(v,[.025,.975]).tolist()}
            areas[reg]={"miou_percent":dict(zip(ARMS,point[i].tolist())),"paired":contrasts}
        output["targets"][target]={"counts_sha256":sha(path),"regions":areas}
    return output

def verify(root=DEFAULT_RECORDS):
    result=compute(root)
    reference=json.loads((root/"results.json").read_text("utf8"))
    for target,data in result["targets"].items():
        assert data["counts_sha256"]==reference["targets"][target]["counts_sha256"]
        for reg,metrics in data["regions"].items():
            old=reference["targets"][target]["regions"][reg]
            for arm,v in metrics["miou_percent"].items(): np.testing.assert_allclose(v,old["miou_percent"][arm],atol=1e-10,rtol=0)
            for name,row in metrics["paired"].items():
                prior=old["paired"][name]
                np.testing.assert_allclose([row["pp"],*row["ci95"]],[prior["pp"],*prior["ci95"]],rtol=0,atol=1e-10)
    return len(REGIONS)*len(PAIRS)*2

def main(default_records=DEFAULT_RECORDS):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=default_records)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    root = args.records.resolve()
    if args.record:
        dest=root/"results.json"
        if dest.exists():
            verify(root)
        else:
            dest.write_text(json.dumps(compute(root),indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(f"Verified {verify(root)} paired comparisons.")


if __name__ == "__main__":
    main()

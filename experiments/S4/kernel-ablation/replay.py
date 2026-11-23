"""Replay Table 3b and both secondary regions from native integer confusion matrices."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
CLASSES=np.asarray([0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,17,18])
TARGETS=("mask2former","segformer")
SEEDS=[101,211,307]
CONTRASTS=(
    ("original","fixed","no_correction","fixed_minus_no_correction"),
    ("original","paper","fixed","paper_minus_fixed"),
    ("refit","colour_only","fixed","colour_only"),
    ("refit","spatial_only","fixed","spatial_only"),
    ("refit","joint","fixed","joint"),
)


def read(name):
    return json.loads((ROOT/name).read_text("utf8"))


def digest(name):
    return sha256((ROOT/name).read_bytes()).hexdigest()


def score_by_recording(counts, groups, sample):
    # Include all 17 mapped classes, including zeros for an empty intersection and union.
    block=np.stack([counts[groups==key].sum(axis=0,dtype=np.int64) for key in np.unique(groups)])
    correct=np.stack([block[...,i,i] for i in CLASSES],axis=-1)
    union=np.stack([block[...,i,:].sum(axis=-1)+block[...,:,i].sum(axis=-1)-block[...,i,i]
                    for i in CLASSES],axis=-1)
    totals=np.einsum("bg,gsj->bsj",sample,union,optimize=True)
    hits=np.einsum("bg,gsj->bsj",sample,correct,optimize=True)
    return 100*np.divide(hits,totals,out=np.zeros(hits.shape,dtype=float),where=totals>0).mean(axis=2).mean(axis=1)


def verify(verbose=True):
    original,refit=read("results.json"),read("refit-results.json")
    assert refit["source_fit_count"]==9 and refit["source_updates"]==2880
    for seed in SEEDS:
        name=f"source-fit-{seed}.json"
        assert digest(name)==refit["source_weights_sha256"][str(seed)]
        record=read(name)
        assert record["updates_per_arm"]==320
        for key, num in (("colour_only",4),("spatial_only",1),("joint",5)):
            assert record["arms"][key]["trainable_scalars"]==num
            assert len(record["arms"][key]["loss"])==320
    draws=np.random.default_rng(11221).multinomial(24,np.full(24,1/24),size=5000)
    sample=np.vstack([np.ones((1,24),dtype=np.int64),draws])
    tested=0
    for target in TARGETS:
        packs={}
        for kind,prefix,expected in (("original","counts-",original["targets"][target]["sha256"]),
                         ("refit","refit-counts-",refit["targets"][target]["counts_sha256"])):
            name=prefix+target+".npz"
            assert digest(name)==expected
            with np.load(ROOT/name,allow_pickle=False) as p:
                assert p["seeds"].tolist()==SEEDS
                packs[kind]=(p["counts"].astype(np.int64),p["arms"].tolist(),p["groups"],p["ids"])
        cm1,arms1,groups,ids=packs["original"]
        cm2,arms2,g2,i2=packs["refit"]
        np.testing.assert_array_equal(ids,i2)
        np.testing.assert_array_equal(groups,g2)
        np.testing.assert_array_equal(cm1[:,:,arms1.index("fixed")],cm2[:,:,arms2.index("fixed")])
        assert len(ids)==146 and len(np.unique(groups))==24
        for region in ("boundary","whole","outside"):
            cache={}
            for kind,first,second,tag in CONTRASTS:
                cm,arms,_,_=packs[kind]
                data=cm[:,1] if region=="boundary" else cm[:,0] if region=="whole" else cm[:,0]-cm[:,1]
                for arm in (first,second):
                    if (kind,arm) not in cache:
                        cache[(kind,arm)]=score_by_recording(data[:,arms.index(arm)],groups,sample)
                diff=cache[kind,first]-cache[kind,second]
                value=[diff[0],*np.quantile(diff[1:],[.025,.975])]
                old=(original["targets"][target]["by_region"][region]["contrasts"][tag] if kind=="original" else
                     refit["targets"][target]["regions"][region]["compared_to_fixed_pp"][tag])
                np.testing.assert_allclose(value,[old["pp"],*old["ci95"]],rtol=0,atol=8e-11)
                if verbose and region=="boundary":
                    print(f"{target:12s} {tag:26s} {value[0]:+.3f} [{value[1]:+.3f}, {value[2]:+.3f}]")
                tested+=1
    assert tested==30
    return tested


if __name__=="__main__":
    print("Verified",verify(),"paired kernel contrasts across three regions.")

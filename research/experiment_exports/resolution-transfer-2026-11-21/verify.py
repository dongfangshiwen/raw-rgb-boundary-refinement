"""Replay dense and projected metrics independently of the summary implementations."""
import numpy as np
from cohort import OUT, PRIOR, digest, read, write
from evaluate import ARMS, TARGETS, panels
from scale_training import SIDES,SEEDS
from stf_data import TO_CITY
import stf_score


def score(cm,classes):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    union=cm.sum(-1)+cm.sum(-2)-tp
    return np.take(100*np.divide(tp,union,out=np.zeros_like(tp,dtype=np.float64),where=union>0),classes,axis=-1).mean(-1)


def dense():
    current=read(OUT/"results.json")["panels"];n=0; historical=0
    for t in TARGETS:
        with np.load(PRIOR/f"bdd-counts-{t}.npz",allow_pickle=False) as f:
            past=f["counts"];names=f["ids"].tolist();arms=f["arms"].tolist()
        for name,rows in panels().items():
            total=None
            for _,r in rows:
                with np.load(OUT/"counts"/t/name/(r["id"]+".npz"),allow_pickle=False) as f:cm=f["counts"]
                total=cm.astype(np.int64) if total is None else total+cm
                if name=="old_fog":
                    for ca,pa in (("released","released"),("paper","paper_fit")):
                        for old_scale,new_scale in ((0,0),(1,2)):
                            old=past[names.index(r["id"]),old_scale,:,arms.index(pa)]
                            np.testing.assert_array_equal(cm[new_scale,:,ARMS.index(ca)],old);historical+=2*len(SEEDS)
            values=score(total,np.arange(19)).mean(-1)
            for s,v in enumerate(SIDES):
                for k,reg in enumerate(("full","boundary")):
                    expected=current[t][name]["scales"][str(v)][reg]["scores"]
                    np.testing.assert_allclose(values[s,k],[expected[m] for m in ARMS],rtol=0,atol=1e-10);n+=len(ARMS)
    return n,historical


def measured():
    result=read(OUT/"stf-results.json")["results"]
    inputs=read(OUT/"stf-manifest.json")["rows"];checks=0
    columns=sorted(TO_CITY.values())
    for t in TARGETS:
        for region,ix in (("all_points",3),("near6",0),("near12",1),("near24",2)):
            for shift in (range(5) if region=="near12" else (0,)):
                selected_name=region if shift==0 else "near12@{},{}".format(*stf_score.SHIFTS[shift])
                for panel in ("all","light_fog","dense_fog"):
                    counts=np.zeros((len(stf_score.ARMS),3,19,19),np.int64)
                    for r in inputs:
                        if panel!="all" and r["weather"]!=panel:continue
                        with np.load(OUT/"measured-points"/t/(r["id"]+".npz"),allow_pickle=False) as f:
                            select=f["band"]<=ix;y=f["truth"][select].astype(int)
                            predictions=f["pred"][shift][:,:,select]
                        for a,b in np.ndindex(predictions.shape[:2]):
                            # Accumulate each truth/prediction pair directly; independent of radius-bin packing.
                            np.add.at(counts[a,b],(y,predictions[a,b]),1)
                    values=score(counts,columns).mean(-1)
                    ref=result[t][panel]["endpoints"][selected_name]
                    assert int(counts[0,0].sum())==ref["valid_points"]
                    np.testing.assert_allclose(values,[ref["scores"][m] for m in stf_score.ARMS],rtol=0,atol=1e-10)
                    checks+=len(stf_score.ARMS)
    return checks


if __name__=="__main__":
    n,old=dense();m=measured()
    r=dict(dense_pooled_values=n,independent_sparse_point_values=m,previously_archived_confusions_exact=old,
           locked_weights_sha256=digest(OUT/"training.json"),stf_fog_manifest_sha256=digest(OUT/"stf-manifest.json"),
           dense_results_sha256=digest(OUT/"results.json"),stf_results_sha256=digest(OUT/"stf-results.json"))
    write(OUT/"verification.json",r)
    print(r,flush=True)

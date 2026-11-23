"""Native real-fog scoring for the deployed RGB guide. All published heads stay frozen."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parents[1]))
from scripts import joint_tone_prototype_v99 as j
from scripts import evaluate_prototype_risk_v97 as diag
from scripts.matched_recall_v88 import capture_id

ARMS = ("author_raw","bilinear","window_fixed","curve_mean","curve_fixed")
SEEDS = j.SEEDS

def metrics(rows):
    out={}
    for field in ("counts","edge"):
        by=defaultdict(dict)
        for r in rows:
            if field not in r:
                continue
            ds=r["dataset"]
            scopes=(ds,"foggy_driving_101") if field=="counts" and ds.startswith("foggy_driving_") else (ds,)
            for scope in scopes:
                group=capture_id(ds,r["group"])
                group_cm=by[scope].setdefault(group,np.zeros((len(ARMS),len(SEEDS),19,19),dtype=np.int64))
                group_cm[ARMS.index(r["arm"]),SEEDS.index(r["seed"])] += r[field]
        out[field]={}
        for ds,groups in by.items():
            mat=np.stack(list(groups.values()))
            n=len(mat)
            draws=diag.draws(n)
            vals=[]
            for k in range(len(ARMS)):
                parts=[]
                for samples in np.array_split(draws,max(1,len(draws)//80)):
                    h=(samples@mat[:,k].reshape(n,-1)).reshape(-1,len(SEEDS),19,19)
                    d=h.diagonal(axis1=-2,axis2=-1)
                    u=h.sum(-2)+h.sum(-1)-d
                    parts.append(np.divide(100.*d,u,out=np.zeros_like(d,dtype=float),where=u>0).mean((1,2)))
                vals.append(np.concatenate(parts))
            out[field][ds]=dict(groups=n,scores={a:vs[0] for a,vs in zip(ARMS,vals)},
                curve_fixed_minus={a:diag.interval(vals[-1]-vs) for a,vs in zip(ARMS[:-1],vals[:-1])})
    return out


@torch.inference_mode()
def run():
    j.configure_cuda()
    it,expected,_,_=diag.cohort("external")
    engine=j.base.DualScaleFeatures()
    heads,curve,raws,nets=j.calib.heads(),j.calib.load(),j.base.anchors(),j.weighted.load()
    rgb_seed=j.road.SEEDS[0]
    rows,seen=[],Counter()
    example_verified=False
    for ds,key,group,_,rgb,truth in it:
        views,window,author=engine(rgb)
        grid=views.fine.shape[-2:]
        fine=j.resize(torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.,window.shape[-2:],antialias=True)
        g=j.road.geometry(fine,engine.core.rgb)
        u,a,w=j.calib.endpoints(g,engine.core.bank,heads,rgb_seed)
        torch.testing.assert_close(w,window,atol=1e-5,rtol=1e-5)
        restored=j.calib.mixture(u,a,w,"curve",curve["curve",rgb_seed])
        guides,g0=(j.base.render_guides(rgb,x,grid) for x in (restored,window))
        edge=None if ds=="foggy_driving_coarse" else truth.masked_fill(~j.boundary_band(truth),255)
        raw_author=j.resize(author,truth.shape).argmax(1)[0]
        for s in SEEDS:
            z=j.raw_anchor(raws["raw_detail",s],views)
            bilinear=j.resize(z,truth.shape)
            template=j.base.guide_stencil(guides[:0],z.argmax(1)[0])
            fixed=j.weighted.color_weights(z,"dual_core")
            pred={"author_raw":raw_author,"bilinear":bilinear.argmax(1)[0]}
            for mode,guide,coeff in (("window_fixed",g0,fixed),("curve_mean",guides,torch.full_like(fixed,1./16)),("curve_fixed",guides,fixed)):
                st=j.weighted.prepare(guide,z,"dual_core",template,coeff)
                delta=j.base.prediction("dual_free",nets["dual_core",s],z,st)
                pred[mode]=(bilinear+j.resize(delta,truth.shape)).argmax(1)[0]
            for arm,p in pred.items():
                row=dict(dataset=ds,id=key,group=group,arm=arm,seed=s,counts=j.confusion_matrix(p,truth,19).tolist())
                if edge is not None:
                    row["edge"]=j.confusion_matrix(p,edge,19).tolist()
                rows.append(row)
            if ds=="foggy_driving_fine" and key=="pedestrian_20161201_101324" and s==SEEDS[0]:
                native=np.asarray(Image.open(j.ROOT/"real-fog-example/mask_train_ids.png"))
                assert np.array_equal(native,pred["curve_fixed"].cpu().numpy())
                actual=j.to_rgb(j.resize(restored,truth.shape))
                assert np.array_equal(actual,np.asarray(Image.open(j.ROOT/"real-fog-example/restored.png")))
                assert np.array_equal(truth.cpu().byte().numpy(),np.asarray(Image.open(j.weighted.ROOT/"examples/foggy_driving_fine/truth.png")))
                example_verified=True
        seen[ds]+=1
        if sum(seen.values())%25==0:
            print("real fog",dict(seen),flush=True)
    assert seen==expected and example_verified, (seen,expected,example_verified)
    scores=metrics(rows)
    prior=j.read(diag.ROOT/"external-test.json")
    for key,pkey in (("counts","full"),("edge","edge")):
        for ds,stats in scores[key].items():
            assert abs(stats["scores"]["window_fixed"]-prior[pkey][ds]["scores"]["fixed"])<1e-8,(key,ds)
            assert abs(stats["scores"]["bilinear"]-prior[pkey][ds]["scores"]["bilinear"])<1e-8,(key,ds)
    with gzip.open(OUT/"real-joint-individual.json.gz","wt",encoding="utf-8") as stream:
        json.dump(rows,stream,separators=(",",":"))
    (OUT/"real-joint.json").write_text(json.dumps(dict(counts_seen=seen,predictions=len(rows),example_verified=example_verified,
        rgb_seed=rgb_seed,semantic_seeds=SEEDS,metrics=scores),indent=2),encoding="utf-8")
    print("verified real fog", dict(seen),len(rows),flush=True)

if __name__=="__main__":
    run()

"""Recalculate Table 8 and the camera-level WeatherProof and SegFix contrasts.
From the extracted S3 folder: python recalculate.py (NumPy required).
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from score_groups import metrics

BASE=Path(__file__).resolve().parent

def read(file):
    return json.loads((BASE/file).read_text(encoding="utf-8"))

def sufficient(cm):
    hit=cm.diagonal(axis1=-1,axis2=-2)
    gt=cm.sum(-1)
    assert cm.dtype.kind in "iu" and np.all(cm>=0)
    return np.stack((hit,gt+cm.sum(-2)-hit,gt),-1)

def grouped(stat,names,classes,arms,suite,comparisons,region):
    vals=np.take(np.stack([stat[g] for g in names]),classes,axis=-2)
    return metrics(vals[:,region,None],names,arms,[suite],comparisons,10517)

def same(x,y):
    np.testing.assert_allclose(x,y,rtol=0,atol=5e-9)

def match(a,b):
    same(a["pp"],b["pp"])
    if a["ci95"] is not None:same(a["ci95"],b["ci95"])

def main():
    road_expected=read("published/featup-results.json")["panels"]
    fixed_expected=read("published/fixed-results.json")["panels"]
    feat=read("design/featup-evaluation.json");four=read("design/fixed-evaluation.json")
    aa=feat["arms"];fa=four["arms"];out={}
    for tag in ("bdd","idd"):
        ids=read("manifests/"+tag+".json")
        classes=[i for i in range(19) if tag!="idd" or i not in (9,16)]
        out[tag]={}
        for suite in ("mask2former","segformer"):
            pooled=defaultdict(lambda:0)
            for r in ids:
                name=r["id"].replace("/","_")+".npz"
                with np.load(BASE/"featup-counts"/tag/name,allow_pickle=False) as f,np.load(BASE/"fixed-counts"/tag/name,allow_pickle=False) as prior:
                    assert f["id"].item()==prior["id"].item()==r["id"] and f["group"].item()==prior["group"].item()==r["group"]
                    assert f["lock"].item()==feat["panels"][tag]["lock"] and prior["lock"].item()==four["panels"][tag]["lock"]
                    c=f[suite]
                    for arm in ("paper","released","fixed4"):
                        np.testing.assert_array_equal(c[:,aa.index(arm)],prior[suite][:,fa.index(arm)])
                    pooled[r["group"]]+=sufficient(c)
            out[tag][suite]={}
            for reg,j in (("boundary",1),("full",0)):
                res=grouped(pooled,sorted(pooled),classes,aa,suite,{"paper":("released","fixed4","featup_frozen","featup_fit","featup_fit_gated")},j)
                contrasts=res["targets"][suite]["contrasts"]
                for k,v in contrasts.items():match(v,road_expected[tag][suite][reg]["targets"][suite]["contrasts"][k])
                out[tag][suite][reg]={k:[v["pp"],v["ci95"]] for k,v in contrasts.items()}
    wp=defaultdict(lambda:0);audit=read("design/view-audit.json");n=0
    for r in read("manifests/wp-transfer.json"):
        g=r["group"]
        if g in audit["exclude_new_groups"] or r["weather"]!="fog":continue
        with np.load(BASE/"fixed-counts/wp"/(r["id"].replace("/","_")+".npz"),allow_pickle=False) as f:
            assert f["lock"].item()==four["panels"]["wp"]["lock"] and f["id"].item()==r["id"]
            wp[audit["merge_into"].get(g,g)]+=sufficient(f["ade5"]);n+=1
    w=grouped(wp,sorted(wp),list(range(5)),fa,"ade5",{"paper":("released","fixed4")},1)
    assert (n,len(wp),w["groups"])==(43,42,34)
    out["wp_transfer"]={}
    for k,v in w["targets"]["ade5"]["contrasts"].items():
        match(v,fixed_expected["wp_fog"]["ade5"]["boundary"]["targets"]["ade5"]["contrasts"][k])
        out["wp_transfer"][k]=[v["pp"],v["ci95"]]
    out["segfix"]={}
    with np.load(BASE/"counts/segfix.npz",allow_pickle=False) as f:
        for tag in ("bdd","idd"):
            groups=f[tag+"_groups"].tolist()
            vals={g:sufficient(c) for g,c in zip(groups,f[tag])}
            cls=[i for i in range(19) if tag!="idd" or i not in (9,16)]
            v=grouped(vals,groups,cls,["gaussian_segfix","paper_segfix"],"mask2former",{"paper_segfix":("gaussian_segfix",)},1)
            got=v["targets"]["mask2former"]["contrasts"]["paper_segfix - gaussian_segfix"];expected=read("published/segfix.json")[tag]
            match(got,expected);out["segfix"][tag]=[got["pp"],got["ci95"]]
    with np.load(BASE/"counts/paired-weather.npz",allow_pickle=False) as f:
        cm=f["cm"];photos=f["psnr"];gids=f["groups"].tolist()
    assert len(gids)==63 and len(set(gids))==21
    cams=sorted(set(gids));gids=np.array(gids)
    rgb=np.stack([photos[gids==g].mean(0) for g in cams])
    # group, target, arm, seed, class, statistic, on the native reference band
    c=np.stack([cm[gids==g].sum(0) for g in cams])
    v=np.take(sufficient(c),[0,2,10],axis=-2)
    informative=v[:,0,0,0,:,2].sum(-1)>0
    assert informative.sum()==5
    def difference(x):
        a,b=x[...,0],x[...,1]
        iou=100*np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0).mean(axis=(-1,-2))
        return iou[...,1]-iou[...,0]
    scene=difference(v[informative]);pooled=difference(v[informative].sum(0))
    same(pooled,[1.2518205959169961,1.8282929297782005])
    truth=read("published/paired-photo-results.json")["strata"]["wp_fog"]
    labels=["manuscript","input","mct","pgl","scalar"]
    for j,k in enumerate(labels):
        same(rgb[:,j].mean(),truth["all"]["scores"][k]);same(rgb[informative,j].mean(),truth["matched_boundary"]["scores"][k])
    both=np.sum((scene>0)&((rgb[informative,0]-rgb[informative,1])[:,None]>0),axis=0)
    assert both.tolist()==[0,1]
    out["wp_paired"]={"all_camera_PSNR":dict(zip(labels,rgb.mean(0).tolist())),"matched_camera_PSNR":dict(zip(labels,rgb[informative].mean(0).tolist())),"boundary_gain_pp":pooled.tolist(),"simultaneous_camera_count":both.tolist()}
    print(json.dumps(out,indent=2))
    print("Verified both metrics, four road comparisons, matched SegFix, five-class transfer and paired fog outcomes.")

if __name__=="__main__":
    main()

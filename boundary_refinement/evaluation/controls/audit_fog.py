"""Independent native-mask reconstruction and WeatherProof duplicate sensitivity."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from boundary_refinement.evaluation.semantic import fog_counts as res
from boundary_refinement.data.staging.weatherproof_files import OUT,WORK,read,write,sha

def band(y):
    edges=np.zeros(y.shape,dtype=bool)
    for prev,nxt in ((np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])):
        b=(y[prev]!=y[nxt])&(y[prev]<19)&(y[nxt]<19)
        edges[prev]|=b;edges[nxt]|=b
    return maximum_filter(edges,size=13,mode="constant",cval=0)&(y<19)

def masks():
    card=read(OUT/"evaluation.json"); arms=card["arms"]
    rows={r["id"]:r for name in ("bdd-manifest","wp-manifest") for r in read(OUT/(name+".json"))["rows"]}
    dense=read(OUT/"dense-roads/manifest.json")["rows"]
    rows.update({r["id"]:r for r in dense})
    checked=[]
    for src in sorted((OUT/"examples").glob("*/*/source.json")):
        ex=read(src);d=src.parent;model=d.name;key=ex["id"];cohort=ex["cohort"]
        predicted_dir=OUT/"counts"/cohort
        if cohort=="bdd_dense":predicted_dir=OUT/"dense-roads/counts"
        y=np.array(Image.open(d/"truth-ids.png"))
        if cohort in ("bdd","wp","bdd_dense"):
            source=np.array(Image.open(WORK/rows[key]["paths"][-1]))
            if source.ndim==3:source=source[:,:,0]
            if cohort=="bdd":expect=np.where(source==255,255,np.isin(source,[1,2]).astype(np.uint8))
            elif cohort=="wp":
                lut=np.full(256,255,np.uint8);lut[[3,1,4]]=[0,2,10];expect=lut[source]
            else:expect=source
            np.testing.assert_array_equal(y,expect)
        edge=band(y)
        with np.load(predicted_dir/(key.replace("/","_")+".npz"),allow_pickle=False) as pkt:
            counts=pkt["counts"][card["targets"].index(model)]
            if cohort!="bdd_dense":counts=counts[0]
            for i,name in enumerate(arms):
                m=np.array(Image.open(d/(name+"-city-ids.png")))
                assert m.shape==y.shape and np.max(m)<19
                if cohort=="bdd":m=(m==0).astype(np.uint8)
                for k,valid in enumerate((y<19,edge)):
                    ref=np.bincount(y[valid].astype(np.int32)*19+m[valid],minlength=361).reshape(19,19)
                    np.testing.assert_array_equal(ref,counts[k,i,0])
                checked.append(f"{cohort}/{model}/{name}")
    return checked

def duplicate_check():
    c=read(OUT/"evaluation.json"); rows=read(OUT/"wp-manifest.json")["rows"]
    ids=[set(r["sha256"][j] for r in rows) for j in range(3)]
    selected=[r for r in rows if r["weather"] in ("fog","mixed")]
    seen=set(); groups={};edits={};dropped=[]
    photo={r["id"]:r for r in read(OUT/"rgb-individual.json")["rows"]}
    phot=[]
    for r in selected:
        key=(r["group"],*r["sha256"])
        if key in seen:dropped.append(r["id"]);continue
        seen.add(key);g=r["group"];phot.append(photo[r["id"]])
        with np.load(OUT/"counts/wp"/(r["id"].replace("/","_")+".npz"),allow_pickle=False) as p:
            v=p["counts"][:,0];e=p["edits"][:,0]
        gt=v.sum(-1);hit=v.diagonal(axis1=-2,axis2=-1);union=gt+v.sum(-2)-hit
        suff=np.stack((hit,union,gt),-1)
        if g not in groups:groups[g]=np.zeros_like(suff);edits[g]=np.zeros_like(e,dtype=np.int64)
        groups[g]+=suff;edits[g]+=e
    dedup,_=res.summary(groups,edits,[0,2,10],c["arms"])
    rgb=res.photo_results(phot,sorted(groups),None,c["arms"])
    return dict(files=len(rows),unique_adverse=len(ids[0]),unique_clear=len(ids[1]),unique_semantic_masks=len(ids[2]),
        repeated_within_optical_stratum=dropped,remaining=len(phot),groups=len(groups),
        boundary_informative=dedup["boundary"]["informative_groups"],
        scores=dedup["boundary"]["targets"]["mask2former"]["scores"],rgb=rgb)

def run():
    result=read(OUT/"results.json"); assert result["lock"]==read(OUT/"evaluation.json")["lock"]
    ex=OUT/"examples/wp/photographs"
    target=np.asarray(Image.open(ex/"clear.png"),dtype=float)/255
    info=read(ex/"metrics.json");errors={}
    orig=read(OUT/"manuscript-rgb-individual.json")
    info["psnr"]["manuscript"]=next(v for v in orig["rows"] if v["id"]==info["id"])["seeds"]["32003"]["psnr"]
    for mode in ("input","scalar","mct","pgl","manuscript"):
        x=np.asarray(Image.open(ex/(mode+".png")),dtype=float)/255
        value=-10*np.log10(max(1e-12,((x-target)**2).mean()))
        errors[mode]=abs(value-info["psnr"][mode])
    assert max(errors.values())<.15
    training=read(OUT/"training.json"); assert len(training["records"])==9
    for r in training["records"]:assert sha((OUT/r["weights"]).read_bytes())==r["sha256"] and np.isfinite(r["loss"]).all()
    reviewed=masks()
    write(OUT/"verification.json",dict(native_PNGs_checked=len(reviewed),native_examples=reviewed,
        boundary_radius=6,rendered_rgb_abs_error_dB=errors,training_records=9,
        checked_drivable_pixels=read(OUT/"mapping-correction.json")["total_pixels"],
        weatherproof_duplicates=duplicate_check(),historic_IDD_identical=62*2*6*3,
        result_hash=sha((OUT/"results.json").read_bytes()),dense_result_hash=sha((OUT/"dense-roads/results.json").read_bytes())))
    print("Native PNGs, two-region integer counts and photographic PSNR independently verified",len(reviewed),flush=True)
if __name__=="__main__":
    run()

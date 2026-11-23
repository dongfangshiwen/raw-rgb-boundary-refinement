"""Lock two external populations and the confirmatory incremental test before fitting."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image,ImageDraw
from scripts import holdout_path_v104 as bdd
from scripts import reserve_weatherproof_v103 as wp
from scripts import weatherproof_release_v103 as io

OUT=bdd.OUT/"manuscript-extension"
ADE="nvidia/segformer-b0-finetuned-ade-512-512"
REV="489d5cd81a0b59fab9b7ea758d3548ebe99677da"
PLAN={
    "timing":"Follow-up motivated by the completed V104 Gaussian experiment. Results of V104 are developmental for this hypothesis. Newly reserved BDD and WeatherProof camera roots have not previously been screened or scored in this project.",
    "question":"Does one scalar of aligned excess RGB path variation add boundary IoU above the exact frozen original manuscript decoder? Contrast against the identical frozen weights, extra endpoint nonlinearity, shifted path and permuted donors, all paired by seed and on identical source minibatches. Also test the archival paper and SegFix against a fitted Gaussian on the SAME lattice before a common SegFix.",
    "fit":"Freeze the three archived dual_core LiDSeg fits (temperature plus four effective range coefficients). Optimize only one nonnegative bounded scalar on the same 96 MUSES fog frames, nine group-balanced recordings, 320 steps per seed, CE on 4096 valid/boundary samples per batch. Seeds 101/211/307, same source sampler as V104, initial strength 0.4, maximum 6. No target fitting.",
    "bdd":"128 new unconstrained BDD keyframes. Exclude all V104 reserve prefixes, all earlier V103 screened and reserved prefixes, V102, V101 and historical inventory. SHA256(anchor105/<id>), one prefix per keyframe. Independent of both the V104 128 random and V104 fog-screen pool. Original release partition train; project uses it only for external testing.",
    "wp":"Use ALL available WeatherProof camera roots beyond the entire V103 registration, including previously screened yet unevaluated roots in the exclusion. One scene per root by SHA256(anchor105/<scene>); one central adverse/reference/semantic triplet. Rate fog / mixed / other / uncertain on RGB and paired clean only before segmenter inference. Primary pure fog, secondary fog+mixed. Strict road-vocabulary transfer: road, building, sky. An expanded secondary ADE20K probe with five literal class correspondences road/building/sky/tree/grass; rock-to-stone sixth only in sensitivity. All unassigned target GT pixels ignored, all unmatched predictions count as errors on scored GT. Gate and predictions use the full source label vocabulary. One RGB reader; these are generally fixed natural cameras, not driving videos.",
    "targets":"Frozen Cityscapes Mask2Former and SegFormer-B1 for both populations. Additionally, frozen released NVIDIA SegFormer-B0/ADE20K for WeatherProof, all 150 logits and no target fit. Its native parameterization differs from the two road backbones. SegFix evaluated only with road backbones. Source and evaluation point at original manuscript nine stored parameters; no duplicate-raw/class semantic guidance switch.",
    "statistics":"Fixed-class pooled boundary mIoU at six native pixels, full mIoU, GT-present sensitivity. Five thousand paired cluster bootstrap draws, one BDD prefix or WeatherProof root per unit, RNG 10517. Average three seed metrics inside each draw. No intervals below six boundary-informative groups. Primary Mask2Former must beat the frozen manuscript and all three one-parameter controls with positive lower limits on new BDD and pure fog to claim an independent cross-domain path improvement. Report incremental and composition results regardless. Other models/subgroups are secondary.",
}
def register():
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/"protocol.json"
    if p.exists():assert io.read(p)==PLAN
    else:io.write(p,PLAN)

def reserve_wp():
    register();p=OUT/"wp-reserve.json"
    if p.exists():return io.read(p)["rows"]
    index=io.index();excluded={r["group"] for r in wp.reservation()["rows"]};pool=defaultdict(lambda:defaultdict(list))
    for key in index:
        if not key.endswith("_degraded.png"):continue
        split,scene,file=key.split("/");group=wp.camera(scene);prefix=key.removesuffix("_degraded.png")
        if group in excluded or not all(prefix+"_"+s+".png" in index for s in ("gt-clean","gt-intern")):continue
        pool[group][split,scene].append(prefix)
    rows=[]
    for i,g in enumerate(sorted(pool,key=lambda x:io.sha(("anchor105/"+x).encode())),1):
        sp,sc=min(pool[g],key=lambda key:io.sha(("anchor105/"+key[1]).encode()))
        first=sorted(pool[g][sp,sc],key=lambda s:int(s.rsplit("/",1)[1]))
        rows.append(dict(id=first[len(first)//2],group=g,scene=sc,split=sp,code=f"P{i:03d}",cohort="wp"))
    assert not ({r["group"] for r in rows}&excluded)
    io.write(p,dict(rows=rows,prior_reserved_groups=len(excluded),zip=io.ID,protocol=io.sha((OUT/"protocol.json").read_bytes())))
    return rows

def weather(action):
    rows=reserve_wp();index=io.index();pairs=action=="pairs"
    if not pairs:
        ratings=io.read(OUT/"weather-lock.json")["ratings"];assert set(ratings)=={r["code"] for r in rows}
        rows=[dict(r,weather=ratings[r["code"]]) for r in rows if ratings[r["code"]] in ("fog","mixed")]
    kinds=("degraded","gt-clean") if pairs else ("degraded","gt-clean","gt-intern")
    with ThreadPoolExecutor(8) as pool:
        paths=[]
        for i,path in enumerate(pool.map(lambda r:wp.payload(r["id"],kinds,index),rows),1):
            paths.append(path)
            if i%40==0:print(action,i,len(rows),flush=True)
    if not pairs:
        out=[]
        for r,files in zip(rows,paths):
            with Image.open(files[0]) as im,Image.open(files[1]) as clean,Image.open(files[-1]) as mask:
                assert im.size==clean.size==mask.size and set(np.unique(np.asarray(mask)))<=set(range(10))|{255}
            out.append(dict(r,paths=[f.relative_to(io.WORK).as_posix() for f in files],sha256=[io.sha(f.read_bytes()) for f in files]))
        io.write(OUT/"wp.json",dict(rows=out,weather_sha=io.sha((OUT/"weather-lock.json").read_bytes())));return
    for start in range(0,len(rows),16):
        subset=rows[start:start+16];canvas=Image.new("RGB",(4*306,((len(subset)+3)//4)*148),"white");draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(subset):
            x=i%4*306;y=i//4*148;draw.text((x+2,y+1),r["code"]+"   adverse / clear",fill="black")
            for j,p in enumerate(paths[start+i]):
                with Image.open(p) as im:
                    im.thumbnail((151,128));canvas.paste(im.convert("RGB"),(x+153*j,y+16))
        canvas.save(OUT/f"weather-{start//16+1:02d}.jpg",quality=92)
    print("Blind paired weather sheets",len(rows),flush=True)

def roads():
    register();old=bdd.reserve();reg=io.read(io.WORK/"benchmark_data/bdd-fog-v73/registration.json")
    prior={r["group"] for r in old}|{r["group"] for r in bdd.previous.reserve()}|{k.split("-")[0] for k in reg["cases"]}
    for path in [*[bdd.road.OLD/(f+".json") for f in ("bdd-screen-inventory","bdd-manifest","bddval-manifest","fogreserve-manifest")],bdd.road.LAST/"manifest.json",io.OUT/"bdd-manifest.json"]:
        prior.update(r["id"].split("-")[0] for r in io.read(path)["rows"] if path!=bdd.road.LAST/"manifest.json" or r["cohort"]=="bdd")
    rev=reg["revisions"]["semantic"];source=reg["source"]["semantic"]
    meta=json.loads(bdd.road.fetch(f"https://huggingface.co/api/datasets/{source}/revision/{rev}"));assert meta["sha"]==rev
    keys={x["rfilename"] for x in meta["siblings"]};base=f"https://huggingface.co/datasets/{source}/resolve/{rev}/"
    imgs={Path(s).stem:s for s in keys if s.startswith("data/images_10k/10k/train/") and s.endswith(".jpg")}
    masks={Path(s).name.removesuffix("_train_id.png"):s for s in keys if s.startswith("data/seg_maps/labels/train/") and s.endswith("_train_id.png")}
    assert imgs.keys()==masks.keys() and len(imgs)==7000
    rows=[]
    for key in sorted(imgs,key=lambda k:io.sha(("anchor105/"+k).encode())):
        group=key.split("-")[0]
        if group in prior:continue
        prior.add(group);rows.append(dict(id=key,group=group,cohort="bdd_dense",rgb_url=base+imgs[key],mask_url=base+masks[key]))
        if len(rows)==128:break
    io.write(OUT/"bdd-reserve.json",dict(rows=rows,source=source,revision=rev))
    with ThreadPoolExecutor(12) as pool:complete=list(pool.map(bdd.stage,rows))
    io.write(OUT/"bdd.json",{"rows":complete});print("New BDD frames",len(rows),flush=True)

def ade_weights():
    p=io.WORK/"model_code/pretrained/segformer-b0-ade"
    p.mkdir(exist_ok=True,parents=True);meta=json.loads(bdd.road.fetch(f"https://huggingface.co/api/models/{ADE}/revision/{REV}?blobs=true"))
    assert meta["sha"]==REV;sizes={r["rfilename"]:r for r in meta["siblings"]};records={}
    for f in ("config.json","preprocessor_config.json","model.safetensors"):
        dest=p/f
        if not dest.exists():
            data=bdd.road.fetch(f"https://huggingface.co/{ADE}/resolve/{REV}/{f}");tmp=p/(f+".part");tmp.write_bytes(data);tmp.replace(dest)
        assert sizes[f]["size"]==dest.stat().st_size;records[f]=io.sha(dest.read_bytes())
        if f=="model.safetensors":assert records[f]==sizes[f]["lfs"]["sha256"]
    io.write(OUT/"ade-weights.json",dict(repo=ADE,revision=REV,files=records))
    print("Verified ADE20K weights",records["model.safetensors"],flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("stage",choices=("pairs","stage","roads","ade"));a=cli.parse_args().stage
    register()
    if a=="roads":roads()
    elif a=="ade":ade_weights()
    else:weather(a)

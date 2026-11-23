"""Reserve, screen and stage an additional fresh set of 19-class BDD driving keyframes."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image,ImageDraw
from boundary_refinement.data.cohorts import reserve_drivable as road
from boundary_refinement.data.staging import weatherproof_files as io

DATA=io.WORK/"benchmark_data/fog-dense-clips-v103"
R=io.OUT/"dense-roads"
RULE=dict(question="Frozen confirmatory direction of manuscript decoding followed by SegFix versus manuscript, SegFix, and matched Gaussian followed by SegFix on genuinely new BDD 19-class clips with visual fog. Both unmodified target backbones. All weights, native six-pixel band and nine arms identical to completed inference on other cohorts.",
    selection="Reserve up to 2048 never-screened semantic-training-partition filename prefixes via SHA256('dense103/'+id), one clip per prefix. Exclude all historical BDD and current official-tagged BDD prefixes. Screen top 240 input-only thumbnails by a pre-defined visibility heuristic, plus 48 comparison thumbnails selected by hash from the remainder. Enlarge visually plausible scenes; freeze strictly visible distance haze, rejecting glare, snow, rain on glass or white sky alone. Evaluate every confirmed keyframe. Single-reader daylight-enriched sample, not fog prevalence or meteorology. Historical outcomes do not set any thresholds.",
    statistics="Fixed 19-class full/band mIoU and GT-supported sensitivity. Primary Mask2Former boundary contrasts must each exceed zero with pointwise paired 95% bootstrap intervals to support stable complementarity under this protocol. Sample 5000 with replacement by clip prefix, RNG 10317. Below six clips, only descriptive results. No target tuning; no post-hoc extensions of screen.")

def reserve():
    R.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (R/"protocol.json").exists():assert io.read(R/"protocol.json")==RULE
    else:io.write(R/"protocol.json",RULE)
    if (R/"reserve.json").exists():return io.read(R/"reserve.json")["rows"]
    prior=io.read(io.WORK/"benchmark_data/bdd-fog-v73/registration.json")
    rev=prior["revisions"]["semantic"]; source=prior["source"]["semantic"]
    excluded=set(prior["cases"])
    for f in ["bdd-screen-inventory","bdd-manifest","bddval-manifest","fogreserve-manifest"]:
        excluded.update(r["id"] for r in io.read(road.OLD/(f+".json"))["rows"])
    excluded.update(r["id"] for r in io.read(road.LAST/"manifest.json")["rows"] if r["cohort"]=="bdd")
    excluded.update(r["id"] for r in io.read(io.OUT/"bdd-manifest.json")["rows"])
    seen={s.split("-")[0] for s in excluded}
    meta=json.loads(road.fetch(f"https://huggingface.co/api/datasets/{source}/revision/{rev}"))
    assert meta["sha"]==rev
    paths=[r["rfilename"] for r in meta["siblings"]]
    base=f"https://huggingface.co/datasets/{source}/resolve/{rev}/"
    def assets(folder,ext):
        return {Path(p).name.removesuffix(ext):base+p for p in paths if p.startswith(folder+"/") and p.endswith(ext)}
    pics=assets("data/images_10k/10k/train",".jpg"); masks=assets("data/seg_maps/labels/train","_train_id.png")
    assert len(pics)==7000 and pics.keys()==masks.keys()
    rows=[]
    for key in sorted(pics,key=lambda x:io.sha(("dense103/"+x).encode())):
        group=key.split("-")[0]
        if group in seen:continue
        seen.add(group);rows.append(dict(id=key,group=group,rgb_url=pics[key],mask_url=masks[key],cohort="bdd_dense"))
        if len(rows)==2048:break
    io.write(R/"reserve.json",dict(rows=rows,protocol=io.sha((R/"protocol.json").read_bytes()),rev=rev,
                                    excluded_prefixes=len({s.split("-")[0] for s in excluded})))
    return rows

def thumbnail(row):
    path=DATA/(row["id"]+"-thumb.jpg")
    if not path.exists():
        with Image.open(BytesIO(road.fetch(row["rgb_url"]))) as im:
            assert min(im.size)>32 and im.format=="JPEG",(row["id"],im.size,im.format)
            small=im.convert("RGB").resize((240,135),Image.Resampling.LANCZOS)
        tmp=path.with_suffix(".part")
        small.save(tmp,"JPEG",quality=87);tmp.replace(path)
    with Image.open(path) as im:a=np.asarray(im,dtype=float)/255
    far=a[44:78,52:188];near=a[91:116,36:204]
    def dark(region):return np.quantile(region.min(-1),.18)
    lum=far.mean(-1)
    score=1.1*dark(far)+.90*(dark(far)-dark(near))-.55*lum.std()-.40*(far.max(-1)-far.min(-1)).mean()
    return float(score)

def sheets(rows,large=False):
    pack=12 if large else 36
    cols=3 if large else 6
    cell=(380,230) if large else (204,135)
    for start in range(0,len(rows),pack):
        sheet=Image.new("RGB",(cell[0]*cols,cell[1]*int(np.ceil(min(pack,len(rows)-start)/cols))),"white")
        draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(rows[start:start+pack]):
            if large:
                p=DATA/(r["id"]+".jpg")
                if not p.exists():p.write_bytes(road.fetch(r["rgb_url"]))
            else:p=DATA/(r["id"]+"-thumb.jpg")
            with Image.open(p) as im:
                im.thumbnail((cell[0]-3,cell[1]-17));x=j%cols*cell[0];y=j//cols*cell[1]
                sheet.paste(im,(x,y+15));draw.text((x+2,y),r["code"]+"   "+r["id"],fill="black")
        sheet.save(R/(("close" if large else "sheet")+f"-{start//pack+1:02d}.jpg"),quality=90)

def screening():
    rows=reserve()
    with ThreadPoolExecutor(12) as pool:
        scores=[]
        for i,s in enumerate(pool.map(thumbnail,rows),1):
            scores.append(s)
            if i%256==0:print("Reserved RGB miniatures",i,len(rows),flush=True)
    order=sorted(range(len(rows)),key=lambda i:(-scores[i],rows[i]["id"]))
    top=order[:240]
    top+=sorted(order[240:],key=lambda i:io.sha(("dense-control/"+rows[i]["id"]).encode()))[:48]
    subset=[dict(**rows[ix],score=scores[ix],code=f"C{j:03d}") for j,ix in enumerate(top,1)]
    io.write(R/"screen.json",dict(rows=subset,all_ranks=[rows[i]["id"] for i in order],
                                       reserve=io.sha((R/"reserve.json").read_bytes())))
    sheets(subset)
    print("Blind dense-scene sheets ready",len(rows),len(subset),flush=True)

def select(stage):
    rows=io.read(R/"screen.json")["rows"]
    chosen=set(io.read(R/("weather-lock.json" if stage else "shortlist.json"))["codes"])
    chosen_rows=[r for r in rows if r["code"] in chosen]
    assert len(chosen_rows)==len(chosen)
    if not stage:
        sheets(chosen_rows,True);return
    assert chosen<=set(io.read(R/"shortlist.json")["codes"])
    old=io.read(io.OUT/"bdd-manifest.json")["rows"]; used={r["sha256"][0] for r in old}
    for r in chosen_rows:
        paths=[DATA/(r["id"]+ext) for ext in (".jpg",".png")]
        if not paths[1].exists():paths[1].write_bytes(road.fetch(r["mask_url"]))
        with Image.open(paths[0]) as im,Image.open(paths[1]) as mask:
            assert im.size==mask.size and set(np.unique(np.asarray(mask)))<=set(range(19))|{255}
        r.update(paths=[p.relative_to(io.WORK).as_posix() for p in paths],sha256=[io.sha(p.read_bytes()) for p in paths])
        assert r["sha256"][0] not in used
        used.add(r["sha256"][0])
    io.write(R/"manifest.json",dict(rows=chosen_rows,weather_lock=io.sha((R/"weather-lock.json").read_bytes())))
if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("step",choices=("screen","close","stage"));arg=cli.parse_args().step
    if arg=="screen":screening()
    else:select(arg=="stage")

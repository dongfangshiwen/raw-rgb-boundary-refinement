"""Stage new video keyframes carrying published BDD100K fog attributes and drivable masks."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import local
import argparse
import json
import time
import numpy as np
from PIL import Image,ImageDraw
import requests
from scripts import weatherproof_release_v103 as io

ROOT=io.WORK/"benchmark_data/fog-drivable-v103"
OLD=io.WORK/"experiment_exports/raw-guidance-validation-2026-11-20"
LAST=io.WORK/"experiment_exports/native-guide-segfix-2026-11-20"
MASK_REPO="dronefreak/BDD100K-Drivable-Area-Segmentation"
MASK_REV="92027ff251195fd7a97c298267b2a470829fc883"
BASE=f"https://huggingface.co/datasets/{MASK_REPO}/resolve/{MASK_REV}/"
LOCAL=local()
def fetch(url,missing=False):
    if not hasattr(LOCAL,"session"):LOCAL.session=requests.Session()
    for i in range(7):
        try:
            with LOCAL.session.get(url,timeout=(8,26)) as r:
                if missing and r.status_code==404:return None
                r.raise_for_status()
                return r.content
        except requests.RequestException:
            if i==6:raise
            time.sleep(min(1+i,4))
def metadata(item):
    blob=fetch(item["url"])
    assert io.sha(blob)==item["SHA256"]
    return [(j["file_name"].removesuffix(".jpg"),item["url"]) for j in map(json.loads,blob.splitlines()) if j["label"]=="foggy"]
def select():
    dest=io.OUT/"bdd-selection.json"
    if dest.exists():return io.read(dest)
    reg=io.read(io.WORK/"benchmark_data/bdd-fog-v73/registration.json")
    previous=set(reg["cases"])
    paths=[OLD/"bdd-screen-inventory.json",*[OLD/(s+"-manifest.json") for s in ("bdd","bddval","fogreserve")]]
    for p in paths:previous.update(r["id"] for r in io.read(p)["rows"])
    previous.update(r["id"] for r in io.read(LAST/"manifest.json")["rows"] if r["cohort"]=="bdd")
    prefixes={k.split("-")[0] for k in previous}
    with ThreadPoolExecutor(4) as pool: joined=dict(row for pack in pool.map(metadata,reg["metadata"]) for row in pack)
    assert len(joined)==reg["fog_attribute_images"]==143
    kept=[];seen=set()
    for k in sorted(joined,key=lambda s:io.sha(("bdd103/"+s).encode())):
        pre=k.split("-")[0]
        if pre in prefixes or pre in seen:continue
        seen.add(pre)
        folder=joined[k].rsplit("/",1)[0]
        split,shard=folder.split("/")[-2:]
        kept.append(dict(id=k,group=pre,split=split,shard=shard,rgb_url=f"{folder}/{k}.jpg"))
    note=dict(untouched_prefixes=len(kept),previous_prefixes=len(prefixes),excluded_by_history=sum(k.split("-")[0] in prefixes for k in joined),
              fog_attributes=len(joined),design=io.sha((io.OUT/"protocol.json").read_bytes()),mask_revision=MASK_REV,rows=kept)
    io.write(dest,note)
    return note
def one(r):
    key=r["id"]; things=[]
    for tag in ("rgb","mask"):
        p=ROOT/(key+(".jpg" if tag=="rgb" else ".png"))
        if tag=="rgb": url=r["rgb_url"]
        else:url=BASE+f"data/drivable_maps/labels/{r['split']}/{r['shard']}/{key}_drivable_id.png"
        urlfile=p.with_suffix(p.suffix+".url")
        if p.exists():
            url=urlfile.read_text(encoding="utf-8");raw=p.read_bytes()
        else:
            raw=fetch(url,missing=tag=="mask")
            if raw is None:
                for j in range(9 if r["split"]=="train" else 2):
                    probe=BASE+f"data/drivable_maps/labels/{r['split']}/shard_{j:03d}/{key}_drivable_id.png"
                    raw=fetch(probe,missing=True)
                    if raw is not None:
                        url=probe;break
            if raw is None:raise FileNotFoundError(key)
            with Image.open(BytesIO(raw)) as image:image.verify()
            urlfile.write_text(url,encoding="utf-8")
            tmp=p.with_suffix(p.suffix+".part");tmp.write_bytes(raw);tmp.replace(p)
        things.append((p,url,io.sha(raw)))
    with Image.open(things[0][0]) as img,Image.open(things[1][0]) as m:
        ids=np.asarray(m)
        assert img.size==m.size==(1280,720) and ids.ndim==2 and set(np.unique(ids))<={0,1,2,255}
    return {**r,"cohort":"bdd","urls":[a[1] for a in things],"paths":[a[0].relative_to(io.WORK).as_posix() for a in things],"sha256":[a[2] for a in things]}
def stage():
    rows=select()["rows"];ROOT.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(5) as pool:
        staged=[]
        for i,row in enumerate(pool.map(one,rows),1):
            staged.append(row)
            if i%20==0:print("New driving clips",i,len(rows),flush=True)
    known={r["rgb_sha"] for r in io.read(LAST/"manifest.json")["rows"] if r["cohort"]=="bdd"}
    for s in ("bdd","bddval","fogreserve"):known.update(r["rgb_sha"] for r in io.read(OLD/(s+"-manifest.json"))["rows"])
    assert not known.intersection(r["sha256"][0] for r in staged)
    assert len({r["sha256"][0] for r in staged})==len(staged)
    io.write(io.OUT/"bdd-manifest.json",dict(rows=staged,selection=io.sha((io.OUT/"bdd-selection.json").read_bytes())))
    for at in range(0,len(staged),30):
        sheet=Image.new("RGB",(1050,6*145),"white");d=ImageDraw.Draw(sheet)
        for j,r in enumerate(staged[at:at+30]):
            with Image.open(io.WORK/r["paths"][0]) as im:
                im.thumbnail((207,115));x=j%5*210;y=j//5*145
                sheet.paste(im,(x,y+14));d.text((x+2,y),f"B{at+j+1:03d}  "+r["id"],fill="black")
        (io.OUT/"screen").mkdir(exist_ok=True)
        sheet.save(io.OUT/"screen"/f"road-{at//30+1:02d}.jpg",quality=88)
    print("Road clips reserved",len(staged),flush=True)
if __name__=="__main__":
    stage()

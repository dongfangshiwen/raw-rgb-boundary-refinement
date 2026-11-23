"""RGB-only reserve screening and final dense-label staging; decisions are supplied by the recorded visual screen."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
import time
import requests

import numpy as np
from PIL import Image,ImageDraw

from boundary_refinement.data.staging import stage_bdd_fog as b
from boundary_refinement.training.fit_range_kernel import OUT
from boundary_refinement.data.provenance import REPO,digest,read
from boundary_refinement.training.train_regional_guidance import atomic_write

DATA=REPO.parent/"benchmark_data/bdd-weather-v101/screen"
DISPLAY=DATA/"sheets"
PROTO=OUT/"bdd-fog-screen-protocol.json"

def setup():
    DATA.mkdir(parents=True,exist_ok=True);DISPLAY.mkdir(exist_ok=True)
    split=read(b.INDEX)
    revision=split["revisions"]["semantic"]
    _,_,images=b.semantic_files((revision,"val","data/images_10k/10k/val",".jpg"))
    excluded=set(split["cases"])|{r["id"] for r in read(OUT/"bdd-manifest.json")["rows"]}
    keys=sorted(set(images)-excluded)
    assert len(images)==1000
    def get(k):
        p=DATA/(k+".jpg")
        if not p.exists():
            for attempt in range(9):
                try:
                    r=requests.get(images[k],params={"download":"true","retry":attempt},timeout=(10,32))
                    r.raise_for_status()
                    from io import BytesIO
                    with Image.open(BytesIO(r.content)) as im:
                        if min(im.size) < 2:
                            raise ValueError("Invalid image dimensions")
                        im.verify()
                    tmp=p.with_suffix(".part");tmp.write_bytes(r.content);tmp.replace(p)
                    break
                except (OSError,requests.RequestException) as e:
                    if attempt==8:
                        print("Unavailable RGB",k,str(e)[:200],flush=True)
                        return None
                    time.sleep(min(attempt+1,5))
        return dict(id=k,rgb_sha=digest(p),rgb_path=str(p.relative_to(REPO.parent)))
    with ThreadPoolExecutor(8) as pool:
        rows=list(pool.map(get,keys))
    if any(r is None for r in rows):
        raise RuntimeError(f"{rows.count(None)} missing RGBs; re-run to finish the full screen")
    assert len(set(r["rgb_sha"] for r in rows))==len(rows)
    # Full-screen visibility is easier to inspect when neighboring thumbnails have similar luminance.
    order=[]
    for row in rows:
        with Image.open(REPO.parent/row["rgb_path"]) as im:
            gray=np.asarray(im.convert("L").resize((96,54)),dtype=float)
            mid=gray[14:31,10:86]
            order.append((-float(mid.mean()-.5*mid.std()),row["id"]))
    by={r["id"]:r for r in rows};rows=[by[k] for _,k in sorted(order)]
    card=dict(protocol=digest(PROTO),excluded=sorted(excluded),revisions=split["revisions"],rows=rows)
    path=OUT/"bdd-screen-inventory.json"
    if path.exists():assert card==read(path)
    atomic_write(path,card)
    for start in range(0,len(rows),48):
        part=rows[start:start+48];sheet=Image.new("RGB",(6*220,8*144),"white");draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(part):
            with Image.open(REPO.parent/r["rgb_path"]) as img:
                pic=img.convert("RGB").resize((218,123),Image.Resampling.LANCZOS)
            x=j%6*220;y=j//6*144;sheet.paste(pic,(x,y))
            draw.text((x+3,y+124),f"{start+j:03d}   {r['id']}",fill="black")
        sheet.save(DISPLAY/f"{start//48:02d}.jpg",quality=92)
    print("RGB-only sheets",len(rows),str(DISPLAY),flush=True)

def enlarge():
    rows=read(OUT/"bdd-screen-inventory.json")["rows"]
    indices=read(OUT/"bdd-screen-candidates.json")["indices"]
    for start in range(0,len(indices),9):
        sub=indices[start:start+9];sheet=Image.new("RGB",(3*520,3*316),"white");draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(sub):
            with Image.open(REPO.parent/rows[i]["rgb_path"]) as im:
                pic=im.convert("RGB").resize((518,291),Image.Resampling.LANCZOS)
            x=j%3*520;y=j//3*316;sheet.paste(pic,(x,y));draw.text((x+4,y+293),f"{i}  {rows[i]['id']}",fill="black")
        sheet.save(DISPLAY/f"large-{start//9:02d}.jpg",quality=94)

def stage():
    rows=read(OUT/"bdd-screen-inventory.json")["rows"]
    choices=read(OUT/"bdd-screen-decisions.json")
    candidates=read(OUT/"bdd-screen-candidates.json")["indices"]
    assert set(map(int,choices["decisions"]))==set(candidates)
    baseline=read(b.INDEX)
    _,_,masks=b.semantic_files((baseline["revisions"]["semantic"],"val","data/seg_maps/labels/val","_train_id.png"))
    new=[]
    for i in sorted(map(int,choices["decisions"])):
        if not choices["decisions"][str(i)]["accept"]:continue
        row=rows[i];p=DATA/(row["id"]+".png")
        if not p.exists():p.write_bytes(b.fetch(masks[row["id"]]))
        with Image.open(p) as img, Image.open(REPO.parent/row["rgb_path"]) as original:
            assert img.size==original.size and set(np.unique(np.asarray(img)))<={*range(19),255}
        prior={r["id"].split("-")[0] for r in read(OUT/"bdd-manifest.json")["rows"]}|{k.split("-")[0] for k in baseline["cases"]}
        assert row["id"].split("-")[0] not in prior
        new.append({**row,"mask_path":str(p.relative_to(REPO.parent)),"mask_sha":digest(p),"group":row["id"].split("-")[0],"weather":"visual fog"})
    atomic_write(OUT/"fogreserve-manifest.json",dict(protocol=digest(PROTO),inventory=digest(OUT/"bdd-screen-inventory.json"),
                 decision=digest(OUT/"bdd-screen-decisions.json"),rows=new))
    print("Locked exploratory fog scenes",len(new),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("step",choices=["screen","enlarge","stage"])
    {"screen":setup,"enlarge":enlarge,"stage":stage}[p.parse_args().step]()

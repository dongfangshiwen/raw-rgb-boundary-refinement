"""Reserve author-hosted IDD-AW drives by RGB alone and verify native dense-label pairs."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import requests

from boundary_refinement.data.provenance import REPO, digest, read, write


DATA = REPO.parent / "benchmark_data/idd-aw-v100"
OUT = REPO / "results/external-fog-v100"
SOURCE = "Furqan7007/IDDAW_OFFICIAL"
REV = "76ae114b8c3b577ff07b5d2bb659866b13563f82"
BASE = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REV}/"
KIT = "https://raw.githubusercontent.com/Furqan7007/IDDAW_kit/c5e8e03bbcebf463329c3670aeebf5c2a898b72b/iddaw/info.json"
# IDD level 3 -> Cityscapes train ID. Terrain, train and the IDD-specific categories have no exact match.
MAP = {0:0, 2:1, 4:11, 5:12, 6:17, 7:18, 9:13, 10:14, 11:15, 14:3, 15:4, 18:7, 19:6,
       20:5, 22:2, 24:8, 25:10}
NAMES = ["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation",
         "terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]


def get(url):
    for attempt in range(8):
        try:
            params = {"download": "true", "attempt": attempt} if attempt and "/resolve/" in url else None
            resp = requests.get(url, params=params, timeout=(12, 55))
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as error:
            if attempt == 7:
                raise OSError(f"Downloading {url} failed after 8 attempts") from error
            time.sleep(min(8, 1 + attempt))


def fixed(path, packet):
    if path.exists() and read(path) != packet:
        raise ValueError(f"Locked record differs: {path}")
    if not path.exists():
        write(path, packet)
    return packet


def catalog():
    lock = DATA / "catalog.json"
    if lock.exists():
        card = read(lock)
        if card["revision"] != REV or card["taxonomy"] != {str(k):v for k,v in MAP.items()}:
            raise ValueError("Changed source or taxonomy")
        return card
    import json
    meta = json.loads(get(f"https://huggingface.co/api/datasets/{SOURCE}/revision/{REV}"))
    keys = {r["rfilename"] for r in meta["siblings"]}
    schema = json.loads(get(KIT))["label"]
    if any(schema[k] != NAMES[v] for k,v in MAP.items()) or meta["sha"] != REV:
        raise ValueError("Taxonomy or revision differs from the author release")
    groups = defaultdict(list)
    for k in sorted(p for p in keys if "/rgb/" in p and p.endswith("_rgb.png")):
        split, _, group, filename = k.split("/")
        mask = k.replace("/rgb/","/gt_labels/").replace("_rgb.png","_labellevel3Ids.png")
        if mask not in keys:
            raise ValueError(f"Unpaired frame {k}")
        groups[f"{split}/{group}"].append(dict(id=k.removesuffix("_rgb.png").replace("/rgb/","/"), rgb=k, mask=mask))
    if len(groups) != 161 or sum(map(len,groups.values())) != 3905:
        raise ValueError("Official drive inventory changed")
    if len({g.split("/")[1] for g in groups}) != len(groups):
        raise ValueError("Duplicate drive IDs across source splits")
    return fixed(lock, dict(revision=REV, source=SOURCE, taxonomy={str(k):v for k,v in MAP.items()},
                             groups=dict(groups), schema_source=KIT))


def fetch(name):
    path = DATA / name
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        raw = get(BASE + name)
        with Image.open(BytesIO(raw)) as im:
            im.verify()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return path


def points(rows, count):
    indices = np.linspace(0, len(rows)-1, min(len(rows),count)).round().astype(int)
    return [rows[i] for i in indices]


def rank(group):
    return sha256(("idd-fog-v100/"+group).encode()).hexdigest()


def screen():
    groups = catalog()["groups"]
    fixed(OUT/"selection-rule.json", dict(source_revision=REV, dataset="All 161 unique author drive IDs",
        screening="RGB only, no labels or predictions. View one temporal midpoint per drive; flag visible distance-dependent "
                  "atmospheric veil. Verify candidate drives with start/mid/end (all available if fewer than three). "
                  "Exclude wet glass, dark scenes, defocus or snow without independently visible depth haze, and ambiguous cases. "
                  "Report as a single-reader visual weather designation, not official tags.",
        primary="At most 24 confirmed drives in SHA256 order; up to 8 evenly spaced RGB+mask frames per drive. "
                "All 18 released validation drives also sampled at up to six evenly spaced frames as a mixed-weather sensitivity check; "
                "overlap disclosed. No IDD frame used to train or choose a model.",
        taxonomy={str(k):NAMES[v] for k,v in MAP.items()}, absent="Fixed 17-class macro, zero when union is zero; "
                  "also report classes with GT support plus full/edge support. Predictions of unmapped Cityscapes classes "
                  "on mapped GT count as false negatives. Boundaries only between different valid mapped labels."))
    ordered=sorted(groups,key=lambda g:(g.split("/")[0],int(g.split("/")[1])))
    mid=[groups[g][len(groups[g])//2]["rgb"] for g in ordered]
    with ThreadPoolExecutor(10) as p:
        files=list(p.map(fetch,mid))
    for page in range(0,len(ordered),24):
        sheet=Image.new("RGB",(4*250,6*166),"white")
        draw=ImageDraw.Draw(sheet)
        for n,(g,path) in enumerate(zip(ordered[page:page+24],files[page:page+24])):
            with Image.open(path) as im:
                im=im.convert("RGB"); im.thumbnail((248,140))
                x,y=(n%4)*250,(n//4)*166
                sheet.paste(im,(x,y));draw.text((x+3,y+141),g,fill="black")
        sheet.save(OUT/f"screen-{page//24+1}.jpg",quality=89)
    print("RGB-only midpoint screening sheets",len(files),flush=True)


def verify_weather():
    groups=catalog()["groups"]
    choices=read(OUT/"screen-candidates.json")["groups"]
    rows=[(g,groups[g][i]["rgb"]) for g in choices for i in dict.fromkeys([0,len(groups[g])//2,len(groups[g])-1])]
    with ThreadPoolExecutor(10) as pool:
        paths=list(pool.map(fetch,(r[1] for r in rows)))
    bank=defaultdict(list)
    for (g,_),p in zip(rows,paths):
        bank[g].append(p)
    for page in range(0,len(choices),6):
        sheet=Image.new("RGB",(3*325,6*203),"white")
        dr=ImageDraw.Draw(sheet)
        for i,g in enumerate(choices[page:page+6]):
            for j,p in enumerate(bank[g]):
                with Image.open(p) as im:
                    im=im.convert("RGB"); im.thumbnail((323,177)); sheet.paste(im,(j*325,i*203))
                dr.text((j*325+3,i*203+178),g+"  "+p.stem,fill="black")
        sheet.save(OUT/f"confirm-{page//6+1}.jpg",quality=90)
    print("RGB-only candidate drive triplets",len(choices),flush=True)


def stage():
    groups=catalog()["groups"]
    confirmed=sorted(read(OUT/"weather-lock.json")["confirmed"],key=rank)[:24]
    if not set(confirmed)<=set(read(OUT/"screen-candidates.json")["groups"]):
        raise ValueError("A confirmed drive was not screened")
    chosen={}
    for cohort,keys,n in (("fog",confirmed,8),("mixed",[g for g in groups if g.startswith("val/")],6)):
        for g in keys:
            for row in points(groups[g],n):
                chosen.setdefault(row["id"],{**row,"group":g,"cohorts":[]})["cohorts"].append(cohort)
    paths=sorted({row[k] for row in chosen.values() for k in ("rgb","mask")})
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(fetch,paths))
    seen=set()
    for row in chosen.values():
        a,b=(DATA/row[k] for k in ("rgb","mask"))
        with Image.open(a) as im, Image.open(b) as label:
            if im.size!=label.size or label.mode not in ("L","P","I;16","I"):
                raise ValueError(f"Unaligned native labels {row['id']}")
            ids=np.unique(np.asarray(label))
            if not set(ids.tolist())<=set(range(26))|{255}:
                raise ValueError(f"Unexpected IDD label {ids}")
            row.update(size=list(im.size),rgb_sha=digest(a),mask_sha=digest(b))
        if row["rgb_sha"] in seen:
            raise ValueError("Duplicate frame in new evaluation")
        seen.add(row["rgb_sha"])
    fixed(DATA/"evaluation.json",dict(selection=digest(OUT/"selection-rule.json"),weather=digest(OUT/"weather-lock.json"),
                                     catalog=digest(DATA/"catalog.json"),rows=list(chosen.values())))
    print("New native pairs",len(chosen),"confirmed drives",len(confirmed),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("step",choices=("screen","confirm","stage"))
    p=cli.parse_args()
    {"screen":screen,"confirm":verify_weather,"stage":stage}[p.step]()

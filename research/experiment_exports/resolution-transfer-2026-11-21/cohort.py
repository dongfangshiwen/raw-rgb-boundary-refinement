"""Reserve new label-unseen BDD keyframes and verify their provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIOR=OUT.parent/"recording-heldout-color-gate-2026-11-21"
sys.path.insert(0,str(PRIOR))
import fresh_clips as previous

DATA=ROOT/"benchmark_data/bdd-resolution-transfer-2026-11-21"


def read(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def digest(file):
    return sha256(Path(file).read_bytes()).hexdigest()


def write(file, value, locked=False):
    if locked and file.is_file():
        assert read(file)==value,file
        return
    tmp=file.with_suffix(".part")
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(file)


def reserve():
    bank=read(PRIOR/"bdd-reserve.json")
    inspected=read(PRIOR/"bdd-screen.json")["rows"]
    former=read(PRIOR/"bdd-evaluation.json")["rows"]
    exclude={r["group"] for r in inspected+former}
    pool=[r for r in bank["rows"] if r["panel"]=="screen" and r["group"] not in exclude]
    assert len(pool)==1600 and len({r["group"] for r in pool})==len(pool)
    rank=lambda r:sha256(("res-transfer-113/"+r["id"]).encode()).hexdigest()
    rows=sorted(pool,key=rank)[:128]
    assert all(not (previous.DATA/(r["id"]+".png")).exists() for r in rows)
    write(OUT/"reserve.json",dict(rows=rows,prior=len(exclude),available=len(pool),dataset=bank["repository"],
          revision=bank["revision"],history={s:digest(PRIOR/s) for s in ("bdd-reserve.json","bdd-screen.json","bdd-evaluation.json")},
          selection="SHA256 sample without conditioning on weather from 1600 previously embedding-ranked camera RGBs. No previous visual nomination, native semantic scoring or decoded test label. No replacements. Exclude near copies before opening the new labels."),True)
    return rows


def fetch(r,mask=False):
    url=previous.base()+r["label" if mask else "rgb"]
    p=DATA/(r["id"]+(".png" if mask else ".jpg"))
    if not p.is_file():
        blob=previous.history.road.fetch(url)
        with Image.open(BytesIO(blob)) as im:
            assert sorted(im.size)==[720,1280]
            if mask:assert set(np.unique(im))<=set(range(19))|{255}
        tmp=p.with_suffix(".part")
        tmp.write_bytes(blob)
        tmp.replace(p)
    return p


def fingerprint(path):
    with Image.open(path) as im:
        v=np.asarray(im.convert("RGB").resize((32,18),Image.Resampling.BILINEAR),np.float32)/255
        a=np.asarray(im.convert("L").resize((9,8),Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits(a[:,1:]>a[:,:-1]).tobytes(),"big"),v


def prep_rgb():
    DATA.mkdir(exist_ok=True)
    rows=reserve()
    with ThreadPoolExecutor(14) as pool:
        list(pool.map(fetch,rows))
    known={}
    paths={p for _,p in previous.old_rgb() if DATA not in p.parents}
    paths.update(previous.DATA/(r["id"]+"-thumb.jpg") for r in read(PRIOR/"bdd-screen.json")["rows"])
    paths.update(ROOT/r["paths"][0] for r in read(PRIOR/"bdd-evaluation.json")["rows"])
    for p in sorted(paths):
        h,v=fingerprint(p)
        known.setdefault(h,[]).append((p.name,v))
    kept,excluded=[],[]
    for r in rows:
        p=DATA/(r["id"]+".jpg")
        h,v=fingerprint(p)
        near=[name for other,images in known.items() if (other^h).bit_count()<=2 for name,im in images if np.abs(im-v).mean()<.035]
        if near:excluded.append(dict(id=r["id"],near=near))
        else:
            kept.append(dict(r,rgb_sha256=digest(p)))
            known.setdefault(h,[]).append((p.name,v))
    write(OUT/"eligible.json",dict(rows=kept,excluded=excluded,historical_rgb_files=len(paths),reserve_sha256=digest(OUT/"reserve.json")),True)
    print("New eligible prefixes",len(kept),"excluded",len(excluded),flush=True)


def stage_labels():
    fit_hash=digest(OUT/"training.json")
    eligible=read(OUT/"eligible.json")
    def stage(r):
        mask=fetch(r,True);rgb=DATA/(r["id"]+".jpg")
        assert digest(rgb)==r["rgb_sha256"]
        with Image.open(rgb) as a, Image.open(mask) as b:assert a.size==b.size
        return dict(r,paths=[str(p.relative_to(ROOT)) for p in (rgb,mask)],sha256=[r["rgb_sha256"],digest(mask)])
    with ThreadPoolExecutor(12) as pool:
        complete=list(pool.map(stage,eligible["rows"]))
    write(OUT/"new-test.json",dict(rows=complete,frozen_fits=fit_hash,eligible_sha256=digest(OUT/"eligible.json")),True)
    print("New native labels",len(complete),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("step",choices=("rgb","labels"));a=p.parse_args()
    if a.step=="rgb":prep_rgb()
    else:stage_labels()

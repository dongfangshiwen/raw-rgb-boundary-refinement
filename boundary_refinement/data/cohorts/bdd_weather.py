"""Reserve a split/attribute-based cohort before any semantic model is run on its images."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from boundary_refinement.data.staging import stage_bdd_fog as b
from boundary_refinement.training.fit_range_kernel import OUT, PLAN
from boundary_refinement.data.provenance import REPO, digest, read, now
from boundary_refinement.training.train_regional_guidance import atomic_write

DATA = REPO.parent/"benchmark_data/bdd-weather-v101"
def rank(key):
    return sha256(("rawguide101/"+key).encode()).hexdigest()

def run():
    DATA.mkdir(parents=True,exist_ok=True)
    baseline=read(b.INDEX)
    revisions=baseline["revisions"]
    tasks=[(s,i,revisions["weather"]) for s,n in b.FOLDERS.items() for i in range(n)]
    files=[(revisions["semantic"],s,f"{folder}/{s}",ext) for s in b.SEMANTIC_COUNTS
           for folder,ext in (("data/images_10k/10k",".jpg"),("data/seg_maps/labels","_train_id.png"))]
    old=set(baseline["cases"])
    with ThreadPoolExecutor(6) as pool:
        lookup_futures=[pool.submit(b.semantic_files,job) for job in files]
        weather={}
        expected={r["url"]:r["SHA256"] for r in baseline["metadata"]}
        for split,_,records,identity in pool.map(b.metadata,tasks):
            assert expected[identity["url"]]==identity["SHA256"]
            weather.update({Path(r["file_name"]).stem:r["label"] for r in records})
        lookup={(s,e):items for s,e,items in (f.result() for f in lookup_futures)}
    byweather={}
    for condition in ("rainy","snowy"):
        eligible={}
        for s in b.SEMANTIC_COUNTS:
            images,masks=lookup[s,".jpg"],lookup[s,"_train_id.png"]
            assert set(images)==set(masks) and len(images)==b.SEMANTIC_COUNTS[s]
            eligible[s]=sorted((k for k in images if weather.get(k)==condition and k not in old),key=rank)
        selected=[(k,"val") for k in eligible["val"][:64]]
        if len(selected)<15:
            selected.extend((k,"train") for k in eligible["train"][:64-len(selected)])
        byweather[condition]=dict(eligible={s:len(v) for s,v in eligible.items()},rows=[
            dict(id=k,group=k,split=s,weather=condition,rgb=lookup[s,".jpg"][k],mask=lookup[s,"_train_id.png"][k])
            for k,s in selected])
    rows=[row for weather in byweather.values() for row in weather["rows"]]
    reg=dict(protocol=digest(PLAN),source_revisions=revisions,excluded_previously_used_bdd_ids=sorted(old),
             clip_selection=byweather)
    lock=OUT/"bdd-selection.json"
    if lock.exists() and reg!=read(lock):
        raise ValueError("Frozen BDD population changed")
    atomic_write(lock,reg)
    def one(row):
        ans={**row}
        for name,ext in (("rgb","jpg"),("mask","png")):
            file=DATA/(row["id"]+"."+ext)
            if not file.exists():
                data=b.fetch(row[name]); tmp=file.with_suffix(".part");tmp.write_bytes(data);tmp.replace(file)
            ans[name+"_path"]=str(file.relative_to(REPO.parent))
            ans[name+"_sha"]=digest(file)
        with Image.open(REPO.parent/ans["rgb_path"]) as image, Image.open(REPO.parent/ans["mask_path"]) as label:
            assert image.size==label.size==(1280,720)
            assert np.asarray(label).ndim==2 and set(np.unique(np.asarray(label)))<={*range(19),255}
        return ans
    print("Reserved BDD clips",dict(Counter(r["weather"] for r in rows)),flush=True)
    with ThreadPoolExecutor(7) as pool:
        rows=list(pool.map(one,rows))
    assert len({r["rgb_sha"] for r in rows})==len(rows)
    assert not {r["rgb_sha"] for r in rows}.intersection({r["image_SHA256"] for r in read(b.MANIFEST)["pairs"]})
    atomic_write(OUT/"bdd-manifest.json",dict(selection=digest(lock),rows=rows,created_utc=now()))
    print("Verified BDD",len(rows),flush=True)
if __name__=="__main__":
    run()

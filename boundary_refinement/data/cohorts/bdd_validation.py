"""Reserve and score an untouched BDD validation cohort for the source-trained axis test."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
import time
from io import BytesIO

from PIL import Image
import numpy as np
import requests

from boundary_refinement.data.cohorts import bdd_fog_screen as screen
from boundary_refinement.data.staging import stage_bdd_fog as b
from boundary_refinement.training import color_axis as axis
from boundary_refinement.evaluation.semantic import range_transfer as ev
from boundary_refinement.training.fit_range_kernel import OUT
from boundary_refinement.data.provenance import REPO,read,digest
from boundary_refinement.training.train_regional_guidance import atomic_write

def lock():
    inventory=read(OUT/"bdd-screen-inventory.json")["rows"]
    first=read(OUT/"bdd-screen-candidates.json")["indices"]
    excluded={inventory[i]["id"] for i in first}
    prefix=lambda k:k.split("-")[0]
    prior={r["id"] for r in ev.rows("bdd")}|set(read(b.INDEX)["cases"])
    blocked={prefix(k) for k in prior|excluded}
    new=[]
    for r in sorted(inventory,key=lambda r:sha256(("axis101-locked/"+r["id"]).encode()).digest()):
        group=prefix(r["id"])
        if group not in blocked:
            new.append({**r,"group":group});blocked.add(group)
        if len(new)==128:break
    assert not {r["id"] for r in new}.intersection({r["id"] for r in ev.rows("bdd")})
    assert len(new)==len({r["rgb_sha"] for r in new})==128
    source=read(b.INDEX)["revisions"]["semantic"]
    selection=dict(axes_protocol=digest(axis.LOCK),amendment=digest(OUT/"bdd-group-amendment.json"),source=source,
                   IDs=[r["id"] for r in new],excluded_possible_fog=sorted(excluded),previous_prefixes=sorted({prefix(k) for k in prior}))
    sel=OUT/"bddval-selection.json"
    if sel.exists():assert read(sel)==selection
    atomic_write(sel,selection)
    _,_,masks=b.semantic_files((source,"val","data/seg_maps/labels/val","_train_id.png"))
    def fetch(r):
        key=r["id"];p=screen.DATA/(key+".png")
        if not p.exists():
            for i in range(8):
                try:
                    data=requests.get(masks[key],params={"try":i,"download":"true"},timeout=(12,35));data.raise_for_status()
                    with Image.open(BytesIO(data.content)) as im:im.verify()
                    temp=p.with_suffix(".part");temp.write_bytes(data.content);temp.replace(p)
                    break
                except (OSError,requests.RequestException):
                    if i==7:raise
                    time.sleep(i+1)
        with Image.open(p) as im,Image.open(REPO.parent/r["rgb_path"]) as original:
            assert im.size==original.size and set(np.unique(np.asarray(im)))<={*range(19),255}
        return {**r,"weather":"validation mix","mask_path":str(p.relative_to(REPO.parent)),"mask_sha":digest(p)}
    with ThreadPoolExecutor(6) as pool:rows=list(pool.map(fetch,new))
    atomic_write(OUT/"bddval-manifest.json",dict(selection=digest(sel),rows=rows))
    print("Reserved and verified 128 distinct official validation clips",flush=True)

def evaluate(cohort):
    if not ev.rows(cohort):
        print("No confirmed clips",cohort,flush=True);return
    for target in ("mask2former","segformer"):
        ev.run(target,cohort)
        axis.score(cohort,target)
        ev.run(target,cohort,only_peer=True)

if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("step",choices=["stage","eval"]);p.add_argument("--cohort",default="bddval")
    a=p.parse_args()
    if a.step=="stage":lock()
    else:evaluate(a.cohort)

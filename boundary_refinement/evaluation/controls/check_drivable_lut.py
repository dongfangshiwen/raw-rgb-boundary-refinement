"""Resolve the actual PNG ID encoding against the released companion color masks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
from boundary_refinement.data.cohorts import reserve_drivable as road
from boundary_refinement.data.staging.weatherproof_files import OUT, WORK, read, write, sha

def check(r):
    url=r["urls"][1].replace("/labels/","/color_labels/").replace("_drivable_id.png","_drivable_color.png")
    color=road.fetch(url)
    with Image.open(BytesIO(color)) as im, Image.open(WORK/r["paths"][1]) as labels:
        color_ids=np.asarray(im.convert("RGB")).astype(np.int32)
        y=np.asarray(labels)
    assert list(y.shape)==list(color_ids.shape[:2]) and set(np.unique(y))<={0,1,2}
    # Black background, red direct lane, blue alternative lane on the published color visualization.
    possible=np.array([[0,0,0],[255,0,0],[0,0,255]],dtype=np.int32)
    assert np.array_equal(possible[y],color_ids),r["id"]
    return dict(id=r["id"],sha256=sha(color),url=url,counts=np.bincount(y.ravel(),minlength=3).tolist())

def run():
    rows=read(OUT/"bdd-manifest.json")["rows"]
    with ThreadPoolExecutor(8) as pool:
        results=list(pool.map(check,rows))
    write(OUT/"mapping-correction.json",dict(invalid_previous_evaluation="c34b6c9247b92b2fa70f4d6c9126a283822ab40b3df4b0f6ae36cef599987512",
        discarded="Initial evaluator treated the label-definition enumeration as the native PNG indices. No preliminary BDD metric is used.",
        rule="Pixelwise agreement with companion color masks for every staged clip; native PNG indices 0=black background, 1=red direct, 2=blue alternative. Road truth is the union of IDs 1 and 2.",
        protocol="The registered semantic union and full cohort are unchanged.",source=sha((OUT/"bdd-manifest.json").read_bytes()),
        total_pixels=sum(sum(r["counts"]) for r in results),files=results))
    print("Exact masked-color parity",len(rows),flush=True)
if __name__=="__main__":
    run()

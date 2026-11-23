"""Fetch the pinned 19-class Cityscapes Mask2Former release for the source-side comparator repair."""
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests

from boundary_refinement.data.provenance import REPO,digest


BASE=REPO.parent/"benchmark_data/mask2former-road-v100"
NAME="facebook/mask2former-swin-tiny-cityscapes-semantic"
REV="01a04783d1e2981b13cf5aace8cd87774012de27"
SHA="489e8dcd9b46f991ba0daa00a9dc3d780094168937dad3ff3a1c0dd884664754"
TOTAL=189935216


def run():
    BASE.mkdir(parents=True,exist_ok=True)
    target=BASE/"model.safetensors"
    if target.exists() and digest(target)==SHA:
        return
    width=4*1024*1024
    offsets=list(range(0,TOTAL,width))
    def part(start):
        end=min(start+width,TOTAL)-1
        path=BASE/f"chunk-{start}.part"
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        for i in range(6):
            try:
                r=requests.get(f"https://huggingface.co/{NAME}/resolve/{REV}/model.safetensors",
                    params={"part":start,"attempt":i},headers={"Range":f"bytes={start}-{end}"},timeout=(12,65))
                r.raise_for_status()
                if r.headers.get("Content-Range")!=f"bytes {start}-{end}/{TOTAL}" or len(r.content)!=end-start+1:
                    raise OSError(f"Incorrect range for byte {start}")
                path.write_bytes(r.content)
                return path
            except (OSError,requests.RequestException):
                if i==5:
                    raise
                sleep(1+i)
    with ThreadPoolExecutor(7) as pool:
        chunks=list(pool.map(part,offsets))
    temp=target.with_suffix(".part")
    with temp.open("wb") as out:
        for p in chunks:
            out.write(p.read_bytes())
    if digest(temp)!=SHA:
        raise ValueError("Cityscapes pretrained weights differ from the author release")
    temp.replace(target)
    for p in chunks:
        p.unlink()
    print("Verified road weights",SHA,flush=True)


if __name__=="__main__":
    run()

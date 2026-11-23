"""Sample the published RW2AH archive using fixed nonoverlapping filename prefixes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import PurePosixPath
import random
import struct
import threading
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED
import zlib

import numpy as np
from PIL import Image, ImageOps
import requests

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import RealPairs, image_tensor, letterbox_rgb, long_edge
from scripts.real_haze_data_v22 import fingerprint, overlap_check
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


ROOT = REPO / "results/curve-transfer-v25"
STORE = REPO.parent / "benchmark_data/rw2ah-v25"
HELD = {"2","9","10"}
LIMITS = {"train":96,"test":48}
LOCAL_HEADER = struct.Struct("<I5H3I2H")


def archive_split(index, split):
    members = {}
    for record in index["members"]:
        path = PurePosixPath(record["name"])
        if len(path.parts)!=4 or path.parts[:2]!=("RW2AH_Seg",split) or path.parts[2] not in ("input","gt"):
            continue
        members.setdefault(path.stem,{})[path.parts[2]] = record
    unmatched = {i: sorted(v) for i,v in members.items() if set(v)!={"input","gt"}}
    matched = {i: v for i,v in members.items() if i not in unmatched}
    return matched, unmatched


def select(index):
    choice, exclusions = {}, {}
    for split in ("train","test"):
        rows, unmatched = archive_split(index,split)
        selector = random.Random(2509 if split=="train" else 2513)
        populations = {}
        for scene in sorted(rows):
            prefix = scene.split("-")[0]
            if (prefix in HELD)!=(split=="test"):
                continue
            populations.setdefault(scene.rsplit("-",1)[0],[]).append(scene)
        if len(populations) < (LIMITS[split] if split=="train" else 18):
            raise ValueError("Too few distinct filename groups")
        candidates = [selector.choice(v) for v in populations.values()]
        selector.shuffle(candidates)
        choice[split] = [(x,rows[x]) for x in sorted(candidates[:LIMITS[split]])]
        exclusions[split] = dict(unmatched=unmatched, available_groups=len(populations), eligible_pairs=len(rows))
    if ({x.split("-")[0] for x,_ in choice["test"]} & {x.split("-")[0] for x,_ in choice["train"]}):
        raise ValueError("RW2AH filename prefixes overlap")
    return choice, exclusions


def get_member(source, session, record):
    start = record["offset"]
    # Read through the local ZIP header and its entire variable-length field budget.
    end = min(source["bytes"]-1,start+record["compressed"]+LOCAL_HEADER.size+2*65535-1)
    expected = f"bytes {start}-{end}/{source['bytes']}"
    for attempt in range(4):
        try:
            headers = {"Range":f"bytes={start}-{end}","If-Range":source["etag"],"Accept-Encoding":"identity"}
            with session.get(source["url"],headers=headers,timeout=(15,54),stream=True) as response:
                if (response.status_code!=206 or response.headers.get("ETag")!=source["etag"]
                        or response.headers.get("Content-Range")!=expected):
                    raise OSError(f"Unexpected official member range {response.status_code}: {response.headers.get('Content-Range')}")
                data = response.content
            if len(data)!=end-start+1:
                raise OSError("Incomplete archive member range")
            header = LOCAL_HEADER.unpack_from(data)
            if header[0]!=0x04034B50 or header[2]&1 or header[3]!=record["method"]:
                raise OSError("Unexpected source archive member header")
            begin = LOCAL_HEADER.size + header[-2] + header[-1]
            raw = data[begin:begin+record["compressed"]]
            if record["method"]==ZIP_DEFLATED:
                raw=zlib.decompress(raw,-15)
            elif record["method"]!=ZIP_STORED:
                raise ValueError("Unsupported archive compression")
            if len(raw)!=record["bytes"] or f"{zlib.crc32(raw):08x}"!=record["CRC32"]:
                raise OSError("Archive payload CRC32 failed")
            return raw
        except (requests.RequestException,OSError,zlib.error):
            if attempt==3:
                raise
            time.sleep(attempt+1)
    raise RuntimeError("Unreachable archive state")


def decode(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def obtain(index, requests_by_role):
    per_thread = threading.local()

    def load(item):
        split, scene, members = item
        if not hasattr(per_thread,"session"):
            per_thread.session=requests.Session()
        raw = {kind:get_member(index,per_thread.session,info) for kind,info in members.items()}
        hazy, clear = decode(raw["input"]), decode(raw["gt"])
        if hazy.size!=clear.size:
            return split, dict(id=scene,rejected="different_native_geometry",sizes=[hazy.size,clear.size]), None
        small, truth, rect = letterbox_rgb(hazy,clear)
        witness = dict(id=scene,filename_prefix=scene.split("-")[0],native=list(clear.size),roi=rect,
                       masked_role="paired RGB",signature=fingerprint(clear),
                       normalized_clean_SHA256=hashlib.sha256(truth.numpy().tobytes()).hexdigest(),
                       clean=dict(member=members["gt"]["name"],SHA256=hashlib.sha256(raw["gt"]).hexdigest()),
                       hazy=dict(member=members["input"]["name"],SHA256=hashlib.sha256(raw["input"]).hexdigest()))
        if split=="train":
            converted = dict(id=scene,hazy=small,clean=truth,roi=rect)
        else:
            factor = min(1024,max(clear.size))
            converted = dict(id=scene,hazy=image_tensor(long_edge(hazy,factor)),clean=image_tensor(long_edge(clear,factor)))
        return split,witness,converted

    output = {"train":[],"test":[]}
    with ThreadPoolExecutor(max_workers=14) as pool:
        jobs=[(split,scene,pair) for split, pairs in requests_by_role.items() for scene,pair in pairs]
        for i,(split, report, frame) in enumerate(pool.map(load,jobs),1):
            output[split].append((report,frame))
            if i%12==0 or i==len(jobs):
                print(dict(paired_downloads=i,total=len(jobs)),flush=True)
    return output


def duplicate_screen(material):
    train = [r for r,x in material["train"] if x is not None]
    test = [r for r,x in material["test"] if x is not None]
    flagged, nearest = overlap_check(train,test)
    exclusions={r["train_id"] for r in flagged}
    extracted={kind:[x for r,x in pairs if x is not None and (kind!="train" or r["id"] not in exclusions)]
               for kind,pairs in material.items()}
    return extracted, flagged, nearest


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    receipt=ROOT/"data_manifest.json"
    if receipt.exists():
        old=read(receipt)
        for scope in ("train","test"):
            if digest(old[scope]["file"])!=old[scope]["SHA256"]:
                raise ValueError("RW2AH staged data changed")
        print(dict(already_verified={s:old[s]["count"] for s in ("train","test")}),flush=True)
        return
    index_path=STORE/"index.json"
    index=read(index_path)
    selection,raw=select(index)
    staged=obtain(index,selection)
    refined,flagged,nearest=duplicate_screen(staged)
    if len(refined["train"])<75 or len(refined["test"])<18:
        raise ValueError("Too few verified RGB pairs after overlap and geometry screening")
    sources={}
    for scope in ("train","test"):
        cache=STORE/f"prepared_{scope}.pth"
        torch_save(cache,refined[scope])
        sources[scope]=dict(file=str(cache),SHA256=digest(cache),count=len(refined[scope]),
                            prefixes=sorted({r["id"].split("-")[0] for r in refined[scope]}),
                            files=[{k:v for k,v in r.items() if k!="signature"} for r,x in staged[scope]])
    # The hosted training-only segmask has no verified Cityscapes taxonomy in this protocol.
    with requests.Session() as session:
        one=next(r for r in index["members"] if r["name"].startswith("RW2AH_Seg/train/segmask/"))
        with Image.open(io.BytesIO(get_member(index,session,one))) as img:
            array=np.array(img)
            distinct=len(np.unique(array.reshape(-1,array.shape[-1]),axis=0)) if array.ndim==3 else len(np.unique(array))
            mask_inspection=dict(member=one["name"],mode=img.mode,shape=list(array.shape),distinct=distinct,
                                 same_19_semantic_taxonomy_verified=False)
    result=dict(all_passed=True,index_SHA256=digest(index_path),source_url=index["url"],source_revision=index["revision"],
                source_primary=index["primary"], source_scope="Pinned public redistribution of the AAAI 2025 webcam paired RGB dataset.",
                train_and_test_rule="Use only archive train or test files in their own role. Disjoint first filename prefixes. One scene suffix per filename group. Prefixes are an archive proxy rather than proof of distinct physical locations.",
                raw_index=raw,exclusions_train_to_test=flagged,nearest=nearest,semantic_mask_inspection=mask_inspection,
                **sources,updated_utc=now())
    atomic_write(receipt,result)
    print(dict(paired_RGB={s:result[s]["count"] for s in ("train","test")},prefixes={s:result[s]["prefixes"] for s in ("train","test")},
               mask_inspection=mask_inspection, flagged=len(flagged)),flush=True)


if __name__=="__main__":
    main()

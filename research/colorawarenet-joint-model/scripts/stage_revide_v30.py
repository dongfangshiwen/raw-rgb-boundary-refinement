"""Read a fixed sample of the authors' REVIDE test videos through ZIP ranges."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import PurePosixPath
from zipfile import ZipFile
import zlib

import requests

from scripts.index_foggy_zurich_reserve import BLOCK, ZipByRange
from scripts.prepare_rw2ah_v25 import LOCAL_HEADER, decode
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_detail_data_v24 import image_tensor, long_edge
from scripts.real_haze_data_v22 import overlap_check
from scripts.stage_fusion_holdouts_v29 import witness
from scripts.stage_postfusion_holdout_v30 import ROOT, STORE, observed_refs
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


AUTHOR = "https://github.com/BookerDeWitt/REVIDE_Dataset"
PAPER = "https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Learning_To_Restore_Hazy_Video_A_New_Real-World_Dataset_and_CVPR_2021_paper.html"
URL = "https://drive.usercontent.google.com/download?id=1MYaVMUtcfqXeZpnbsfoJ2JBcpZUUlXGg&export=download&confirm=t"
SIZE = 1013070340
CARD = ROOT/"revide.json"


def range_bytes(session,url,size,start,end):
    headers = {"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
    with session.get(url,headers=headers,timeout=(14,35),stream=True) as response:
        response.raise_for_status()
        if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}":
            raise OSError("REVIDE ZIP range differs")
        payload = response.content
    if len(payload)!=end-start+1:
        raise OSError("Incomplete REVIDE archive range")
    return payload


class DriveRanges(ZipByRange):
    def request_block(self,start):
        raw = range_bytes(self.session,self.url,self.size,start,min(self.size,start+BLOCK)-1)
        if len(self.cache)>=8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[start] = raw
        self.received += len(raw)
        return raw


def zip_directory():
    index = STORE/"revide_index.json"
    if index.exists():
        data = read(index)
        if data["url"]!=URL or data["bytes"]!=SIZE:
            raise ValueError("Previously inspected REVIDE ZIP differs")
        return index,data
    with requests.Session() as session:
        remote = DriveRanges(session,URL,SIZE)
        with ZipFile(remote) as archive:
            members = [dict(name=p.filename,compressed=p.compress_size,bytes=p.file_size,
                            CRC32=f"{p.CRC:08x}",offset=p.header_offset,method=p.compress_type)
                       for p in archive.infolist() if not p.is_dir() and "__MACOSX" not in p.filename]
        payload = dict(url=URL,bytes=SIZE,members=members,received=remote.received)
    atomic_write(index,payload)
    return index,payload


def get_rgb(session,entry):
    start = entry["offset"]
    end = min(SIZE-1,start+entry["compressed"]+LOCAL_HEADER.size+2*65535-1)
    for attempt in range(4):
        try:
            data = range_bytes(session,URL,SIZE,start,end)
            header = LOCAL_HEADER.unpack_from(data)
            if header[0]!=0x04034b50 or header[2]&1 or header[3]!=entry["method"]:
                raise ValueError("Unexpected ZIP image header in REVIDE")
            offset = LOCAL_HEADER.size+header[-2]+header[-1]
            raw = data[offset:offset+entry["compressed"]]
            if entry["method"]==8:
                raw = zlib.decompress(raw,-15)
            elif entry["method"]!=0:
                raise ValueError("Unexpected JPEG archive compression")
            if len(raw)!=entry["bytes"] or f"{zlib.crc32(raw):08x}"!=entry["CRC32"]:
                raise OSError("REVIDE image CRC32 differs")
            return raw
        except (requests.RequestException,OSError) as exc:
            if attempt==3:
                raise
            print("Retrying REVIDE member",entry["name"],type(exc).__name__,flush=True)
    raise RuntimeError("Unreachable REVIDE retry state")


def stage():
    ROOT.mkdir(parents=True,exist_ok=True)
    STORE.mkdir(parents=True,exist_ok=True)
    if CARD.exists():
        old = read(CARD)
        if digest(old["file"])!=old["SHA256"]:
            raise ValueError("Saved REVIDE evaluation pairs changed")
        return old
    index,metadata = zip_directory()
    by_scene = defaultdict(lambda: defaultdict(dict))
    for row in metadata["members"]:
        parts = PurePosixPath(row["name"]).parts
        if len(parts)==4 and parts[0]=="Test" and parts[1] in ("hazy","gt") and parts[3].lower().endswith(".jpg"):
            by_scene[parts[2]][parts[3]][parts[1]] = row
    selected = {}
    for scene,frames in sorted(by_scene.items()):
        keys = sorted(k for k,roles in frames.items() if set(roles)=={"hazy","gt"})
        if len(keys)<5:
            raise ValueError("Incomplete video pairing in "+scene)
        for q in (1,2,3,4):
            frame = keys[round(q*(len(keys)-1)/5)]
            selected[scene+"/"+frame] = frames[frame]

    def obtain(key,record):
        with requests.Session() as session:
            fog,clear = (get_rgb(session,record[r]) for r in ("hazy","gt"))
        x,y = decode(fog),decode(clear)
        if x.size!=y.size:
            raise ValueError("REVIDE frame geometry differs")
        side = min(1024,max(y.size))
        card = witness(key,y,hashlib.sha256(clear).hexdigest())
        parts = key.split("/")
        values = dict(id=key,group=parts[0],hazy=image_tensor(long_edge(x,side)),clean=image_tensor(long_edge(y,side)))
        provenance = {k:dict(name=row["name"],CRC32=row["CRC32"]) for k,row in record.items()}
        return key,card,values,provenance

    staged = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(obtain,k,v) for k,v in selected.items()]
        for task in as_completed(futures):
            key,card,values,provenance = task.result()
            staged[key] = card,values,provenance
            if len(staged)%5==0 or len(staged)==len(selected):
                print(dict(cohort="REVIDE",paired=len(staged),total=len(selected)),flush=True)
    flagged,nearest = overlap_check(list(observed_refs()),[value[0] for value in staged.values()])
    blocked = {v["id"].split("/")[0] for v in flagged}
    kept = [staged[k][1] for k in sorted(staged) if staged[k][1]["group"] not in blocked]
    if not kept:
        raise ValueError("No REVIDE test scenes passed the overlap check")
    target = STORE/"revide_test_1024.pth"
    torch_save(target,kept)
    receipt = dict(count=len(kept),groups=sorted({r["group"] for r in kept}),file=str(target),SHA256=digest(target),
                   source=AUTHOR,paper=PAPER,url=URL,index_SHA256=digest(index),selection="Four frames at the 20%, 40%, 60%, and 80% indices of each author Test sequence. Pair by exact filenames, resample by video.",
                   excluded=flagged,nearest=nearest,registry={k:v[2] for k,v in staged.items()},finished_utc=now())
    atomic_write(CARD,receipt)
    print(dict(revide_pairs=len(kept),videos=receipt["groups"],blocked=sorted(blocked)),flush=True)
    return receipt


if __name__=="__main__":
    stage()

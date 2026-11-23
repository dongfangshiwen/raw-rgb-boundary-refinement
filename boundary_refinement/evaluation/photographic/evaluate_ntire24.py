"""Verify a fixed sample of NTIRE24 pairs and test the locked routing rule."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import PurePosixPath
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw
import requests
import torch

from boundary_refinement.utils import write_csv
from boundary_refinement.training.calibrate_expert_choice import ROOT, MODES, switches
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float, as_rgb
from boundary_refinement.evaluation.paired.evaluate_real_transfer import METRICS, quality
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.inference.infer_guarded_expert import load_router
from boundary_refinement.data.staging.prepare_rw2ah import ROOT as V25, STORE, LOCAL_HEADER, decode
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, letterbox_rgb, long_edge
from boundary_refinement.data.real_haze_data import fingerprint, load_data, overlap_check
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


DATA = ROOT/"ntire24_manifest.json"
PRIMARY = "https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Ancuti_NTIRE_2024_Dense_and_Non-Homogeneous_Dehazing_Challenge_Report_CVPRW_2024_paper.html"
SAMPLE_IDS = ("01","04","08","12","15","19","22","26","29","33","36","40")
BLOCK = 2**21


def member_bytes(source,entry):
    start = entry["offset"]
    end = start+entry["compressed"]+LOCAL_HEADER.size+2*65535-1
    joined = bytearray()
    with requests.Session() as session:
        for left in range(start,end+1,BLOCK):
            right = min(end,left+BLOCK-1)
            block = bytearray()
            failures = 0
            while left+len(block)<=right:
                begin = left+len(block)
                headers = {"Range":f"bytes={begin}-{right}","If-Range":source["etag"],"Accept-Encoding":"identity"}
                try:
                    with session.get(source["url"],headers=headers,timeout=(15,22),stream=True) as response:
                        expected = f"bytes {begin}-{right}/{source['bytes']}"
                        if response.status_code!=206 or response.headers.get("Content-Range")!=expected or response.headers.get("ETag")!=source["etag"]:
                            raise OSError("Incorrect NTIRE24 transfer range")
                        for packet in response.iter_content(chunk_size=32768):
                            block.extend(packet)
                    if left+len(block)!=right+1:
                        raise OSError("Incomplete NTIRE24 block")
                except (requests.RequestException,OSError):
                    failures += 1
                    if failures>=6:
                        raise
                    time.sleep(min(failures,3))
            joined.extend(block)
    header = LOCAL_HEADER.unpack_from(joined)
    if header[0]!=0x04034B50 or header[2]&1 or header[3]!=entry["method"] or entry["method"]!=8:
        raise ValueError("Unexpected archive member header in NTIRE24")
    offset = LOCAL_HEADER.size+header[-2]+header[-1]
    raw = zlib.decompress(joined[offset:offset+entry["compressed"]],-15)
    if len(raw)!=entry["bytes"] or f"{zlib.crc32(raw):08x}"!=entry["CRC32"]:
        raise ValueError("NTIRE24 archive member failed CRC32")
    return raw


def obtain(source,by_id):
    output,unavailable = [],{}
    with ThreadPoolExecutor(max_workers=16) as workers:
        jobs = {workers.submit(member_bytes,source,by_id[scene][kind]):(scene,kind)
                for scene in SAMPLE_IDS for kind in ("input","gt")}
        records = defaultdict(dict)
        for job in as_completed(jobs):
            scene,kind = jobs[job]
            try:
                records[scene][kind] = job.result()
            except (requests.RequestException,OSError) as error:
                unavailable[scene] = type(error).__name__
                print(dict(unavailable=scene,kind=kind,error=type(error).__name__),flush=True)
                continue
            if len(records[scene])!=2:
                continue
            pixels = records.pop(scene)
            hazy,clear = (decode(pixels[k]) for k in ("input","gt"))
            if hazy.size!=clear.size:
                unavailable[scene] = "geometry_mismatch"
                continue
            _, normalized, rect = letterbox_rgb(hazy,clear)
            witness = dict(id=scene,native=list(clear.size),roi=rect,signature=fingerprint(clear),
                           normalized_clean_SHA256=hashlib.sha256(normalized.numpy().tobytes()).hexdigest(),
                           clean=dict(member=by_id[scene]["gt"]["name"],SHA256=hashlib.sha256(pixels["gt"]).hexdigest()),
                           hazy=dict(member=by_id[scene]["input"]["name"],SHA256=hashlib.sha256(pixels["input"]).hexdigest()))
            frame = dict(id=scene,hazy=image_tensor(long_edge(hazy,1024)),clean=image_tensor(long_edge(clear,1024)))
            output.append((witness,frame))
            print(dict(paired=len(output),selected=len(SAMPLE_IDS),scene=scene),flush=True)
    return sorted(output,key=lambda p:p[0]["id"]),unavailable


def train_signatures():
    rw = read(V25/"data_manifest.json")["train"]
    if digest(rw["file"])!=rw["SHA256"]:
        raise ValueError("Previous training cache changed")
    first = torch.load(rw["file"],map_location="cpu",weights_only=True)
    second = load_data("training")["training"]
    original_sha = {r["id"]:r["clean"]["SHA256"] for r in rw["files"]}
    for domain,rows in (("RW2AH",first),("NH-HAZE",second)):
        for row in rows:
            x,y,end_x,end_y = row["roi"]
            original = row["clean"]
            view = original[:,y:end_y,x:end_x].permute(1,2,0).numpy()
            yield dict(id=domain+"/"+row["id"],signature=fingerprint(Image.fromarray(view)),
                       normalized_clean_SHA256=hashlib.sha256(original.numpy().tobytes()).hexdigest(),
                       clean={"SHA256":original_sha.get(row["id"]) if domain=="RW2AH" else None})


def stage():
    if DATA.exists():
        manifest = read(DATA)
        if digest(manifest["file"])!=manifest["SHA256"]:
            raise ValueError("Supplementary NTIRE24 RGB cache changed")
        return manifest
    file = STORE/"extension_index.json"
    source = read(file)
    registered = read(STORE/"index.json")
    if any(source[k]!=registered[k] for k in ("url","etag","revision","bytes")):
        raise ValueError("Supplement archive revision changed")
    by_id = defaultdict(dict)
    for item in source["members"]:
        folder,role,name = PurePosixPath(item["name"]).parts[:3]
        if folder=="NTIRE24" and name.upper().endswith(".JPG") and role in ("clear","hazy"):
            by_id[PurePosixPath(name).stem]["input" if role=="hazy" else "gt"] = item
    if sorted(by_id)!=[f"{i:02d}" for i in range(1,41)] or any(set(v)!={"gt","input"} for v in by_id.values()):
        raise ValueError("The public challenge training photo IDs differ")
    pairs,unavailable = obtain(source,by_id)
    compared = [r for r,x in pairs]
    excluded,nearest = overlap_check(list(train_signatures()),compared)
    bad = {x["id"] for x in excluded}
    cache = [v for r,v in pairs if v is not None and r["id"] not in bad]
    if not cache:
        raise ValueError("No NTIRE24 pairs passed transfer, geometry and overlap screening")
    target = STORE/"ntire24_test_1024.pth"
    torch_save(target,cache)
    state = dict(count=len(cache),source_revision=source["revision"],source_url=source["url"],primary=PRIMARY,
                 index_SHA256=digest(file),file=str(target),SHA256=digest(target),excluded=excluded,nearest=nearest,
                 pairs=[{k:v for k,v in row.items() if k!="signature"} for row,x in pairs],sampled_ids=list(SAMPLE_IDS),
                 unavailable=unavailable,
                 scope="12 evenly spaced IDs from the 40 publicly labeled NTIRE 2024 challenge training photographs. Full-file HTTP ranges terminated early, so the fixed sample uses resumable 2 MiB blocks. No scores were seen when choosing IDs. Only held-out evaluation of the previously frozen local models.",
                 overlap="Compare hashes and perceptual thumbnails with the two local paired RGB fit cohorts. The author MCT checkpoint exposure is unknown.",
                 finished_utc=now())
    atomic_write(DATA,state)
    print(dict(staged=len(cache),duplicates=len(excluded)),flush=True)
    return state


@torch.inference_mode()
def assess(source):
    configure_cuda()
    state,prior = load_router()
    forward,registration = experts()
    rows,first = [],None
    counts = dict.fromkeys(MODES,0)
    for sample in torch.load(source["file"],map_location="cpu",weights_only=True):
        scene = sample["id"]
        x,y = (as_float(sample[k]) for k in ("hazy","clean"))
        a,b = (f(x).clamp(0,1) for f in forward)
        result = switches(state,x,a,b)
        values = [quality(v,y)[0] for v in (a,b)]
        for method,selected in {"author_mct":False,"author_pgl":True,**result}.items():
            rows.append(dict(scene=scene,method=method,pgl_selected=int(selected),**dict(zip(METRICS,values[int(selected)]))))
            if method in counts:
                counts[method] += selected
        if first is None:
            first = (scene,[as_rgb(v) for v in (x,a,b,b if result["guarded"] else a,y)])
    target = ROOT/"ntire24_metrics.csv"
    write_csv(target,rows)
    pooled = {m:float(np.mean([r["psnr"] for r in rows if r["method"]==m])) for m in ("author_mct","author_pgl",*MODES)}
    a,b = (np.array([r["psnr"] for r in rows if r["method"]==k]) for k in ("guarded","author_mct"))
    rng = np.random.default_rng(27024)
    errors = a-b
    boot = errors[rng.integers(0,len(errors),size=(10000,len(errors)))].mean(1)
    demo = ROOT/"ntire24_prediction.png"
    canvas = Image.new("RGB",(1600,256),"#f2f5f7")
    draw = ImageDraw.Draw(canvas)
    scene,images = first
    for i,(view,name) in enumerate(zip(images,("Hazy","MCT","PGL-Net-T","Guarded selection","Clear reference"))):
        view.thumbnail((310,215),Image.Resampling.LANCZOS)
        canvas.paste(view,(320*i+(320-view.width)//2,27))
        draw.text((320*i+7,6),f"NTIRE24 {scene} | {name}",fill="#153042")
    canvas.save(demo)
    receipt = dict(count=source["count"],scores=pooled,PGL_selected=counts,change_dB=float(errors.mean()),
                   interval95=np.percentile(boot,[2.5,97.5]).tolist(),
                   improves=int((errors>1e-6).sum()),degrades=int((errors< -1e-6).sum()),
                   manifest_SHA256=digest(DATA),scores_SHA256=digest(target),source_SHA256=digest(__file__),
                   calibration_SHA256=prior["calibration_SHA256"],comparison_SHA256=digest(demo),
                   author_weights=registration,finished_utc=now())
    atomic_write(ROOT/"ntire24.json",receipt)
    print(dict(scored=receipt["count"],scores=pooled,change=receipt["change_dB"],selections=counts),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    source = stage()
    if not args.prepare_only:
        assess(source)


if __name__=="__main__":
    main()

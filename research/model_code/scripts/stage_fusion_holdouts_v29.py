"""Stage the authors' RW-Haze release and the unseen NTIRE24 complement."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from PIL import Image, ImageOps
import requests
import torch

from scripts.evaluate_ntire24_v27 import DATA as NTIRE, member_bytes, train_signatures
from scripts.prepare_rw2ah_v25 import STORE as PINNED, decode
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor, letterbox_rgb, long_edge
from scripts.real_haze_data_v22 import fingerprint, overlap_check
from scripts.stage_bedde_v28 import RECEIPT as BEDDE
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


ROOT = REPO / "results/conditional-fusion-v29"
STORE = REPO.parent / "benchmark_data/fusion-holdouts-v29"
RW_SOURCE = "https://github.com/CVhnu/DehIQA"
RW_PAPER = "https://ieeexplore.ieee.org/document/9897706"


def witness(group, view, raw_sha):
    _, pixels, _ = letterbox_rgb(view, view)
    return dict(id=str(group), signature=fingerprint(view), clean={"SHA256":raw_sha},
                normalized_clean_SHA256=hashlib.sha256(pixels.numpy().tobytes()).hexdigest())


def prior_refs():
    yield from train_signatures()
    registry = read(BEDDE)
    for name in registry["members"]:
        p = PurePosixPath(name)
        if p.parts[-2] == "gt":
            path = Path(registry["directory"])/p
            if digest(path) != registry["image_hashes"][name]:
                raise ValueError("Registered BeDDE clear image changed")
            with Image.open(path) as im:
                yield witness("BeDDE/"+p.parts[-3], ImageOps.exif_transpose(im).convert("RGB"), digest(path))
    old = read(NTIRE)
    if digest(old["file"]) != old["SHA256"]:
        raise ValueError("Previously scored NTIRE24 subset changed")
    for row in torch.load(old["file"], map_location="cpu", weights_only=True):
        img = Image.fromarray(row["clean"].permute(1, 2, 0).numpy())
        yield witness("NTIRE24/"+row["id"], img, None)


def stage_rwhaze():
    target, register = STORE/"RW-Haze.zip", ROOT/"rwhaze_manifest.json"
    if register.exists():
        result = read(register)
        if digest(target) != result["SHA256"]:
            raise ValueError("Registered RW-Haze archive changed")
        return result
    with requests.Session() as session:
        base = "https://api.github.com/repos/CVhnu/DehIQA"
        head = session.get(base+"/commits/main", timeout=(12, 24))
        head.raise_for_status()
        commit = head.json()["sha"]
        listing = session.get(base+f"/contents/RW_HAZE?ref={commit}", timeout=(12, 24))
        listing.raise_for_status()
        source = "https://codeload.github.com/CVhnu/DehIQA/zip/" + commit
        blob = {r["name"]:r["sha"] for r in listing.json() if r["type"]=="file"}
        temp = target.with_suffix(".partial")
        with session.get(source, timeout=(16, 65), stream=True) as data, temp.open("wb") as out:
            data.raise_for_status()
            for part in data.iter_content(2**19):
                out.write(part)
        temp.replace(target)
    sequence = defaultdict(dict)
    refs = []
    with ZipFile(target) as archive:
        for name in archive.namelist():
            path = PurePosixPath(name)
            match = re.fullmatch(r"([1-9][0-9]*)(?:_([1-9][0-9]*))?\.jpg", path.name)
            if len(path.parts) != 3 or path.parts[-2] != "RW_HAZE" or not match:
                continue
            raw = archive.read(name)
            actual = hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
            if actual != blob[path.name]:
                raise ValueError("RW-Haze image does not match the pinned author's Git tree")
            group,frame = match.groups()
            row = dict(member=name,SHA256=hashlib.sha256(raw).hexdigest(),size=list(decode(raw).size))
            sequence[group][frame or "clear"] = row
            if frame is None:
                refs.append(witness(group,decode(raw),row["SHA256"]))
    if not sequence or any("clear" not in v or len(v)<2 for v in sequence.values()):
        raise ValueError("Incomplete RW-Haze author image groups")
    for group,rows in sequence.items():
        if any(row["size"]!=rows["clear"]["size"] for row in rows.values()):
            raise ValueError("Unaligned RGB grid in RW-Haze " + group)
    overlap,nearest = overlap_check(list(prior_refs()),refs)
    excluded = {v["id"] for v in overlap}
    allowed = {k:v for k,v in sequence.items() if k not in excluded}
    registry = dict(author=RW_SOURCE,paper=RW_PAPER,revision=commit,url=source,file=str(target),SHA256=digest(target),
                    groups=allowed,count=sum(len(rows)-1 for rows in allowed.values()),excluded=overlap,nearest=nearest,
                    scope="All released RW_HAZE JPEG pairs in the pinned public author repository. Group IDs share a clear frame.",
                    finished_utc=now())
    atomic_write(register,registry)
    print(dict(cohort="RW-Haze",pairs=registry["count"],groups=len(allowed),duplicate_groups=len(excluded)),flush=True)
    return registry


def stage_ntire():
    target,register = STORE/"ntire24_complement.pth",ROOT/"ntire24_complement_manifest.json"
    if register.exists():
        record = read(register)
        if digest(target)!=record["SHA256"]:
            raise ValueError("Registered NTIRE24 complement changed")
        return record
    data = read(PINNED/"extension_index.json")
    locked = read(PINNED/"index.json")
    if any(data[k]!=locked[k] for k in ("url","etag","revision","bytes")):
        raise ValueError("NTIRE24 archive source changed")
    prior = read(NTIRE)
    by_id = defaultdict(dict)
    for r in data["members"]:
        p = PurePosixPath(r["name"])
        if len(p.parts)==3 and p.parts[0]=="NTIRE24" and p.suffix.upper()==".JPG":
            by_id[p.stem][p.parts[1]] = r
    intended = sorted(set(by_id)-set(prior["sampled_ids"]))
    if len(intended)!=28 or any(set(by_id[k])!={"hazy","clear"} for k in intended):
        raise ValueError("The 28 unseen NTIRE24 IDs differ from the registered complement")
    rows,signatures,checks,unavailable = {},[],{},{}
    pixels = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=16) as pool:
        jobs = {pool.submit(member_bytes,data,by_id[k][role]):(k,role)
                for k in intended for role in ("hazy","clear")}
        for job in as_completed(jobs):
            scene,role = jobs[job]
            try:
                pixels[scene][role] = job.result()
            except (requests.RequestException,OSError) as error:
                unavailable[scene] = type(error).__name__
                continue
            if len(pixels[scene])!=2:
                continue
            pair = pixels.pop(scene)
            hazy,clear = (decode(pair[r]) for r in ("hazy","clear"))
            if hazy.size != clear.size:
                raise ValueError("New NTIRE24 pair coordinates differ")
            source = {r:hashlib.sha256(v).hexdigest() for r,v in pair.items()}
            signatures.append(witness(scene,clear,source["clear"]))
            checks[scene] = source
            rows[scene] = dict(id=scene,hazy=image_tensor(long_edge(hazy,1024)),clean=image_tensor(long_edge(clear,1024)))
            if len(rows)%5==0 or len(rows)==len(intended):
                print(dict(cohort="NTIRE24 complement",paired=len(rows),total=len(intended)),flush=True)
    repeated,nearest = overlap_check(list(prior_refs()),signatures)
    allowed = sorted(set(rows)-{r["id"] for r in repeated})
    if not allowed:
        raise ValueError("No verified NTIRE24 complement pairs are available")
    torch_save(target,[rows[k] for k in allowed])
    state = dict(count=len(allowed),ids=allowed,intended=intended,excluded=repeated,nearest=nearest,pairs_SHA256=checks,
                 unavailable=unavailable,
                 file=str(target),SHA256=digest(target),source_url=data["url"],revision=data["revision"],primary=prior["primary"],
                 scope="Uninspected complement of the earlier local NTIRE24 12-image subset, from labeled challenge-training capture pairs.",
                 finished_utc=now())
    atomic_write(register,state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("rwhaze","ntire24"))
    args = parser.parse_args()
    STORE.mkdir(parents=True,exist_ok=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    (stage_rwhaze if args.cohort=="rwhaze" else stage_ntire)()


if __name__ == "__main__":
    main()

"""Reserve all remaining RW2AH captures and an unused HD-NH-HAZE subset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import PurePosixPath

from PIL import Image
import torch

from scripts.evaluate_ntire24_v27 import member_bytes
from scripts.prepare_rw2ah_v25 import ROOT as RW25, STORE, archive_split, decode, obtain
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor, long_edge
from scripts.real_haze_data_v22 import overlap_check
from scripts.stage_fusion_holdouts_v29 import witness
from scripts.stage_lmhaze_v31 import TRAIN_CARD, reference
from scripts.stage_postfusion_holdout_v30 import RW_CARD, camera_code, observed_refs
from scripts.stage_rw_confirmation_v32 import CARD as RW32, capture
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


ROOT = REPO/"results/selective-adaptation-v33"
PIXELS = REPO.parent/"benchmark_data/selective-adaptation-v33"
HD_SOURCE = "https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Ancuti_NTIRE_2023_HR_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2023_paper.html"
HD_IDS = ("36","37","38","39","40")
HD_SUPPLEMENT = tuple(f"ntire23-{i:03d}" for i in (2,6,10,14,18,22,26,30,34))


def previous_references():
    yield from observed_refs()
    for path in (RW_CARD,RW32):
        card = read(path)
        if digest(card["file"])!=card["SHA256"]:
            raise ValueError("An earlier RW²AH archive changed")
        for row in torch.load(card["file"],map_location="cpu",weights_only=True):
            clean = Image.fromarray(row["clean"].permute(1,2,0).numpy())
            yield witness("RW2AH/"+row["id"],clean,"previous:"+row["id"])
    for row in read(TRAIN_CARD)["scenes"]:
        if row["role"]=="fit":
            yield reference(row)


def rw_pairs():
    index = STORE/"index.json"
    old = read(RW25/"data_manifest.json")
    if digest(index)!=old["index_SHA256"]:
        raise ValueError("The RW²AH central directory differs")
    archive = read(index)
    seen = {capture(row["id"]) for role in ("train","test") for row in old[role]["files"]}
    for card,key in ((RW_CARD,"intended"),(RW32,"selected")):
        seen.update(capture(v) for v in read(card)[key])
    members,_ = archive_split(archive,"test")
    selection = {}
    for k in sorted(members,key=lambda id:sha256(("selective-adaptation-v33/"+id).encode()).digest()):
        group = capture(k)
        if group not in seen:
            selection.setdefault(group,k)
    ids = sorted(selection.values())
    if len(ids)!=48:
        raise ValueError("Remaining RW²AH captures differ from the indexed population")
    result = obtain(archive,{"test":[(id,members[id]) for id in ids]})["test"]
    return result,dict(index_SHA256=digest(index),revision=archive["revision"],planned=ids,
                       scope="Remaining 48 unobserved capture codes, with a fixed single sample per code. Same archive and two camera prefixes as a prior test. The author checkpoint's original coverage is unknown.")


def hd_archive():
    index = STORE/"extension_index.json"
    archive = read(index)
    original = read(STORE/"index.json")
    if any(archive[k]!=original[k] for k in ("url","bytes","etag","revision")):
        raise ValueError("The HD archive revision is inconsistent")
    return index,archive,{v["name"]:v for v in archive["members"]}


def hd_pairs(which):
    index,archive,entries = hd_archive()
    folder, ids = ("Test", HD_IDS) if which=="hd" else ("Train",HD_SUPPLEMENT)
    names = {k:[f"HD-NH-HAZE/{folder}/{role}/{k}.png" for role in ("hazy","clean")] for k in ids}

    def load_pair(job):
        key,members = job
        raw = [member_bytes(archive,entries[name]) for name in members]
        x,y = map(decode,raw)
        if x.size!=y.size:
            raise ValueError("HD-NH-HAZE reference geometry differs")
        card = witness(key,y,sha256(raw[1]).hexdigest())
        card["member"] = members
        print(dict(hd_pair=key,width=x.width,height=x.height),flush=True)
        return card,dict(id=key,hazy=image_tensor(long_edge(x,1024)),clean=image_tensor(long_edge(y,1024)))

    with ThreadPoolExecutor(max_workers=min(len(ids),8)) as workers:
        pairs = list(workers.map(load_pair,names.items()))
    return pairs,dict(index_SHA256=digest(index),revision=archive["revision"],planned=list(ids),paper=HD_SOURCE,
                      scope=f"Public labeled HD-NH-HAZE pairs from redistribution folder {folder}. Use as a local external transfer subset, consistent with the stated author exposures.")


def stage(which):
    ROOT.mkdir(parents=True,exist_ok=True)
    PIXELS.mkdir(parents=True,exist_ok=True)
    registry = ROOT/(which+"_data.json")
    if registry.exists():
        card = read(registry)
        if digest(card["file"])!=card["SHA256"]:
            raise ValueError("Staged held-out pixels changed")
        return card
    candidates,card = rw_pairs() if which=="rw" else hd_pairs(which)
    ready = [(entry,pixels) for entry,pixels in candidates if pixels is not None]
    past = list(previous_references())
    if which=="hd_supplement":
        old_hd = read(ROOT/"hd_data.json")
        if digest(old_hd["file"])!=old_hd["SHA256"]:
            raise ValueError("Previous HD transfer pixels changed")
        for row in torch.load(old_hd["file"],map_location="cpu",weights_only=True):
            clean = Image.fromarray(row["clean"].permute(1,2,0).numpy())
            past.append(witness("HD/"+row["id"],clean,"hd:"+row["id"]))
    flags,nearest = overlap_check(past,[entry for entry,_ in ready])
    excluded = {v["id"] for v in flags}
    unique = [(entry,pixels) for entry,pixels in ready if entry["id"] not in excluded]
    if not unique:
        raise ValueError("No held-out image passed the overlap check")
    path = PIXELS/(which+"_1024.pth")
    torch_save(path,[pixels for _,pixels in unique])
    result = dict(card,count=len(unique),ids=[pixels["id"] for _,pixels in unique],
                  excluded=flags,nearest=nearest,file=str(path),SHA256=digest(path),finished_utc=now())
    if which=="rw":
        result["camera_prefixes"] = sorted({camera_code(k) for k in result["ids"]})
    atomic_write(registry,result)
    print(dict(set=which,pairs=len(unique),excluded=len(excluded)),flush=True)
    return result


def audit_supplement():
    """Archive exclusion evidence even when an entire downloaded batch overlaps."""
    index,archive,entries = hd_archive()

    def clean_reference(identifier):
        member = f"HD-NH-HAZE/Train/clean/{identifier}.png"
        raw = member_bytes(archive,entries[member])
        return witness(identifier,decode(raw),sha256(raw).hexdigest())

    with ThreadPoolExecutor(max_workers=9) as workers:
        refs = list(workers.map(clean_reference,HD_SUPPLEMENT))
    flags,nearest = overlap_check(list(previous_references()),refs)
    omitted = {r["id"] for r in flags}
    record = dict(paper=HD_SOURCE,index_SHA256=digest(index),checked=list(HD_SUPPLEMENT),excluded=flags,
                  kept=sorted(set(HD_SUPPLEMENT)-omitted),nearest=nearest,finished_utc=now())
    atomic_write(ROOT/"hd_supplement_overlap.json",record)
    print(dict(checked=len(refs),excluded=len(omitted),kept=len(record["kept"])),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("rw","hd","hd_supplement","audit_supplement"))
    split = parser.parse_args().cohort
    audit_supplement() if split=="audit_supplement" else stage(split)

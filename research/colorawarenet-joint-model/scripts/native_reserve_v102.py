"""Reserve new BDD prefixes and previously unstaged frames of screened IDD fog drives."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
import requests

from scripts import idd_fog_data_v100 as idd
from scripts import stage_bdd_fog_v73 as bdd
from scripts.fit_range_v101 import OUT as PRIOR
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write

ROOT = REPO.parent / "experiment_exports/native-guide-segfix-2026-11-20"
DATA = REPO.parent / "benchmark_data/native-guide-v102"
SEEDS = (101, 211, 307)
ARMS = ("released", "bilinear", "paper_512", "segfix", "jbu_native", "fixed_native",
        "fixed_lowpass", "scalar", "rgb_gap", "class_gap", "class_gap_lowpass")


def protocol():
    ROOT.mkdir(exist_ok=True)
    path = ROOT / "protocol.json"
    doc = dict(primary="Boundary-band mIoU, 19 BDD or 17 exact IDD classes. Full-image mIoU is secondary.",
               question="Does querying original pixels improve upon downsampled RGB and the existing decoder? "
                        "Does class-conditioned donor separation add value beyond its two-scalar RGB-only control?",
               arms=ARMS, seeds=SEEDS, fit="Frozen released LiDSeg source, 512-pixel backbone input. Only the 96 "
               "MUSES fog images in nine source recordings. Three 320-update AdamW fits for each of scalar, "
               "RGB-only separation and class-conditioned separation; shared sampled native pixels per seed. "
               "The held-out images provide no fit, stopping, tuning, or arm selection.",
               kernel="Four positive half-pixel bilinear donors with clamped border and a 3x3 coarse transition gate. "
               "Donor colors are group means on the 512 input lattice. Native or bilinearly lifted low-pass query RGB. "
               "Fixed four saturating scales .02,.05,.10,.20. Learned overall strength depends on nearest versus "
               "nearest different-class donor color distance, or nearest versus second nearest for RGB control. "
               "Standard JBU uses nine Gaussian spatial donors (sigma .65), RGB sigma .05; same transition gate.",
               comparators="Exact source-fitted manuscript decoder at 512 and released target reference. SegFix "
               "uses the authors' Cityscapes train-only HRNet-W48 checkpoint, single native-size RGB pass, "
               "two-pixel offsets on the released native prediction; no target training, no multi-scale testing.",
               fresh_bdd="128 public BDD semantic training-partition images by SHA256(native102/ID), one per "
               "filename prefix. Exclude all previously scored prefixes, official fog subset and the entire prior "
               "validation RGB inventory. No weather restriction. The project treats these images as evaluation only.",
               fog="For every previously visually confirmed IDD-AW fog drive, use up to four evenly spaced frames "
               "with neither a previously staged RGB nor a semantic mask. These are new frames from previously "
               "screened drives. This is a within-drive replication, not an independent-route claim.",
               inference="Identical frozen 512-long-side Mask2Former and SegFormer-B1 scores, native labels, radii "
               "six in both directions, 5000 paired bootstrap draws by drive/prefix seed 10211. Average seed metrics "
               "after pooling within each draw. Intervals pointwise, fixed primary conjunction across both backbones.",
               decisions="Claim learned class gating only if class_gap beats scalar AND rgb_gap on both targets "
               "on reserved BDD with lower paired limits above zero, and report fog replication. Claim native "
               "color within method only if fixed_native beats fixed_lowpass. Compare against paper_512, native "
               "JBU and released+SegFix separately. Do not infer photographic gains from segmentation.")
    if path.exists():
        assert read(path) == {**doc, "arms":list(ARMS), "seeds":list(SEEDS)}
    else:
        atomic_write(path, doc)
    return path


def reserve():
    protocol()
    lock = ROOT / "selection.json"
    if lock.exists():
        return read(lock)
    old = read(bdd.INDEX)
    rev = old["revisions"]["semantic"]
    previous = set(old["cases"])
    previous.update(r["id"] for r in read(PRIOR/"bdd-screen-inventory.json")["rows"])
    for cohort in ("bdd", "bddval", "fogreserve"):
        previous.update(r["id"] for r in read(PRIOR/f"{cohort}-manifest.json")["rows"])
    used = {i.split("-")[0] for i in previous}
    prefix_count = len(used)
    repo = bdd.REPOS["semantic"]
    meta = json.loads(bdd.fetch(f"{bdd.HUB}/api/datasets/{repo}/revision/{rev}"))
    assert meta["sha"] == rev
    files = [r["rfilename"] for r in meta["siblings"]]
    def artifacts(folder, suffix):
        return {Path(f).name.removesuffix(suffix):bdd.path_at(repo,rev,f) for f in files
                if f.startswith(folder+"/") and f.endswith(suffix)}
    images = artifacts("data/images_10k/10k/train",".jpg")
    masks = artifacts("data/seg_maps/labels/train","_train_id.png")
    assert images.keys() == masks.keys() and len(images)==7000
    fresh = []
    for key in sorted(images,key=lambda k:sha256(("native102/"+k).encode()).digest()):
        prefix = key.split("-")[0]
        if prefix not in used:
            fresh.append(dict(id=key,group=prefix,rgb_url=images[key],mask_url=masks[key],cohort="bdd"))
            used.add(prefix)
        if len(fresh)==128:
            break
    groups = idd.catalog()["groups"]
    confirmed = read(idd.OUT/"weather-lock.json")["confirmed"]
    fog = []
    for g in sorted(confirmed):
        available=[r for r in groups[g] if not any((idd.DATA/r[k]).exists() for k in ("mask","rgb"))]
        if available:
            for r in idd.points(available,4):
                fog.append(dict(**r,group=g,cohort="fog"))
    assert len(fresh)==128 and len({r["group"] for r in fog})>=10
    doc=dict(protocol=digest(ROOT/"protocol.json"), frozen_utc=now(), bdd_revision=rev,
             excluded_bdd_prefixes=prefix_count, confirmed_idd=confirmed,
             groups=dict(Counter(r["group"] for r in fog)), rows=fresh+fog)
    atomic_write(lock,doc)
    return doc


def stage():
    doc = reserve()
    DATA.mkdir(exist_ok=True)
    def get(row):
        out=dict(row)
        for field,ext in (("rgb",".jpg"),("mask",".png")):
            if row["cohort"]=="bdd":
                path=DATA/(row["id"]+ext)
                if not path.exists():
                    tmp=path.with_suffix(".part")
                    for attempt in range(7):
                        try:
                            response=requests.get(row[field+"_url"],params={"download":"1","retry":attempt},timeout=(12,45))
                            response.raise_for_status()
                            tmp.write_bytes(response.content)
                            break
                        except requests.RequestException:
                            if attempt==6:
                                raise
                            time.sleep(1+attempt)
                    with Image.open(tmp) as im:
                        im.verify()
                    tmp.replace(path)
            else:
                path=idd.fetch(row[field])
            out[field+"_path"]=str(path.relative_to(REPO.parent))
            hasher=sha256()
            with path.open("rb") as file:
                for block in iter(lambda:file.read(262144),b""):
                    hasher.update(block)
            out[field+"_sha"]=hasher.hexdigest()
        with Image.open(REPO.parent/out["rgb_path"]) as im, Image.open(REPO.parent/out["mask_path"]) as lab:
            assert im.size==lab.size and len(lab.getbands())==1
            if row["cohort"]=="bdd":
                assert set(np.unique(np.asarray(lab)))<={*range(19),255}
        return out
    with ThreadPoolExecutor(4) as pool:
        rows=list(pool.map(get,doc["rows"]))
    assert len({r["rgb_sha"] for r in rows})==len(rows)
    old=set()
    for cohort in ("bdd","bddval","fogreserve"):
        old.update(r["rgb_sha"] for r in read(PRIOR/f"{cohort}-manifest.json")["rows"])
    old.update(r["rgb_sha"] for r in read(idd.DATA/"evaluation.json")["rows"])
    old.update(r["image_SHA256"] for r in read(bdd.MANIFEST)["pairs"])
    assert old.isdisjoint(r["rgb_sha"] for r in rows)
    atomic_write(ROOT/"manifest.json",dict(selection=digest(ROOT/"selection.json"),rows=rows))
    print([(c,len([r for r in rows if r["cohort"]==c]),len({r["group"] for r in rows if r["cohort"]==c})) for c in ("bdd","fog")],flush=True)


if __name__=="__main__":
    stage()

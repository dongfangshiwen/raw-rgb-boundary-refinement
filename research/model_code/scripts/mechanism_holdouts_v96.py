"""Reserve paired tiles and recording-disjoint real scenes before scoring either intervention."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from zipfile import ZipFile

import numpy as np
import requests
import torch

from scripts.decision_overlap_v55 import MANIFEST, pair_index
from scripts.evaluate_complementary_rgb_v59 import CONDITIONS, HELD, NEW
from scripts.fresh_real_v53 import CARD, RR, image
from scripts.projective_ledger_v12 import REPO, digest, read
from scripts.real_fog_correction_v70 import verified_image
from scripts.real_haze_data_v22 import fingerprint
from scripts.stage_muses_rain_v90 import FOG_MANIFESTS, MANIFEST as RAIN
from scripts.stage_muses_snow_v92 import MANIFEST as SNOW
from scripts.stage_muses_v67 import ARCHIVES, archive_index, stage_item
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = REPO.parent / "benchmark_data/mechanism-holdouts-v96"
TILES, SCENES, SCENE_INDEX = (ROOT / name for name in ("tiles.json", "scenes.json", "scene-selection.json"))


def rank(key):
    return sha256(("mechanism96/" + key).encode()).digest()


def tiles():
    sources = {str(f): digest(f) for f in (CARD, MANIFEST, HELD, NEW)}
    archive = RR / "RRSHID.zip"
    sources["archive"] = read(RR / "registration.json")["SHA256"]
    if TILES.exists():
        if read(TILES)["sources"] != sources:
            raise ValueError("RRSHID selection changed")
        return
    if digest(archive) != sources["archive"]:
        raise ValueError("RRSHID archive changed")
    prior = [r for r in read(CARD)["records"] if r["dataset"] == "rrshid"] + read(MANIFEST)["pairs"]
    prior += read(HELD)["records"] + read(NEW)["records"]
    numbers = {r["id"].split("/")[-1] for r in prior}
    selected = []
    with ZipFile(archive) as file:
        items = pair_index(file, {"train"})
        signatures = [fingerprint(image(file, r["clean"])) for r in prior]
        for condition in CONDITIONS:
            candidates = sorted((i for _, c, i in items if c == condition), key=lambda i: rank(condition + i))
            take = 0
            for i in candidates:
                if i in numbers:
                    continue
                row = items["train", condition, i]
                clean = image(file, row["clean"])
                if clean.size != (256, 256) or image(file, row["haze"]).size != clean.size:
                    raise ValueError("Misaligned tile")
                p, d = fingerprint(clean)
                if any(np.count_nonzero(p != q) <= 8 and float(y @ d) >= .96 for q, y in signatures):
                    continue
                key = condition + "/" + i
                selected.append(dict(**row, id=key, group=key, condition=condition))
                numbers.add(i)
                signatures.append((p, d))
                take += 1
                if take == 30:
                    break
            if take != 30:
                raise ValueError(f"Insufficient unused tiles in {condition}")
    atomic_write(TILES, dict(sources=sources, prior_count=len(prior), records=selected,
                             sampling="30 per severity by filename hash. Exclude prior numeric IDs and visually "
                                      "similar reference tiles via 63-bit pHash distance <=8 and correlation >=.96."))
    print("reserved tiles", len(selected), flush=True)


def stage_one(job):
    with requests.Session() as session:
        return stage_item(job, ROOT / "frames", session)


def scenes():
    files = (*FOG_MANIFESTS, RAIN, SNOW, REPO.parent / "benchmark_data/muses-v67/staged-fog-val.json")
    signatures = {str(f): digest(f) for f in files}
    previous = [r for f in files for r in read(f).get("pairs", [])]
    old_groups = {r["group"] for r in previous}
    if SCENE_INDEX.exists():
        card = read(SCENE_INDEX)
        if card["sources"] != signatures:
            raise ValueError("Prior recording selection changed")
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            indexes = list(pool.map(lambda kind: archive_index(kind, ("val",), ("clear", "fog")), ARCHIVES))
        archives = {kind: site for kind, site, _ in indexes}
        catalogs = {kind: rows for kind, _, rows in indexes}
        rgb, gt = catalogs["frame_camera"], catalogs["gt_semantic"]
        if rgb.keys() != gt.keys():
            raise ValueError("Unmatched official MUSES labels")
        selection, cohorts = {}, {}
        for weather in ("fog", "clear"):
            groups = defaultdict(list)
            for key, r in rgb.items():
                if f"/{weather}/" in r["name"] and r["group"] not in old_groups:
                    groups[r["group"]].append(key)
            for values in groups.values():
                values.sort(key=rank)
            order = sorted(groups, key=rank)
            chosen = [groups[g][i] for i in range(4) for g in order if i < len(groups[g])][:48]
            cohorts[weather] = dict(eligible_groups=len(groups), selected=len(chosen))
            for k in chosen:
                selection[k] = {kind: rows[k] for kind, rows in catalogs.items()}
            old_groups.update(rgb[k]["group"] for k in chosen)
        if not selection:
            raise ValueError("No eligible new recording")
        card = dict(sources=signatures, cohorts=cohorts, archives=archives, selected=selection,
                    rule="Validation only, fog then clear, four frames maximum per recording, hash "
                         "round-robin, exclude all prior listed recordings. No outcome-based selection.")
        atomic_write(SCENE_INDEX, card)
    if SCENES.exists():
        doc = read(SCENES)
        if doc["selection_SHA256"] != digest(SCENE_INDEX):
            raise ValueError("Staged reservation changed")
        return
    jobs = [(k, v, card["archives"]) for k, v in card["selected"].items()]
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(stage_one, jobs))
    old_sha = {r["frame_camera"]["SHA256"] for r in previous}
    fresh = {r["frame_camera"]["SHA256"] for r in rows}
    if old_sha & fresh or len(fresh) != len(rows):
        raise ValueError("Repeated evaluation frame")
    for r in rows:
        r["weather"] = "fog" if "/fog/" in r["frame_camera"]["source"] else "clear"
    atomic_write(SCENES, dict(selection_SHA256=digest(SCENE_INDEX), records=rows,
                              groups=dict(Counter(r["group"] for r in rows)), cohorts=card["cohorts"]))
    print(card["cohorts"], flush=True)


def scene_cases():
    doc = read(SCENES)
    if digest(SCENE_INDEX) != doc["selection_SHA256"]:
        raise ValueError("Changed recording manifest")
    for r in doc["records"]:
        rgb, gt = (verified_image(r[k]["file"], r[k]["SHA256"], label)
                   for k, label in (("frame_camera", False), ("gt_semantic", True)))
        yield "new_muses_" + r["weather"], r["id"], r["group"], r["daytime"], rgb, torch.from_numpy(gt.copy()).cuda().long()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=["tiles", "scenes"])
    args = cli.parse_args()
    (tiles if args.part == "tiles" else scenes)()

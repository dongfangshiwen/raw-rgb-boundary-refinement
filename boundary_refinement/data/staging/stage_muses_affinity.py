"""Stage official training frames omitted when their recordings overlapped MUSES val."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from threading import local
import zipfile

import numpy as np
import requests
from urllib3.exceptions import HTTPError as RangeHTTPError

from boundary_refinement.data.cohorts.index_foggy_zurich_reserve import ZipByRange
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.staging.stage_muses_real_fog import INDEX as V70_INDEX, ROOT as V70, catalog, rank, thumbnail
from boundary_refinement.data.staging.stage_muses_recording import ROOT as V74
from boundary_refinement.data.staging.stage_muses import ARCHIVES, stage_item
from boundary_refinement.training.train_regional_guidance import atomic_write


ROOT = REPO.parent / "benchmark_data/muses-affinity-v75"
INDEX = ROOT / "registration.json"
MANIFEST = ROOT / "held.json"
LOCAL = local()


def register():
    old = read(V70_INDEX)
    expected = sorted(old["shared_official_recordings"])
    if INDEX.is_file():
        document = read(INDEX)
        if document["v70_SHA256"] != digest(V70_INDEX) or document["groups"] != expected:
            raise ValueError("The source of the MUSES recording holdout changed")
        return document
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = {key: (archive, frames) for key, archive, frames in pool.map(catalog, ARCHIVES)}
    rgb, gt = (sources[key][1] for key in ARCHIVES)
    if set(rgb) != set(gt):
        raise ValueError("Official MUSES RGB and semantic image identifiers differ")
    chosen = {key for key, entry in rgb.items() if "/train/" in entry["name"] and entry["group"] in expected}
    previous = {key for cohort in old["scopes"].values() for key in cohort}
    previous.update(r["id"] for scope in ("fit", "calibrate", "held") for r in read(V74 / f"{scope}.json")["pairs"])
    if chosen & previous or len(chosen) != old["train_frames_excluded"]:
        raise ValueError("Unexpected public recording cohort or prior frame reuse")
    doc = dict(archives={key: archive for key, (archive, _) in sources.items()},
               frames={key: {name: entries[key] for name, (_, entries) in sources.items()} for key in sorted(chosen, key=rank)},
               groups=expected, v70_SHA256=digest(V70_INDEX), created_utc=now(),
               selection="Use all official fog-train frames of the five recordings shared with official validation. "
                         "They have no frame IDs in the prior V70 and V74 experiment manifests. Their V70 validation "
                         "frames were previously evaluated. Full recordings remain out of V75 fitting and calibration.")
    atomic_write(INDEX, doc)
    return doc


def cached_labels(doc):
    archive = doc["archives"]["gt_semantic"]
    items = sorted((r["gt_semantic"] for r in doc["frames"].values()), key=lambda r: r["first"])
    directory = ROOT / "frames/gt_semantic"
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with requests.Session() as session:
            stream = ZipByRange(session, archive["url"], archive["size"])
            stream.etag = archive["etag"]
            with zipfile.ZipFile(stream) as packet:
                for item in items:
                    path = directory / PurePosixPath(item["name"]).name
                    if not path.is_file():
                        data = packet.read(item["name"])
                        pending = path.with_suffix(".download")
                        pending.write_bytes(data)
                        pending.replace(path)
                        written += 1
    except (OSError, RangeHTTPError, zipfile.BadZipFile) as error:
        print(dict(label_range_retry=type(error).__name__, cached=written), flush=True)


def fetch_one(key, pairs, sources):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    return stage_item((key, pairs, sources), directory=ROOT / "frames", session=LOCAL.session)


def run():
    doc = register()
    cached_labels(doc)
    records, fails = {}, {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        jobs = {pool.submit(fetch_one, key, entries, doc["archives"]): key for key, entries in doc["frames"].items()}
        for result in as_completed(jobs):
            key = jobs[result]
            try:
                records[key] = result.result()
            except (ValueError, OSError) as error:
                fails[key] = type(error).__name__
    if fails:
        raise OSError(f"Unstaged MUSES pairs: {len(fails)}, {dict(Counter(fails.values()))}")
    used = [(role, row, thumbnail(row)) for role in ("fit", "calibrate", "held")
            for row in read(V70 / f"{role}.json")["pairs"]]
    excluded, retained = {}, []
    for key in doc["frames"]:
        row = records[key]
        array, hashed = thumbnail(row)
        repeats = [dict(role=role, id=prior["id"]) for role, prior, (photo, mark) in used
                   if row["frame_camera"]["SHA256"] == prior["frame_camera"]["SHA256"] or
                   ((mark != hashed).mean() < .035 and np.abs(photo - array).mean() < .035)]
        if repeats:
            excluded[key] = repeats
        else:
            retained.append(row)
    groups = Counter(r["group"] for r in retained)
    if (set(groups) != set(doc["groups"]) or len(retained) < 100 or
            len({r["frame_camera"]["SHA256"] for r in retained}) != len(retained)):
        raise ValueError("The source audit found insufficient distinct holdout frames")
    card = dict(registration_SHA256=digest(INDEX), staged=len(records), groups=dict(groups), pairs=retained,
                excluded_near_repeat=excluded, v70_manifests={s: digest(V70 / f"{s}.json") for s in ("fit", "calibrate", "held")},
                finished_utc=now())
    atomic_write(MANIFEST, card)
    print(dict(staged=len(records), retained=len(retained), excluded=len(excluded), groups=dict(groups)), flush=True)


if __name__ == "__main__":
    run()

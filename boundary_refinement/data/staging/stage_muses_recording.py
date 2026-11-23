"""Hold out previously unused frames by entire MUSES training recording."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath

import numpy as np
import requests

from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.staging.stage_muses_real_fog import INDEX as PREVIOUS, ROOT as PRIOR_DATA, catalog, rank, thumbnail
from boundary_refinement.data.staging.stage_muses import ARCHIVES, stage_item
from boundary_refinement.training.train_regional_guidance import atomic_write


ROOT = REPO.parent / "benchmark_data/muses-recording-v74"
INDEX = ROOT / "registration.json"
SCOPES = ("fit", "calibrate", "held")


def register():
    if INDEX.exists():
        return read(INDEX)
    original = read(PREVIOUS)
    known = {key for scope in original["scopes"].values() for key in scope}
    earlier_evaluation = {row["frame_camera"]["group"] for row in original["scopes"]["held"].values()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = {m: (source, files) for m, source, files in pool.map(catalog, ARCHIVES)}
    images, labels = (found[k][1] for k in ARCHIVES)
    if set(images) != set(labels):
        raise ValueError("Official frame identities or semantic masks differ")
    eligible = {key: row for key, row in images.items()
                if "/train/" in row["name"] and row["group"] not in earlier_evaluation}
    novel = {key: row for key, row in eligible.items() if key not in known}
    held_groups = {r["group"] for r in novel.values()}
    retained = {key: r for key, r in eligible.items() if r["group"] not in held_groups}
    prior_calibration = {r["frame_camera"]["group"] for r in original["scopes"]["calibrate"].values()}
    calibration = set()
    for light in ("day", "night"):
        options = {v["group"] for v in retained.values() if v["daytime"] == light and v["group"] in prior_calibration}
        if not options:
            raise ValueError(f"No remaining previous calibration recording: {light}")
        calibration.add(min(options, key=rank))
    cohorts = dict(fit={i for i, v in retained.items() if v["group"] not in calibration},
                   calibrate={i for i, v in retained.items() if v["group"] in calibration}, held=set(novel))
    groups = {k: sorted({images[key]["group"] for key in v}) for k, v in cohorts.items()}
    if len(held_groups) < 3 or len(cohorts["fit"]) < 35 or sum(map(len, groups.values())) != len(set().union(*map(set, groups.values()))):
        raise ValueError(f"Insufficient independent MUSES recording groups: {groups}")
    packet = dict(prior_registration_SHA256=digest(PREVIOUS), archives={k: v[0] for k, v in found.items()}, recordings=groups,
                  counts={k: len(v) for k, v in cohorts.items()}, previous_frames=len(known),
                  cases={scope: {key: {kind: val[1][key] for kind, val in found.items()}
                                 for key in sorted(ids, key=rank)} for scope, ids in cohorts.items()},
                  scope="Use previously unscored frames absent from every V70 manifest as targets. Refit all adapters "
                        "from released weights and exclude the full corresponding recordings. Earlier experiments "
                        "used other frames of these recording IDs. This is a fresh-frame, recording-disjoint evaluation "
                        "for these models, with previous project exposure to the same routes.", created_utc=now())
    atomic_write(INDEX, packet)
    print(dict(counts=packet["counts"], recordings=groups), flush=True)
    return packet


def source_location(key, row, scope):
    for prior in ("fit", "calibrate"):
        directory = PRIOR_DATA / prior
        if all((directory / kind / PurePosixPath(v["name"]).name).is_file() for kind, v in row.items()):
            if scope == "held":
                raise ValueError(f"A held frame was previously staged: {key}")
            return directory
    return ROOT / scope


def prepare(job):
    scope, key, row, archives = job
    place = source_location(key, row, scope)
    with requests.Session() as network:
        return stage_item((key, row, archives), directory=place, session=network)


def stage(scope, packet):
    path = ROOT / f"{scope}.json"
    if path.is_file():
        if read(path)["registration_SHA256"] != digest(INDEX):
            raise ValueError(f"Existing {scope} data no longer match registration")
        return
    collected, failed = {}, {}
    items = packet["cases"][scope]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(prepare, (scope, key, row, packet["archives"])): key for key, row in items.items()}
        for task in as_completed(tasks):
            key = tasks[task]
            try:
                collected[key] = task.result()
            except (OSError, ValueError) as err:
                failed[key] = type(err).__name__
    if failed:
        raise OSError(f"MUSES bytes still missing for {scope}: {failed}")
    rows = [collected[key] for key in items]
    atomic_write(path, dict(pairs=rows, registration_SHA256=digest(INDEX), groups=dict(Counter(x["group"] for x in rows)),
                lighting=dict(Counter(x["daytime"] for x in rows)), finished_utc=now()))
    print(dict(scope=scope, staged=len(rows)), flush=True)


def audit():
    manifests = {scope: ROOT / f"{scope}.json" for scope in SCOPES}
    records, scopes = [], defaultdict(set)
    source = read(INDEX)
    for name, path in manifests.items():
        value = read(path)
        if value["registration_SHA256"] != digest(INDEX) or len(value["pairs"]) != source["counts"][name]:
            raise ValueError(f"Data enumeration differs from registration: {name}")
        for row in value["pairs"]:
            scopes[row["group"]].add(name)
            records.append((name, row, thumbnail(row)))
    if any(len(v) != 1 for v in scopes.values()):
        raise ValueError("A full recording crosses training and estimation")
    for i, (role, a, (x, h)) in enumerate(records):
        for other, b, (y, k) in records[i + 1:]:
            if role != other and (a["frame_camera"]["SHA256"] == b["frame_camera"]["SHA256"] or
                                  (h != k).mean() < .035 and np.abs(x - y).mean() < .035):
                raise ValueError(f"Near duplicates between {role}/{a['id']} and {other}/{b['id']}")
    card = dict(recordings_disjoint=True, counts=source["counts"], recordings=source["recordings"],
                manifests={k: digest(v) for k, v in manifests.items()}, registered_SHA256=digest(INDEX), finished_utc=now())
    atomic_write(ROOT / "audit.json", card)
    print(card, flush=True)


if __name__ == "__main__":
    packet = register()
    for name in SCOPES:
        stage(name, packet)
    audit()

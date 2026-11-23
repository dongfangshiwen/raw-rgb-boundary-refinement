"""Register real-fog fitting, calibration, and scoring by official MUSES recordings."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
import requests

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.stage_muses_v67 import ARCHIVES, MANIFEST as PRIOR, ROOT as LEGACY, SOURCE, archive_index, stage_item
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = REPO.parent / "benchmark_data/muses-v70"
INDEX = ROOT / "registration.json"
SIZE = dict(fit=96, calibrate=24, held=58)


def catalog(kind):
    return archive_index(kind, sections=("train", "val"))


def rank(key):
    return sha256(f"muses-v70/{key}".encode()).digest()


def round_robin(eligible, limit):
    by_group = defaultdict(list)
    for key, value in eligible.items():
        by_group[value["group"]].append(key)
    sequence = sorted(by_group, key=rank)
    for names in by_group.values():
        names.sort(key=rank)
    return [by_group[group][n] for n in range(max(map(len, by_group.values())))
            for group in sequence if n < len(by_group[group])][:limit]


def stage_registration():
    if INDEX.exists():
        return read(INDEX)
    with ThreadPoolExecutor(max_workers=2) as pool:
        catalogs = {kind: (source, names) for kind, source, names in pool.map(catalog, ARCHIVES)}
    sources = {kind: source for kind, (source, _) in catalogs.items()}
    images, labels = (catalogs[k][1] for k in ARCHIVES)
    if set(images) != set(labels):
        raise ValueError(f"Official real-fog PNG IDs differ: {len(images)} RGB, {len(labels)} labels")
    official = {name: {key: value for key, value in images.items() if PurePosixPath(value["name"]).parts[2] == name}
                for name in ("train", "val")}
    official_val = {r["group"] for r in official["val"].values()}
    shared = {r["group"] for r in official["train"].values()} & official_val
    eligible_train = {key: row for key, row in official["train"].items() if row["group"] not in official_val}
    group_lighting = defaultdict(set)
    for value in eligible_train.values():
        group_lighting[value["group"]].add(value["daytime"])
    # Reserve calibration recordings within each illumination stratum without inspecting labels.
    strata = defaultdict(list)
    for group, light in group_lighting.items():
        strata[tuple(sorted(light))].append(group)
    calibration = {group for items in strata.values() for group in sorted(items, key=rank)[:max(2, len(items) // 5)]}
    subsets = dict(
        fit=round_robin({k: v for k, v in eligible_train.items() if v["group"] not in calibration}, SIZE["fit"]),
        calibrate=round_robin({k: v for k, v in eligible_train.items() if v["group"] in calibration}, SIZE["calibrate"]),
        held=sorted(official["val"], key=rank),
    )
    expected = {name: len(keys) for name, keys in subsets.items()}
    if expected != SIZE:
        raise ValueError(f"Candidate sample sizes: {expected}. Remaining train: {len(eligible_train)}, "
                         f"recordings: {len(group_lighting)}, removed shared recordings: {sorted(shared)}")
    old_ids = {row["id"] for row in read(PRIOR)["pairs"]}
    if len(old_ids) != 20 or not old_ids <= set(subsets["held"]):
        raise ValueError("The earlier 20 scoring IDs must be a subset of the official validation set")
    packet = dict(source=SOURCE, archives=sources, previous_scoring_ids=sorted(old_ids),
                  previous_manifest_SHA256=digest(PRIOR), counts={name: len(rows) for name, rows in official.items()},
                  shared_official_recordings=sorted(shared), train_frames_excluded=len(official["train"]) - len(eligible_train),
                  scopes={name: {key: {kind: catalogs[kind][1][key] for kind in ARCHIVES} for key in keys}
                          for name, keys in subsets.items()},
                  plan="Official fog train provides every fitting and calibration image. Exclude every recording present in "
                       "official validation before splitting remaining recordings into fitting and calibration. "
                       "Take a fixed SHA256 order round-robin over recordings. Score the entire official fog val split. "
                       "Score first-use frames separately and retain all common recordings for grouped uncertainty.", created_utc=now())
    atomic_write(INDEX, packet)
    print(dict(official=packet["counts"], recorded=expected,
               groups={name: dict(Counter(images[key]["group"] for key in keys)) for name, keys in subsets.items()}), flush=True)
    return packet


def download_one(key, records, archives, territory):
    legacy = LEGACY / "fog-val"
    reusable = territory == "held" and all((legacy / kind / PurePosixPath(item["name"]).name).is_file() for kind, item in records.items())
    target = legacy if reusable else ROOT / territory
    with requests.Session() as network:
        row = stage_item((key, records, archives), directory=target, session=network)
    return row


def stage(scope, registered, workers):
    final = ROOT / f"{scope}.json"
    if final.exists():
        card = read(final)
        if card["registration_SHA256"] != digest(INDEX):
            raise ValueError(f"Previously staged {scope} manifest has another registration")
        print(dict(scope=scope, reused=len(card["pairs"])), flush=True)
        return
    source = registered["scopes"][scope]
    ready = {}
    failed = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(download_one, key, rec, registered["archives"], scope): key for key, rec in source.items()}
        for completed in as_completed(jobs):
            try:
                item = completed.result()
            except (OSError, ValueError) as error:
                failed[jobs[completed]] = type(error).__name__
                continue
            ready[item["id"]] = item
            if len(ready) % 12 == 0 or len(ready) == len(source):
                print(dict(scope=scope, ready=len(ready), total=len(source)), flush=True)
    if failed:
        raise OSError(f"Registered {scope} frames still missing: {failed}")
    pairs = [ready[key] for key in source]
    if len({row["frame_camera"]["SHA256"] for row in pairs}) != len(pairs):
        raise ValueError(f"Exact duplicates in {scope}")
    atomic_write(final, dict(scope=scope, pairs=pairs, groups=dict(Counter(row["group"] for row in pairs)),
                 lighting=dict(Counter(row["daytime"] for row in pairs)), registration_SHA256=digest(INDEX), finished_utc=now()))


def thumbnail(row):
    with Image.open(row["frame_camera"]["file"]) as picture:
        value = np.asarray(picture.convert("L").resize((33, 18), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    return value, value[:, 1:] > value[:, :-1]


def audit():
    packet = {scope: read(ROOT / f"{scope}.json") for scope in SIZE}
    if any(value["registration_SHA256"] != digest(INDEX) for value in packet.values()):
        raise ValueError("Data manifest does not match preregistration")
    annotations = defaultdict(set)
    signatures = {}
    for scope, note in packet.items():
        for row in note["pairs"]:
            if row["id"] in signatures:
                raise ValueError(f"Repeated frame ID {row['id']}")
            for kind in ARCHIVES:
                asset = Path(row[kind]["file"])
                if digest(asset) != row[kind]["SHA256"]:
                    raise ValueError(f"RGB or label changed: {asset.name}")
            signatures[row["id"]] = scope, row, thumbnail(row)
            annotations[row["group"]].add(scope)
    if any(len(scope) > 1 for scope in annotations.values()):
        raise ValueError("A recording crosses splits")
    suspects = []
    records = list(signatures.values())
    for i, (s, a, (image, sign)) in enumerate(records):
        for t, b, (pic, q) in records[i + 1:]:
            if s == t:
                continue
            shared = a["frame_camera"]["SHA256"] == b["frame_camera"]["SHA256"]
            near = (sign != q).mean() < .035 and np.abs(image - pic).mean() < .035
            if shared or near:
                suspects.append((s, a["id"], t, b["id"], bool(shared)))
    if suspects:
        raise ValueError(f"Possible cross-split duplicates: {suspects}")
    card = dict(scope={scope: {"images": len(note["pairs"]), "groups": note["groups"], "lighting": note["lighting"]}
                       for scope, note in packet.items()}, first_use_frames=SIZE["held"] - len(read(INDEX)["previous_scoring_ids"]),
                manifests={scope: digest(ROOT / f"{scope}.json") for scope in packet},
                cross_split_duplicate_screen="Exact file digests and conservative joint grayscale and dHash screen",
                recordings_disjoint=True, finished_utc=now())
    atomic_write(ROOT / "audit.json", card)
    print(card, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("register", "all", *SIZE, "audit"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    options = parser.parse_args()
    scope = options.scope
    if scope == "audit":
        audit()
    else:
        registered = stage_registration()
        selected = SIZE if scope == "all" else () if scope == "register" else (scope,)
        for group in selected:
            stage(group, registered, options.workers)
        if scope == "all":
            audit()

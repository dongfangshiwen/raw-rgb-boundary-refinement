"""Register all BDD100K frames shared by the fog attribute and dense-mask annotations."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
from urllib.parse import quote, urljoin

import numpy as np
from PIL import Image
import requests

from scripts.audit_real_fog_v17 import NAMES
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = REPO.parent / "benchmark_data/bdd-fog-v73"
INDEX = ROOT / "registration.json"
MANIFEST = ROOT / "held.json"
HUB = "https://huggingface.co"
REPOS = {"semantic": "dronefreak/BDD100K-Semantic-Segmentation", "weather": "dronefreak/BDD100K-Weather-Classification"}
FOLDERS = {"train": 9, "val": 2}
SEMANTIC_COUNTS = {"train": 7000, "val": 1000}
WEATHER_TRAIN_COUNTS = (69863, 70000)
LICENSE = "https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst"


def fetch(url):
    for attempt in range(4):
        try:
            with requests.get(url, timeout=(12, 35)) as response:
                response.raise_for_status()
                return response.content
        except requests.RequestException:
            if attempt == 3:
                raise


def path_at(repo, revision, name):
    return f"{HUB}/datasets/{repo}/resolve/{revision}/{quote(name, safe='/')}"


def metadata(job):
    split, shard, revision = job
    directory = f"data/images/{split}/shard_{shard:03d}"
    url = path_at(REPOS["weather"], revision, f"{directory}/metadata.jsonl")
    blob = fetch(url)
    records = [json.loads(line) for line in blob.splitlines() if line.strip()]
    return split, directory, records, dict(url=url, rows=len(records), SHA256=sha256(blob).hexdigest())


def tree(revision, directory):
    url = f'{HUB}/api/datasets/{REPOS["semantic"]}/tree/{revision}/{directory}?limit=1000'
    while url:
        with requests.get(url, timeout=(10, 30)) as r:
            r.raise_for_status()
            yield from r.json()
            next_page = r.links.get("next", {}).get("url")
        url = urljoin(url, next_page) if next_page else None
        if url and not url.startswith(f"{HUB}/api/datasets/"):
            raise ValueError("The mirror returned an unexpected pagination origin")


def semantic_files(job):
    revision, split, folder, suffix = job
    files = {}
    directories = [folder]
    while directories:
        directory = directories.pop()
        for row in tree(revision, directory):
            path = row["path"]
            if row["type"] == "directory":
                directories.append(path)
            elif path.endswith(suffix):
                key = PurePosixPath(path).name.removesuffix(suffix)
                if key in files:
                    raise ValueError(f"Multiple BDD100K semantic assets for {key}")
                files[key] = path_at(REPOS["semantic"], revision, path)
    return split, suffix, files


def register():
    if INDEX.exists():
        return read(INDEX)
    refs = {key: json.loads(fetch(f"{HUB}/api/datasets/{repo}"))["sha"] for key, repo in REPOS.items()}
    heads, fog, used, files, weather_counts = {}, {}, set(), [], Counter()
    jobs = [(s, j, refs["weather"]) for s, n in FOLDERS.items() for j in range(n)]
    sem_jobs = [(refs["semantic"], s, f"{folder}/{s}", ext) for s in SEMANTIC_COUNTS
                for folder, ext in (("data/images_10k/10k", ".jpg"), ("data/seg_maps/labels", "_train_id.png"))]
    with ThreadPoolExecutor(max_workers=5) as pool:
        sem_futures = [pool.submit(semantic_files, job) for job in sem_jobs]
        for split, directory, records, blob in pool.map(metadata, jobs):
            files.append(blob)
            weather_counts[split] += len(records)
            for row in records:
                name = PurePosixPath(row["file_name"])
                key = name.stem
                if name.name != row["file_name"] or name.suffix.lower() != ".jpg" or key in used:
                    raise ValueError(f"Unexpected source frame ID {row['file_name']}")
                used.add(key)
                if row["label"] == "foggy":
                    fog[key] = dict(split=split, url=path_at(REPOS["weather"], refs["weather"], f"{directory}/{name}"))
        catalog = {(split, ext): paths for split, ext, paths in (future.result() for future in sem_futures)}
    if weather_counts["train"] not in WEATHER_TRAIN_COUNTS or weather_counts["val"] != 10000:
        raise ValueError(f"Some BDD100K weather shards changed: {weather_counts}")
    for split, count in SEMANTIC_COUNTS.items():
        images, masks = (catalog[split, ext] for ext in (".jpg", "_train_id.png"))
        if len(images) != count or set(masks) != set(images) or set(heads) & set(images):
            raise ValueError(f"BDD semantic split differs from official counts: {split}, {len(images)}, {len(masks)}")
        heads.update({key: dict(split=split, image=url, mask=masks[key]) for key, url in images.items()})
    label_ids = json.loads(fetch(path_at(REPOS["semantic"], refs["semantic"], "id2label.json")))
    if tuple(label_ids[str(i)] for i in range(19)) != tuple(NAMES):
        raise ValueError("The mirror and Cityscapes train-ID taxonomy differs")
    joined = {key: {**heads[key], "weather": fog[key]} for key in sorted(heads.keys() & fog.keys())}
    if not joined:
        raise ValueError("The sequestered BDD subset contains no dense masks paired with the fog attribute")
    ROOT.mkdir(parents=True, exist_ok=True)
    registration = dict(source=REPOS, revisions=refs, metadata=files, label_ids=label_ids, weather_counts=dict(weather_counts), fog_attribute_images=len(fog),
                        semantic_images=len(heads), cases=joined, license=LICENSE,
                        selection="Take the complete intersection by original frame IDs. Both BDD100K labelled splits "
                                  "are held out from this project. Source clips are distinct by official image identity. "
                                  "Fog density and geographic separation between BDD clips are not independently asserted.",
                        created_utc=now())
    (ROOT / "SOURCE-LICENSE.rst").write_bytes(fetch(LICENSE))
    atomic_write(INDEX, registration)
    print(dict(registered=len(joined), splits=dict(Counter(r["split"] for r in joined.values())), revisions=refs), flush=True)
    return registration


def stage_one(job):
    key, item = job
    image, mask = ROOT / "rgb" / f"{key}.jpg", ROOT / "mask" / f"{key}.png"
    if not image.is_file():
        raw = fetch(item["image"])
        weather = fetch(item["weather"]["url"])
        if raw != weather:
            with Image.open(BytesIO(raw)) as a, Image.open(BytesIO(weather)) as b:
                if not np.array_equal(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB"))):
                    raise ValueError(f"Weather attribute and segmentation RGB differ: {key}")
        image.write_bytes(raw)
    if not mask.is_file():
        mask.write_bytes(fetch(item["mask"]))
    with Image.open(image) as a, Image.open(mask) as b:
        ids = np.asarray(b)
        if a.size != b.size or ids.ndim != 2 or not set(np.unique(ids)) <= {*range(19), 255}:
            raise ValueError(f"Invalid BDD100K sample {key}")
    return dict(id=key, group=key, split=item["split"], image=str(image), label=str(mask), image_SHA256=digest(image),
                label_SHA256=digest(mask), match="Official fog weather label and identical RGB in both task collections")


def stage():
    packet = register()
    if MANIFEST.exists():
        note = read(MANIFEST)
        if note["registration_SHA256"] != digest(INDEX):
            raise ValueError("The sealed BDD score population changed")
        return
    for name in ("rgb", "mask"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(stage_one, packet["cases"].items()))
    if len({row["image_SHA256"] for row in rows}) != len(rows):
        raise ValueError("BDD100K score data contain repeated RGB scenes")
    atomic_write(MANIFEST, dict(pairs=rows, registration_SHA256=digest(INDEX), licensing_SHA256=digest(ROOT / "SOURCE-LICENSE.rst"),
                 counts=dict(Counter(x["split"] for x in rows)), finished_utc=now()))
    print(dict(staged=len(rows), valid_masks=len(rows), unique_image_IDs=len(rows)), flush=True)


if __name__ == "__main__":
    stage()

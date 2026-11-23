"""Stage a registered sample of the public MUSES validation set without a bulk RGB download."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import struct
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
import requests

from scripts.index_foggy_zurich_reserve import ZipByRange
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = REPO.parent / "benchmark_data/muses-v67"
INDEX = ROOT / "registered-fog-val.json"
MANIFEST = ROOT / "staged-fog-val.json"
SOURCE = "https://muses.ethz.ch/MUSES_packages/"
ARCHIVES = {"frame_camera": ("frame_camera_trainvaltest.zip", 5662843611),
            "gt_semantic": ("gt_semantic_trainval.zip", 88330362)}
# The first view in each recording is fixed by hash. Round-robin selection across recordings limits temporal duplication.
LIMIT = 20


def archive_index(kind, sections=("val",), conditions=("fog",)):
    name, length = ARCHIVES[kind]
    with requests.Session() as session:
        data = ZipByRange(session, SOURCE + name, length)
        with zipfile.ZipFile(data) as archive:
            entries = sorted(archive.infolist(), key=lambda p: p.header_offset)
            result = {}
            suffix = "_gt_labelTrainIds.png" if kind == "gt_semantic" else "_frame_camera.png"
            for entry, following in zip(entries, [*entries[1:], None], strict=True):
                path = PurePosixPath(entry.filename)
                if kind not in path.parts or not path.name.endswith(suffix):
                    continue
                section, weather, daytime = path.parts[path.parts.index(kind) + 1:-1]
                if section not in sections or weather not in conditions:
                    continue
                key = path.name.removesuffix(suffix)
                group = key.split("_frame_", maxsplit=1)[0]
                if "_frame_" not in key or key in result:
                    raise ValueError(f"Unexpected public validation name: {entry.filename}")
                end = following.header_offset if following is not None else archive.start_dir
                result[key] = dict(name=entry.filename, group=group, daytime=daytime,
                                   first=entry.header_offset, last=end - 1, size=entry.file_size,
                                   compressed=entry.compress_size, method=entry.compress_type, crc=entry.CRC)
        return kind, dict(url=SOURCE + name, size=length, etag=data.etag), result


def register():
    if INDEX.exists():
        return read(INDEX)
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = list(pool.map(archive_index, ARCHIVES))
    urls, catalog = {}, {}
    for kind, source, entries in found:
        urls[kind], catalog[kind] = source, entries
    images, labels = (catalog[k] for k in ARCHIVES)
    if set(images) != set(labels) or len(images) < LIMIT:
        raise ValueError(f"Public validation RGB and semantic files differ: {len(images)}, {len(labels)}")
    by_group = defaultdict(list)
    for key, value in images.items():
        by_group[value["group"]].append(key)
    key_hash = lambda name: sha256(("v67/fog-validation/" + name).encode()).digest()
    groups = sorted(by_group, key=key_hash)
    for values in by_group.values():
        values.sort(key=key_hash)
    keys = [by_group[g][i] for i in range(max(map(len, by_group.values()))) for g in groups if i < len(by_group[g])][:LIMIT]
    selected = {key: {kind: catalog[kind][key] for kind in ARCHIVES} for key in keys}
    packet = dict(source=SOURCE, data=urls, count=len(images), sequence_count=len(groups), sample=selected,
                  selection="Pre-registered 20 fog validation views, ranked by fixed SHA256 within recording, "
                            "one per recording per round. No labels or model predictions consulted during selection.",
                  created_utc=now())
    atomic_write(INDEX, packet)
    print(dict(public_val=len(images), selected=len(keys), recordings=len({images[k]['group'] for k in keys})), flush=True)
    return packet


def load_member(site, info, session=requests):
    headers = {"Range": f'bytes={info["first"]}-{info["last"]}', "If-Range": site["etag"], "Accept-Encoding": "identity"}
    expected = f'bytes {info["first"]}-{info["last"]}/{site["size"]}'
    for attempt in range(5):
        try:
            with session.get(site["url"], headers=headers, stream=True, timeout=(12, 40)) as response:
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError(f"Unexpected range response {response.status_code} for {info['name']}")
                if response.headers.get("ETag") != site["etag"]:
                    raise ValueError("The source archive revision changed")
                data = response.content
            break
        except requests.exceptions.RequestException as error:
            if attempt == 4:
                raise
            print(dict(retry=info["name"], cause=type(error).__name__), flush=True)
            time.sleep(1 + 2 * attempt)
    if data[:4] != b"PK\x03\x04" or len(data) != info["last"] - info["first"] + 1:
        raise ValueError("The server delivered an incomplete ZIP member")
    name, extra = struct.unpack_from("<HH", data, 26)
    offset = 30 + name + extra
    payload = data[offset:offset + info["compressed"]]
    if info["method"] == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(payload, -15)
    elif info["method"] == zipfile.ZIP_STORED:
        raw = payload
    else:
        raise ValueError("Unexpected compression format")
    if len(raw) != info["size"] or zlib.crc32(raw) != info["crc"]:
        raise ValueError("The selected archive entry failed CRC validation")
    return raw


def stage_item(job, directory=ROOT / "fog-val", session=requests):
    key, records, urls = job
    row = dict(id=key, group=records["frame_camera"]["group"], daytime=records["frame_camera"]["daytime"])
    dimensions = {}
    for kind, item in records.items():
        stem = PurePosixPath(item["name"]).name
        file = directory / kind / stem
        if file.exists():
            raw = file.read_bytes()
            if len(raw) != item["size"] or zlib.crc32(raw) != item["crc"]:
                raise ValueError(f"An existing public sample changed: {stem}")
        else:
            raw = load_member(urls[kind], item, session)
            file.parent.mkdir(parents=True, exist_ok=True)
            pending = file.with_suffix(".download")
            pending.write_bytes(raw)
            pending.replace(file)
        with Image.open(BytesIO(raw)) as picture:
            dimensions[kind] = picture.size
            if kind == "gt_semantic":
                mask = np.asarray(picture)
                if mask.ndim != 2:
                    raise ValueError(f"Expected the semantic PNG IDs: {stem}")
                row["observed_label_ids"] = np.unique(mask).tolist()
                if not set(row["observed_label_ids"]) <= {*range(19), 255}:
                    raise ValueError(f"Unexpected Cityscapes train IDs: {stem}")
        row[kind] = dict(file=str(file), SHA256=digest(file), source=item["name"])
    if len(set(dimensions.values())) != 1:
        raise ValueError(f"The MUSES validation pair is misaligned: {key}")
    print(dict(staged=key, group=row["group"], daytime=row["daytime"]), flush=True)
    return row


def stage():
    packet = register()
    expected = [(key, row, packet["data"]) for key, row in packet["sample"].items()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(stage_item, expected))
    if len(pairs) != len(expected) or len({r["frame_camera"]["SHA256"] for r in pairs}) != len(pairs):
        raise ValueError("Stage all preregistered photographs once")
    atomic_write(MANIFEST, dict(source=SOURCE, index_SHA256=digest(INDEX), pairs=pairs,
                 groups=dict(Counter(r["group"] for r in pairs)), daytime=dict(Counter(r["daytime"] for r in pairs)),
                 finished_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "stage"))
    {"register": register, "stage": stage}[parser.parse_args().stage]()

"""Extract evenly spaced, paired fog frames from five official Virtual KITTI 2 locations."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import re
import tarfile
import threading
import time

import numpy as np
from PIL import Image
import requests
import torch

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = REPO.parent / "benchmark_data/vkitti-fog-v93"
METH = REPO / "results/subpixel-transport-v92/completed.json"
LOCK, HELD = ROOT / "protocol.json", ROOT / "held.json"
PAGE = "https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/"
SITE = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_"
SIZE = {"rgb": 7532472320, "classSegmentation": 1015961600}
SCENES = ("Scene01", "Scene02", "Scene06", "Scene18", "Scene20")
BLOCK = 2 ** 21
PATTERN = re.compile(r"(Scene[0-9]+)/([^/]+)/frames/([^/]+)/(Camera_[01])/(?:rgb|classgt)_([0-9]{5})[.](?:jpg|png)")
PALETTE = {
    (210, 0, 200): 9, (90, 200, 255): 10, (0, 199, 0): 8, (90, 240, 0): 8,
    (140, 140, 140): 2, (100, 60, 100): 0, (250, 100, 255): 255, (255, 255, 0): 7,
    (200, 200, 0): 6, (255, 130, 0): 5, (80, 80, 80): 255, (160, 60, 60): 14,
    (255, 127, 80): 13, (0, 139, 139): 255, (0, 0, 0): 255,
}
CLASSES = tuple(sorted(set(PALETTE.values()) - {255}))
LOCAL = threading.local()


def register():
    cfg = dict(source=PAGE, attribution="Naver Corporation, Virtual KITTI 2 (Cabon, Murray, Humenberger); "
               "adaptation of the Virtual KITTI dataset (Gaidon et al.). CC BY-NC-SA 3.0.",
               archives={k: dict(url=SITE + k + ".tar", bytes=n) for k, n in SIZE.items()},
               scenes=list(SCENES), weather="fog", camera="Camera_0", subset=12,
               selection="All five synthetic scenes. In each, select the midpoint index of each of twelve equal "
               "intervals of the sorted complete fog frame IDs. Select by filename only before reading pixels. "
               "All frozen model weights predate registration; this auxiliary endpoint precedes scoring these scenes.",
               endpoints="Boundary-band and full common-ten-class mean IoU. Inspect traffic light IoU and positive "
               "scene count separately. Keep scene IDs as the resampling unit. Do not interpret as real fog.",
               label_map={",".join(map(str, color)): k for color, k in PALETTE.items()},
               model_SHA256=digest(METH), source_SHA256=digest(__file__))
    if LOCK.exists() and read(LOCK) != cfg:
        raise ValueError("Virtual KITTI protocol changed")
    atomic_write(LOCK, cfg)


def fetch(kind, start, end):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(6):
        try:
            with LOCAL.session.get(SITE + kind + ".tar", headers={"Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity"}, stream=True, timeout=(12, 40)) as reply:
                if reply.status_code != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE[kind]}":
                    raise ValueError(f"Unexpected official archive response: {reply.status_code}")
                raw = reply.content
            if len(raw) != end + 1 - start:
                raise ValueError("Incomplete byte range")
            return raw
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(1 + attempt)


def headers(blob):
    for off in range(0, len(blob) - 511, 512):
        if not blob.startswith(b"Scene", off):
            continue
        try:
            item = tarfile.TarInfo.frombuf(blob[off:off + 512], "utf-8", "strict")
        except tarfile.HeaderError:
            continue
        match = PATTERN.fullmatch(item.name)
        if item.isfile() and match:
            yield off, item, match.groups()


@lru_cache(maxsize=400)
def first_key(kind, index):
    start = index * BLOCK
    for length in (256 * 1024, BLOCK):
        end = min(start + length, SIZE[kind]) - 1
        if start > end:
            return ("~", "~", "~")
        for _, _, (scene, weather, _, camera, _) in headers(fetch(kind, start, end)):
            return scene, weather, camera
        if end + 1 == SIZE[kind]:
            return ("~", "~", "~")
    raise ValueError("No valid TAR file header in 2 MiB probe")


def first_block(kind, key):
    left, right = 0, (SIZE[kind] + BLOCK - 1) // BLOCK
    while left < right:
        middle = (left + right) // 2
        if first_key(kind, middle) < key:
            left = middle + 1
        else:
            right = middle
    return left


def select(job):
    kind, scene = job
    output = ROOT / f"{scene}-{kind}.json"
    if output.exists():
        data = read(output)
        if data["protocol_SHA256"] != digest(LOCK) or any(digest(r["file"]) != r["SHA256"] for r in data["selected"].values()):
            raise ValueError(f"Staged data changed: {scene}/{kind}")
        return scene, kind, data
    lower = max(0, first_block(kind, (scene, "fog", "Camera_0")) - 1) * BLOCK
    upper = min(SIZE[kind], (first_block(kind, (scene, "fog", "Camera_1")) + 1) * BLOCK)
    if upper <= lower:
        raise ValueError("Fog camera directory is missing")
    parts = [fetch(kind, byte, min(byte + 4 * BLOCK, upper) - 1) for byte in range(lower, upper, 4 * BLOCK)]
    content = b"".join(parts)
    members = {int(number): (off, item) for off, item, (location, weather, typ, camera, number) in headers(content)
               if (location, weather, typ, camera) == (scene, "fog", kind, "Camera_0")}
    ids = sorted(members)
    if len(ids) < 12 or ids != list(range(len(ids))):
        raise ValueError(f"Incomplete camera directory {scene}/{kind}")
    after = [(loc, weather, cam) for _, _, (loc, weather, _, cam, _) in headers(content)]
    if not (any(k < (scene, "fog", "Camera_0") for k in after) and any(k > (scene, "fog", "Camera_0") for k in after)):
        raise ValueError("Camera directory was not bracketed by both neighbors")
    sample = [ids[(2 * j + 1) * len(ids) // 24] for j in range(12)]
    saved = {}
    for j in sample:
        offset, item = members[j]
        file = ROOT / scene / kind / Path(item.name).name
        file.parent.mkdir(parents=True, exist_ok=True)
        raw = content[offset + 512:offset + 512 + item.size]
        if len(raw) != item.size:
            raise ValueError("Truncated TAR entry")
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        file.write_bytes(raw)
        saved[str(j)] = dict(file=str(file), SHA256=digest(file), member=item.name, original_bytes=item.size,
                             header_offset=lower + offset)
    data = dict(protocol_SHA256=digest(LOCK), frames=len(ids), selected=saved)
    atomic_write(output, data)
    print(dict(scene=scene, archive=kind, complete_camera=len(ids), selected=len(sample)), flush=True)
    return scene, kind, data


def prepare():
    register()
    jobs = [(k, scene) for k in SIZE for scene in SCENES]
    with ThreadPoolExecutor(max_workers=8) as pool:
        packets = {(scene, k): card for scene, k, card in pool.map(select, jobs)}
    rows = []
    for scene in SCENES:
        rgb, gt = (packets[scene, k] for k in SIZE)
        if rgb["frames"] != gt["frames"] or rgb["selected"].keys() != gt["selected"].keys():
            raise ValueError("Image and semantic TAR selections disagree")
        for frame, img in rgb["selected"].items():
            mask = gt["selected"][frame]
            with Image.open(img["file"]) as im, Image.open(mask["file"]) as truth:
                if im.size != truth.size:
                    raise ValueError("Virtual KITTI label shape mismatch")
                pairs = truth.convert("RGB").getcolors(maxcolors=32)
                if not pairs or any(color not in PALETTE for _, color in pairs):
                    raise ValueError("Unknown official label colors")
            rows.append(dict(id=f"{scene}/{int(frame):05d}", group=scene, rgb=img, mask=mask))
    if len({r["rgb"]["SHA256"] for r in rows}) != 60:
        raise ValueError("Paired synthetic data have duplicate frames")
    atomic_write(HELD, dict(pairs=rows, registration_SHA256=digest(LOCK), counts={"vkitti_fog": len(rows)}, utc=now()))


def cases():
    card = read(HELD)
    if card["registration_SHA256"] != digest(LOCK):
        raise ValueError("Synthetic protocol mismatch")
    for row in card["pairs"]:
        for data in (row["rgb"], row["mask"]):
            if digest(data["file"]) != data["SHA256"]:
                raise ValueError(f"Image changed for {row['id']}")
        with Image.open(row["rgb"]["file"]) as image, Image.open(row["mask"]["file"]) as mask:
            rgb, colors = np.array(image.convert("RGB")), np.array(mask.convert("RGB"))
        encoded = colors[..., 0].astype(np.uint32) * 65536 + colors[..., 1].astype(np.uint32) * 256 + colors[..., 2]
        labels = np.full(colors.shape[:2], 255, np.uint8)
        for (r, g, b), cls in PALETTE.items():
            labels[encoded == (r << 16 | g << 8 | b)] = cls
        yield "vkitti_fog", row["id"], row["group"], "synthetic fog", rgb, torch.from_numpy(labels).cuda().long()


if __name__ == "__main__":
    prepare()

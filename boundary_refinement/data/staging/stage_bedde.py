"""Download and index the BeDDE authors' public archive for held-out evaluation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
import re
import subprocess
import time

import requests

from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.staging.stage_preview_training import ROOT
from boundary_refinement.training.train_regional_guidance import atomic_write


STORE = REPO.parent / "benchmark_data/bedde-v28"
AUTHOR = "https://github.com/xiaofeng94/BeDDE-for-defogging"
DOWNLOAD = "https://drive.usercontent.google.com/download?id=12p-MY2ZygT5Tl8q0oFxDIUg9B5Jn042-&export=download&confirm=t"
SIZE = 483354633
BLOCK = 4 * 2**20
ARCHIVE = STORE / "BeDDE.rar"
RECEIPT = ROOT / "bedde_archive.json"


def block(start):
    end = min(start + BLOCK, SIZE) - 1
    received = bytearray()
    with requests.Session() as session:
        for attempt in range(6):
            begin = start + len(received)
            try:
                with session.get(DOWNLOAD, headers={"Range": f"bytes={begin}-{end}", "Accept-Encoding": "identity"},
                                 stream=True, timeout=(12, 24)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {begin}-{end}/{SIZE}":
                        raise OSError(f"Unexpected BeDDE range response {response.status_code}")
                    for packet in response.iter_content(65536):
                        received.extend(packet)
                if len(received) != end - start + 1:
                    raise OSError("Incomplete BeDDE block")
                return start, received
            except (requests.RequestException, OSError):
                if attempt == 5:
                    raise
                time.sleep(min(attempt+1, 3))
    raise RuntimeError("Unreachable BeDDE download state")


def download():
    pending = ARCHIVE.with_suffix(".partial")
    progress = STORE / "blocks.json"
    starts = list(range(0, SIZE, BLOCK))
    complete = set(read(progress)["offsets"]) if progress.exists() and pending.exists() else set()
    with pending.open("r+b" if complete else "w+b") as out:
        out.truncate(SIZE)
        with ThreadPoolExecutor(max_workers=16) as pool:
            workers = [pool.submit(block, start) for start in starts if start not in complete]
            for index, job in enumerate(as_completed(workers), 1):
                start, payload = job.result()
                out.seek(start)
                out.write(payload)
                out.flush()
                complete.add(start)
                atomic_write(progress, dict(offsets=sorted(complete)))
                if index % 12 == 0 or len(complete) == len(starts):
                    print(dict(downloaded_blocks=len(complete), total=len(starts)), flush=True)
        out.seek(0)
        if out.read(7) != b"Rar!\x1a\x07\x00" or complete != set(starts):
            raise ValueError("Incomplete or unexpected author RAR archive")
    pending.replace(ARCHIVE)


def index_and_extract():
    # List regular image members first. Pass this explicit list back to the native archive reader.
    names = subprocess.run(["tar.exe", "-tf", str(ARCHIVE)], check=True, capture_output=True, text=True,
                           encoding="utf-8").stdout.splitlines()
    patterns = re.compile(r"[A-Za-z0-9_ .()-]+")
    accepted = []
    for name in names:
        path = PurePosixPath(name)
        if len(path.parts) < 3:
            continue
        role = path.parts[-2].lower()
        suffix = path.suffix.lower()
        if (role not in ("fog","gt") or suffix not in (".jpg",".jpeg",".png",".bmp")) and (role != "mask" or suffix != ".mat"):
            continue
        if any(not patterns.fullmatch(v) or v in (".", "..") for v in path.parts):
            raise ValueError("Archive image entry has an unsafe path")
        accepted.append(name)
    if not accepted or len(set(accepted)) != len(accepted):
        raise ValueError("No unique BeDDE fog, clear or mask images found")
    file_index = STORE / "accepted.txt"
    file_index.write_text("\n".join(accepted) + "\n", encoding="utf-8")
    destination = STORE / "images"
    destination.mkdir(exist_ok=True)
    subprocess.run(["tar.exe", "-xf", str(ARCHIVE), "-C", str(destination), "-T", str(file_index)], check=True)
    paths = [destination / PurePosixPath(name) for name in accepted]
    if not all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(destination.resolve()) for p in paths):
        raise ValueError("Missing or redirected BeDDE image")
    return destination, accepted


def main():
    STORE.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists() and any(name.endswith(".mat") for name in read(RECEIPT)["members"]):
        card = read(RECEIPT)
        if digest(ARCHIVE) != card["SHA256"]:
            raise ValueError("BeDDE author archive changed")
        print(dict(indexed=len(card["members"]), archive=str(ARCHIVE)), flush=True)
        return card
    if not ARCHIVE.exists():
        download()
    destination, members = index_and_extract()
    result = dict(source=AUTHOR, link=DOWNLOAD, size=SIZE, SHA256=digest(ARCHIVE), directory=str(destination),
                  members=members, image_hashes={name: digest(destination / PurePosixPath(name)) for name in members},
                  scope="Author BeDDE archive. All available paired images and masks reserved for evaluation.",
                  finished_utc=now())
    atomic_write(RECEIPT, result)
    print(dict(indexed=len(members), archive=str(ARCHIVE)), flush=True)
    return result


if __name__ == "__main__":
    main()

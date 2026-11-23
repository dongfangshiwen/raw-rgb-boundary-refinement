"""Register and verify the author's paired Unreal Engine night images for transfer evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
from io import BytesIO
import re
from time import sleep
from urllib.request import Request, urlopen

from PIL import Image

from boundary_refinement.data.cohorts.fresh_real import groups
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, long_edge
from boundary_refinement.training.train_regional_guidance import atomic_write


ROOT = REPO.parent / "benchmark_data/unreal-nh-v57"
MANIFEST = ROOT / "manifest.json"
PAPER = "https://www.mdpi.com/2313-433X/9/8/153"
RELEASE = "https://github.com/madone7/Nighttime-Image-Dehazing-by-Render"
FOLDERS = {"haze": "1AUFE6Ylkhm_Uo5pnXa05gLVVzwKeyKfb", "clean": "1Y9Ymz3v_ZTVcineOX1ICIu7Lu5hvICwz"}
USER_AGENT = "Mozilla/5.0"


def retrieve(url):
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=22) as response:
        return response.read()


def folder(identity):
    text = retrieve("https://drive.google.com/embeddedfolderview?id=" + identity).decode("utf-8")
    entries = re.findall(
        r'<a[^>]+href="https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)/[^"]+"[^>]*>[\s\S]{0,700}?flip-entry-title">([^<]+)',
        text
    )
    result = {unescape(name).strip(): file_id for file_id, name in entries}
    if len(result) != len(entries) or not all(re.fullmatch(r"\d+\.png", n) for n in result):
        raise ValueError("Unexpected names in the published Unreal-NH folder")
    return result


def download(entry):
    name, role, file_id = entry
    dest = ROOT / role / name
    if dest.is_file():
        data = dest.read_bytes()
    else:
        for attempt in range(3):
            try:
                data = retrieve(f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("The release link did not return a PNG")
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                sleep(.75 * (attempt + 1))
        part = dest.with_suffix(".pending")
        part.write_bytes(data)
        part.replace(dest)
    with Image.open(BytesIO(data)) as frame:
        size = list(frame.size)
        frame.verify()
    return dict(file_id=file_id, SHA256=sha256(data).hexdigest(), bytes=len(data), size=size)


def stage():
    if MANIFEST.is_file():
        saved = read(MANIFEST)
        for row in saved["records"]:
            for role in FOLDERS:
                if digest(ROOT / role / row["id"]) != row[role]["SHA256"]:
                    raise ValueError("An Unreal-NH input or reference changed")
        return saved
    with ThreadPoolExecutor(max_workers=8) as workers:
        links = dict(zip(FOLDERS, workers.map(folder, FOLDERS.values()), strict=True))
        if links["haze"].keys() != links["clean"].keys() or len(links["haze"]) != 94:
            raise ValueError("Unexpected number or alignment of published Unreal-NH pairs")
        names = sorted(links["haze"])
        for role in FOLDERS:
            (ROOT / role).mkdir(parents=True, exist_ok=True)
        tasks = [(n, r, links[r][n]) for n in names for r in FOLDERS]
        receipts = {}
        for i, (entry, details) in enumerate(zip(tasks, workers.map(download, tasks), strict=True), start=1):
            receipts[entry[:2]] = details
            if i % 30 == 0:
                print(dict(verified_files=i, total=len(tasks)), flush=True)
    records = []
    photos = []
    for n in names:
        row = dict(id=n, **{role: receipts[n, role] for role in FOLDERS})
        if row["haze"]["size"] != row["clean"]["size"]:
            raise ValueError(f"Unmatched registration for {n}")
        records.append(row)
        with Image.open(ROOT / "clean" / n) as opened:
            photos.append(long_edge(opened.convert("RGB"), 256))
    clusters = groups(records, photos)
    for row in records:
        row["group"] = clusters[row["id"]]
    saved = dict(paper=PAPER, release=RELEASE, drive_folders=FOLDERS, records=records, count=len(records),
                 visual_groups=len(set(clusters.values())), code_SHA256=digest(__file__), locked_utc=now(),
                 scope="Use the 94 matched files from the author's ue/input and ue/target folders only. "
                       "Evaluation only for V56 and V57, rendered night haze, resize matched grids to at most "
                       "1024 pixels on the long side. The author folder has no split metadata.")
    atomic_write(MANIFEST, saved)
    print(dict(count=saved["count"], groups=saved["visual_groups"]), flush=True)
    return saved


def pairs(packet):
    for row in packet["records"]:
        images = []
        for role in FOLDERS:
            with Image.open(ROOT / role / row["id"]) as opened:
                frame = opened.convert("RGB")
                images.append(image_tensor(long_edge(frame, min(1024, max(frame.size)))))
        yield "unreal_nh", row["id"], row["group"], *images, None, "rendered_night"


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.parse_args()
    stage()

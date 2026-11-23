"""Lock both published real-haze RGB renderings and the official closer-pair subset."""
import csv
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

from PIL import Image

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor
from scripts.train_regional_guidance_v18 import atomic_write


SOURCE = REPO.parent / "benchmark_data/realhyper-v52"
PAPER = "https://www.mdpi.com/2313-433X/11/12/422"
DATASET = "https://huggingface.co/datasets/nikos74/RRealHyperPDID"
CARD = REPO / "results/orthogonal-risk-v52/realhyper_manifest.json"
COUNTS = dict(realhyper_csnc=161, realhyper_csso=161)


def archive_manifest():
    card = read(SOURCE / "downloads.json")
    author = read(SOURCE / "author_split.json")
    files = {r["name"]: r["SHA256"] for r in card["files"]}
    if digest(SOURCE / "RGB.zip") != files["RGB.zip"] or digest(SOURCE / "official_closest.csv") != author["subset_SHA256"]:
        raise ValueError("The published evaluation inputs differ from the pinned registry")
    return card, author


def archive_images(archive):
    index = defaultdict(lambda: defaultdict(list))
    for item in archive.infolist():
        p = PurePosixPath(item.filename)
        if p.suffix not in (".png", ".PNG") or len(p.parts) != 4:
            continue
        top, rendering, crop, file = p.parts
        role = p.stem.rsplit("_", 1)[-1]
        if top != "RGB" or rendering not in ("CSNC", "CSSO") or role not in ("hazed", "clean"):
            raise ValueError("Unexpected public RGB archive member")
        index[rendering, crop][role].append(item.filename)
    return index


def closest_index():
    with (SOURCE / "official_closest.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {(r["Set"].upper() + r["Crop"].upper(), r["Hazed-Dehazed"], r["Clean"]) for r in csv.DictReader(handle)}


def stage():
    source, author = archive_manifest()
    close = closest_index()
    with ZipFile(SOURCE / "RGB.zip") as rgb:
        index = archive_images(rgb)
        written = []
        for (rendering, crop), match in sorted(index.items()):
            if len(match["hazed"]) != 1 or not match["clean"]:
                raise ValueError("A crop must have one hazy acquisition and at least one published clean reference")
            hazy = match["hazed"][0]
            haze_id = PurePosixPath(hazy).stem.removesuffix("_hazed")
            for target in sorted(match["clean"]):
                clear_id = PurePosixPath(target).stem.removesuffix("_clean")
                exact = (crop, haze_id, clear_id) in close
                written.append(dict(dataset="realhyper_" + rendering.lower(), site=crop.split("C", 1)[0], crop=crop,
                    id=crop + "/" + clear_id, flight=haze_id, haze=hazy, clean=target, officially_closest=exact,
                    hazy_CRC32=f"{rgb.getinfo(hazy).CRC:08x}", clean_CRC32=f"{rgb.getinfo(target).CRC:08x}"))
    count = Counter(p["dataset"] for p in written)
    close_counts = Counter(p["dataset"] for p in written if p["officially_closest"])
    site_counts = {d: len({r["site"] for r in written if r["dataset"] == d}) for d in count}
    if count != COUNTS or set(site_counts.values()) != {17} or any(close_counts[d] != len(close) for d in count):
        raise ValueError(f"Unexpected public evaluation groups {count} {site_counts} {close_counts}")
    manifest = dict(paper=PAPER, data=DATASET, license="CC BY 4.0", source=source, author_meta=author,
        counts=dict(count), closest=dict(close_counts), sites=site_counts,
        input_crops={d: len({r["crop"] for r in written if r["dataset"] == d}) for d in count},
        rule="Pair each hazy tile with all clear references in its own crop, as the author does. Report both RGB rendering methods separately and group inference by site. The closest subset uses the author CSV. These are paired captures under different acquisition conditions, with possible temporal scene changes.",
        selection="Only evaluate this release. No parameter fitting or radius selection uses any of these images.",
        pairs=written, finished_utc=now())
    if CARD.exists():
        prior = read(CARD)
        if prior["pairs"] != manifest["pairs"]:
            raise ValueError("Locked external scenes changed")
        return prior
    CARD.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(CARD, manifest)
    print(dict(all_references=dict(count), closest=dict(close_counts), sites=site_counts, crops=manifest["input_crops"]), flush=True)
    return manifest


def pairs():
    source = stage()
    def pixels(archive, name):
        with Image.open(BytesIO(archive.read(name))) as opened:
            return image_tensor(opened.convert("RGB"))
    with ZipFile(SOURCE / "RGB.zip") as z:
        previous, haze = None, None
        for row in source["pairs"]:
            if row["haze"] != previous:
                haze, previous = pixels(z, row["haze"]), row["haze"]
            clear = pixels(z, row["clean"])
            if clear.shape != haze.shape or clear.shape != (3, 256, 256):
                raise ValueError("Expected the original RGB reference grid")
            yield row, haze, clear


if __name__ == "__main__":
    stage()

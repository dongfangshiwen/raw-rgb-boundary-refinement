"""Register the public RRSHID test and paired NT-HAZE development releases."""
import argparse
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile

from PIL import Image
import numpy as np

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor, long_edge
from scripts.real_haze_data_v22 import fingerprint
from scripts.train_regional_guidance_v18 import atomic_write


STORE = REPO.parent / "benchmark_data"
RR = STORE / "rrshid-v53"
NT = STORE / "nt-haze-v53"
CARD = REPO / "results/overlap-quadratic-v53/fresh_data.json"
COUNTS = dict(rrshid=304, nt_haze=25)


def image(archive, name):
    with Image.open(BytesIO(archive.read(name))) as opened:
        return opened.convert("RGB")


def groups(pairs, images):
    features = [fingerprint(v) for v in images]
    parents = list(range(len(features)))
    def head(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i, (p, n) in enumerate(features):
        for j in range(i):
            q, m = features[j]
            if np.count_nonzero(p != q) <= 8 and float(n @ m) >= .96:
                parents[head(i)] = head(j)
    return {row["id"]: pairs[head(i)]["id"] for i, row in enumerate(pairs)}


def registration():
    rr = read(RR / "registration.json")
    nt = read(NT / "registration.json")
    if digest(RR / "RRSHID.zip") != rr["SHA256"] or any(digest(NT / row["archive"]) != row["SHA256"] for row in nt["data"]):
        raise ValueError("The pinned photometric benchmark archives changed")
    return rr, nt


def stage():
    rr, nt = registration()
    if CARD.exists():
        data = read(CARD)
        if data["archives"] != [rr, nt] or data["code_SHA256"] != digest(__file__):
            raise ValueError("The verified independent evaluation registry changed")
        return data
    records = []
    sizes = Counter()
    with ZipFile(RR / "RRSHID.zip") as z:
        matched = defaultdict(dict)
        for name in z.namelist():
            n = PurePosixPath(name)
            if len(n.parts) != 4 or n.parts[1] != "test" or n.suffix.lower() != ".png":
                continue
            level, _, role, _ = n.parts
            key = "haze" if role == "hazy" else "clean"
            if role.lower() not in ("clear", "gt", "hazy") or key in matched[level, n.stem]:
                raise ValueError("Malformed original dataset test branch")
            matched[level, n.stem][key] = name
        cohort, photos = [], []
        for (level, number), status in sorted(matched.items()):
            if set(status) != {"haze", "clean"}:
                raise ValueError("A released test crop has an unmatched clear or hazy tile")
            fog, clean = (image(z, status[k]) for k in ("haze", "clean"))
            if fog.size != clean.size or fog.size != (256, 256):
                raise ValueError("The official registered crop grid differs")
            cohort.append(dict(dataset="rrshid", id=level + "/" + number, level=level, **status))
            photos.append(clean)
            sizes[level] += 1
        grouped = groups(cohort, photos)
        records.extend(dict(r, group=grouped[r["id"]]) for r in cohort)
    with ZipFile(NT / "hazy.zip") as fog, ZipFile(NT / "clean.zip") as clean:
        ids = {PurePosixPath(p).stem.removesuffix("_NTHazy"): p for p in fog.namelist() if p.endswith("_NTHazy.png")}
        refs = {PurePosixPath(p).stem.removesuffix("_GT"): p for p in clean.namelist() if p.endswith("_GT.png")}
        if set(ids) != set(refs) or len(ids) != 25:
            raise ValueError("Unexpected paired images in the public NT-HAZE training archives")
        cohort, photos = [], []
        for key in sorted(ids):
            frame = image(clean, refs[key])
            if frame.size != image(fog, ids[key]).size:
                raise ValueError("The published NT-HAZE reference grids differ")
            cohort.append(dict(dataset="nt_haze", id=key, level="night", haze=ids[key], clean=refs[key], size=list(frame.size)))
            photos.append(long_edge(frame, 256))
        grouped = groups(cohort, photos)
        records.extend(dict(r, group=grouped[r["id"]]) for r in cohort)
    if Counter(r["dataset"] for r in records) != COUNTS or sizes != dict(thin_fog=76, moderate_fog=152, thick_fog=76):
        raise ValueError("Unexpected official cohorts")
    card = dict(archives=[rr, nt], counts=COUNTS, rrshid_conditions=dict(sizes), records=records,
        groups={d: len({r["group"] for r in records if r["dataset"] == d}) for d in COUNTS},
        note="RRSHID original Test, density strata from the release. NT-HAZE published training pairs, evaluation only here, resized jointly to longest side at most 1024. Challenge validation and test references are not publicly supplied in the used archives. Cluster visually near-duplicate in-domain clear crops before bootstrapping. IDs do not establish independent geographic locations.",
        rrshid_paper="https://arxiv.org/abs/2503.17966", code_SHA256=digest(__file__), finished_utc=now())
    CARD.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(CARD, card)
    print(dict(counts=COUNTS, groups=card["groups"], conditions=dict(sizes)), flush=True)
    return card


def pairs():
    data = stage()
    with ZipFile(RR / "RRSHID.zip") as rr, ZipFile(NT / "hazy.zip") as fog, ZipFile(NT / "clean.zip") as clean:
        for row in data["records"]:
            night = row["dataset"] == "nt_haze"
            a, b = image(fog if night else rr, row["haze"]), image(clean if night else rr, row["clean"])
            if night:
                length = min(1024, max(a.size))
                a, b = long_edge(a, length), long_edge(b, length)
            yield row, image_tensor(a), image_tensor(b)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.parse_args()
    stage()

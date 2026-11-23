"""Reserve the complete previously unused recording-disjoint MUSES snow validation subset."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
import torch

from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_fog_correction_v70 import verified_image
from scripts.stage_muses_rain_v90 import DATA, FOG_MANIFESTS, MANIFEST as RAIN
from scripts.stage_muses_v67 import ARCHIVES, SOURCE, archive_index, stage_item
from scripts.train_regional_guidance_v18 import atomic_write


ROOT = DATA / "muses-snow-v92"
INDEX, MANIFEST = ROOT / "registration.json", ROOT / "held.json"


def prior():
    files = (*FOG_MANIFESTS, RAIN)
    rows = [r for f in files for r in read(f)["pairs"]]
    return ({str(f): digest(f) for f in files}, {r["group"] for r in rows},
            {r["id"] for r in rows}, {r["frame_camera"]["SHA256"] for r in rows})


def snow_index(kind):
    return archive_index(kind, ("val",), ("snow",))


def stage(job):
    with requests.Session() as session:
        return stage_item(job, directory=ROOT / "frames", session=session)


def prepare():
    old, groups, ids, hashes = prior()
    if INDEX.exists():
        card = read(INDEX)
        if card["source_SHA256"] != digest(__file__) or old != card["prior"]:
            raise ValueError("Snow selection changed")
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sources = list(pool.map(snow_index, ARCHIVES))
        archives = {kind: site for kind, site, _ in sources}
        records = {kind: rows for kind, _, rows in sources}
        keys = set(records["frame_camera"])
        if keys != set(records["gt_semantic"]):
            raise ValueError("Unpaired public validation records")
        selected = {k: {kind: rows[k] for kind, rows in records.items()} for k in sorted(keys)
                    if k not in ids and records["frame_camera"][k]["group"] not in groups}
        if not selected:
            raise ValueError("No new snow recordings")
        card = dict(source=SOURCE, source_SHA256=digest(__file__), prior=old, archives=archives, cases=selected,
                    public_snow=len(keys), selection="All official snow validation frames disjoint by recording "
                    "and ID from all registered fog and rain cases, before any inspection of labels or outputs.", utc=now())
        atomic_write(INDEX, card)
    jobs = [(k, records, card["archives"]) for k, records in card["cases"].items()]
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(stage, jobs))
    fresh = {r["frame_camera"]["SHA256"] for r in rows}
    if fresh & hashes or len(fresh) != len(rows):
        raise ValueError("Duplicate original image")
    atomic_write(MANIFEST, dict(registration_SHA256=digest(INDEX), pairs=rows, counts={"muses_snow": len(rows)},
                 groups=dict(Counter(r["group"] for r in rows)), utc=now()))
    print(dict(snow=len(rows), groups=len({r["group"] for r in rows})), flush=True)


def cases():
    card = read(MANIFEST)
    if card["registration_SHA256"] != digest(INDEX):
        raise ValueError("Snow manifest differs from its registration")
    for row in card["pairs"]:
        im, gt = row["frame_camera"], row["gt_semantic"]
        rgb = verified_image(im["file"], im["SHA256"], False)
        truth = verified_image(gt["file"], gt["SHA256"], True)
        yield "muses_snow", row["id"], row["group"], row["daytime"], rgb, torch.from_numpy(truth.copy()).cuda().long()


if __name__ == "__main__":
    prepare()

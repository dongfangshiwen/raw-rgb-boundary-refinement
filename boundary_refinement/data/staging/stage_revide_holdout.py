"""Stage previously unused author REVIDE train videos for external evaluation."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import PurePosixPath

from PIL import Image
import requests
import torch

from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, long_edge
from boundary_refinement.data.real_haze_data import overlap_check
from boundary_refinement.data.staging.stage_fusion_holdouts import witness
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD, reference
from boundary_refinement.data.staging.stage_revide import AUTHOR, CARD, PAPER, decode, get_rgb, zip_directory
from boundary_refinement.training.train_regional_guidance import atomic_write, torch_save


ROOT = REPO / "results/cross-scale-rgb-v44"
FILE = REPO.parent / "benchmark_data/post-fusion-holdout-v30/revide_external_v44.pth"
MANIFEST = ROOT / "revide_external.json"
GROUPS = 16


def capture(name):
    return name.split("_", 1)[0]


def order(name):
    return sha256(("revide-external-v44/" + name).encode()).digest()


def plan(entries):
    old = {capture(x) for x in read(CARD)["groups"]}
    pairs = defaultdict(lambda: defaultdict(dict))
    for row in entries:
        path = PurePosixPath(row["name"])
        if len(path.parts) == 4 and path.parts[0] == "Train" and path.parts[1] in ("hazy", "gt"):
            video, frame = path.parts[2:]
            if capture(video) not in old and path.suffix.lower() == ".jpg":
                pairs[video][frame][path.parts[1]] = row
    groups = defaultdict(list)
    for video, frames in pairs.items():
        matched = sorted(key for key, views in frames.items() if set(views) == {"hazy", "gt"})
        if len(matched) >= 8:
            groups[capture(video)].append((video, frames, matched))
    chosen = []
    selected = sorted(groups, key=order)[:GROUPS]
    if len(selected) != GROUPS:
        raise ValueError("Unexpected size of previously available capture codes")
    for group in selected:
        video, frames, matched = min(groups[group], key=lambda values: order(values[0]))
        for fraction in (.32, .71):
            key = matched[round(fraction * (len(matched) - 1))]
            chosen.append((video, key, frames[key]))
    return chosen


def references():
    record = read(CARD)
    if record["SHA256"] != digest(record["file"]):
        raise ValueError("Prior REVIDE pixels changed")
    for row in torch.load(record["file"], map_location="cpu", weights_only=True):
        rgb = Image.fromarray(row["clean"].permute(1, 2, 0).numpy())
        yield witness("REVIDE/" + row["id"], rgb, row["id"])
    for row in read(TRAIN_CARD)["scenes"]:
        if row["role"] == "fit":
            yield reference(row)


def fetch(item):
    video, frame, entries = item
    with requests.Session() as session:
        fog, clear = (get_rgb(session, entries[key]) for key in ("hazy", "gt"))
    x, y = map(decode, (fog, clear))
    if x.size != y.size:
        raise ValueError("REVIDE pair dropped geometric alignment")
    key = video + "/" + frame
    side = min(1024, max(x.size))
    record = dict(id=key, group=capture(video), hazy=image_tensor(long_edge(x, side)),
                  clean=image_tensor(long_edge(y, side)))
    return dict(data=record, reference=witness(key, y, sha256(clear).hexdigest()), known=entries)


def stage():
    ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        old = read(MANIFEST)
        if old["SHA256"] != digest(FILE):
            raise ValueError("REVIDE additional external pairs changed")
        return old
    index, document = zip_directory()
    chosen = plan(document["members"])
    packets, errors = [], {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        work = {pool.submit(fetch, row): row for row in chosen}
        for future in as_completed(work):
            try:
                packets.append(future.result())
            except (requests.RequestException, OSError) as error:
                video, frame, _ = work[future]
                errors[video + "/" + frame] = type(error).__name__
            if (len(packets) + len(errors)) % 8 == 0:
                print(dict(revide_archive_pairs=len(packets), errors=len(errors)), flush=True)
    flags, nearest = overlap_check(list(references()), [value["reference"] for value in packets])
    blocked = {capture(x["id"].split("/")[0]) for x in flags}
    blocked.update(capture(x.split("/")[0]) for x in errors)
    data = [x["data"] for x in sorted(packets, key=lambda r: r["data"]["id"])
            if x["data"]["group"] not in blocked]
    if len(data) < 20:
        raise ValueError("The additional real paired archive could not yield enough complete capture groups")
    torch_save(FILE, data)
    result = dict(file=str(FILE), SHA256=digest(FILE), member_index_SHA256=digest(index),
                  source=AUTHOR, author_paper=PAPER, count=len(data), groups=sorted({r["group"] for r in data}),
                  planned=[group + "/" + k for group, k, _ in chosen], blocked=sorted(blocked),
                  download_failures=errors, checked=flags, nearest=nearest,
                  scope="Previously unused videos from the official REVIDE archive's Train folder are evaluation-only in this project. Select one video per base capture code, two ordinal quantiles, disjoint from the earlier six Test video codes. Source training exposure for external author weights is unknown.",
                  finished_utc=now())
    atomic_write(MANIFEST, result)
    print(dict(new_real_pairs=len(data), capture_codes=len(result["groups"]), blocked=sorted(blocked)), flush=True)
    return result


if __name__ == "__main__":
    stage()

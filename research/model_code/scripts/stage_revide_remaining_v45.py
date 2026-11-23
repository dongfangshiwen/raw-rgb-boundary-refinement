"""Freeze the remaining REVIDE capture codes for a new method check."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from PIL import Image
import torch

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_haze_data_v22 import overlap_check
from scripts.stage_fusion_holdouts_v29 import witness
from scripts.stage_revide_holdout_v44 import FILE as PREVIOUS, MANIFEST as FIRST, capture, fetch, order, references
from scripts.stage_revide_v30 import AUTHOR, CARD, PAPER, zip_directory
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


ROOT = REPO / "results/veil-response-v45"
FILE = REPO.parent / "benchmark_data/post-fusion-holdout-v30/revide_final_v45.pth"
MANIFEST = ROOT / "revide_final.json"


def stage():
    ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        known = read(MANIFEST)
        if known["SHA256"] != digest(FILE):
            raise ValueError("Final REVIDE RGB changed")
        return known
    prior = read(FIRST)
    old = {capture(g) for g in read(CARD)["groups"]} | set(prior["groups"])
    if prior["SHA256"] != digest(PREVIOUS):
        raise ValueError("Earlier independent cohort changed")
    index, parsed = zip_directory()
    videos = defaultdict(lambda: defaultdict(dict))
    for item in parsed["members"]:
        path = PurePosixPath(item["name"])
        if len(path.parts) == 4 and path.parts[0] == "Train" and path.suffix.lower() == ".jpg":
            _, role, video, frame = path.parts
            if role in ("gt", "hazy") and capture(video) not in old:
                videos[video][frame][role] = item
    population = defaultdict(list)
    for video, frames in videos.items():
        matching = sorted(k for k, views in frames.items() if set(views) == {"hazy", "gt"})
        if len(matching) >= 8:
            population[capture(video)].append((video, frames, matching))
    planned = []
    for group, choices in sorted(population.items(), key=lambda entry: order(entry[0])):
        video, frames, keys = min(choices, key=lambda value: order(value[0]))
        for fraction in (.32, .71):
            frame = keys[round(fraction * (len(keys) - 1))]
            planned.append((video, frame, frames[frame]))
    if len(planned) != 16:
        raise ValueError("Unexpected number of unused REVIDE capture codes")
    with ThreadPoolExecutor(max_workers=8) as pool:
        retrieved = list(pool.map(fetch, planned))
    earlier = list(references())
    for sample in torch.load(PREVIOUS, map_location="cpu", weights_only=True):
        clear = Image.fromarray(sample["clean"].permute(1, 2, 0).numpy())
        earlier.append(witness("REVIDE/V44/" + sample["id"], clear, sample["id"]))
    flags, nearest = overlap_check(earlier, [r["reference"] for r in retrieved])
    excluded = {capture(r["id"].split("/")[0]) for r in flags}
    values = [r["data"] for r in retrieved if r["data"]["group"] not in excluded]
    if len(values) < 12:
        raise ValueError("Near duplicates left too few independent capture codes")
    torch_save(FILE, values)
    receipt = dict(file=str(FILE), SHA256=digest(FILE), author=AUTHOR, paper=PAPER,
        groups=sorted({r["group"] for r in values}), count=len(values), index_SHA256=digest(index),
        prior_SHA256=digest(FIRST), excluded=flags, nearest=nearest, finished_utc=now(),
        scope="Remaining base capture codes in the author Train archive, excluding those evaluated earlier in the project. This project uses them for inference only. The official Test videos are separate.")
    atomic_write(MANIFEST, receipt)
    print(dict(captures=len(receipt["groups"]), frames=len(values)), flush=True)
    return receipt


def pairs():
    card = read(MANIFEST)
    if card["SHA256"] != digest(FILE):
        raise ValueError("The final real paired archive changed")
    for item in torch.load(FILE, map_location="cpu", weights_only=True):
        yield "revide_final", item["id"], item["group"], item["hazy"], item["clean"]


if __name__ == "__main__":
    stage()

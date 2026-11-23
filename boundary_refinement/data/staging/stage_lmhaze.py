"""Stage evenly sampled LMHaze frames, with original reference scenes kept together."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from hashlib import sha256
from pathlib import PurePosixPath
from zipfile import ZipFile
import zlib

from PIL import Image
import requests
import torch

from boundary_refinement.data.staging.prepare_rw2ah import LOCAL_HEADER, decode
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, long_edge
from boundary_refinement.data.real_haze_data import overlap_check
from boundary_refinement.data.staging.stage_fusion_holdouts import witness
from boundary_refinement.data.staging.stage_postfusion_holdout import observed_refs
from boundary_refinement.data.staging.stage_revide import DriveRanges, range_bytes
from boundary_refinement.training.train_regional_guidance import atomic_write, torch_save


ROOT = REPO / "results/scene-utility-v31"
STORE = REPO.parent / "benchmark_data/lmhaze-v31"
CARD = ROOT / "lmhaze.json"
TRAIN_CARD = ROOT / "lmhaze_training.json"
PAPER = "https://arxiv.org/abs/2410.16095"
AUTHOR = "https://github.com/wangzrk/LMHaze"
ARCHIVES = {"train": ("10BiQF9oTexwo3EdRKeyqhY39Q_qAAAc2", 9389769110, 3925, 40),
            "test": ("1V-PwHbLfNgg1nF64bPK9HHUlc93bDSGs", 2567756202, 1115, 14)}
FRAMES = {"train": 8, "test": 10}


def fetch_index(split):
    key, size, _, _ = ARCHIVES[split]
    url = f"https://drive.usercontent.google.com/download?id={key}&export=download&confirm=t"
    path = STORE / (split+"_index.json")
    if path.exists():
        dataset = read(path)
        if dataset["url"] != url or dataset["bytes"] != size:
            raise ValueError("Author LMHaze archive changed")
        return path, dataset
    with requests.Session() as session:
        remote = DriveRanges(session, url, size)
        with ZipFile(remote) as source:
            members = [dict(name=m.filename, compressed=m.compress_size, bytes=m.file_size,
                            CRC32=f"{m.CRC:08x}", offset=m.header_offset, method=m.compress_type)
                       for m in source.infolist() if not m.is_dir() and "__MACOSX" not in m.filename]
        dataset = dict(url=url, bytes=size, members=members, received=remote.received)
    atomic_write(path, dataset)
    return path, dataset


def scenes(split, index):
    groups, fog = defaultdict(list), {}
    for row in index["members"]:
        path = PurePosixPath(row["name"])
        if len(path.parts) != 3 or path.parts[0] != split or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected LMHaze archive entry {path}")
        if path.parts[1] == "GT":
            groups[f"{row['CRC32']}-{row['bytes']}"].append(row["name"])
        elif path.parts[1] == "hazy":
            fog[path.name] = row["name"]
        else:
            raise ValueError(f"Unexpected LMHaze image role {path}")
    _, _, n, g = ARCHIVES[split]
    if len(groups) != g or len(fog) != n or sum(map(len, groups.values())) != n:
        raise ValueError("Author LMHaze population differs")
    output = {}
    for key, members in sorted(groups.items()):
        ordered = sorted(members)
        count = FRAMES[split]
        selected = [ordered[(2*i+1)*len(ordered)//(2*count)] for i in range(count)]
        if len(set(selected)) != count:
            raise ValueError("LMHaze reference has insufficient distinct frames")
        output[key] = (selected[0], [fog[PurePosixPath(name).name] for name in selected])
    return output


def fetch_member(session, index, entry):
    start = entry["offset"]
    end = min(index["bytes"]-1, start+entry["compressed"]+LOCAL_HEADER.size+2*65535-1)
    for attempt in range(4):
        try:
            record = range_bytes(session, index["url"], index["bytes"], start, end)
            header = LOCAL_HEADER.unpack_from(record)
            if header[0] != 0x04034b50 or header[2]&1 or header[3] != entry["method"]:
                raise ValueError("Invalid LMHaze local image header")
            position = LOCAL_HEADER.size+header[-2]+header[-1]
            data = record[position:position+entry["compressed"]]
            if entry["method"] == 8:
                data = zlib.decompress(data,-15)
            elif entry["method"] != 0:
                raise ValueError("Unsupported LMHaze image compression")
            if len(data) != entry["bytes"] or f"{zlib.crc32(data):08x}" != entry["CRC32"]:
                raise OSError("LMHaze image CRC32 mismatch")
            return data
        except (requests.RequestException, OSError) as exc:
            if attempt == 3:
                raise
            print("Retrying LMHaze image", entry["name"], type(exc).__name__, flush=True)


def fetch_scene(key, pair, index, directory):
    ref_name, selected = pair
    path = STORE / f"scene-{key}.pth"
    provenance = STORE / f"scene-{key}.json"
    names = [ref_name, *selected]
    if provenance.exists():
        card = read(provenance)
        if digest(path) != card["SHA256"] or card["members"] != names:
            raise ValueError("Registered LMHaze scene changed")
        return card
    with requests.Session() as session:
        images = [fetch_member(session, index, directory[name]) for name in names]
    clean = decode(images[0])
    if f"{len(images[0])}" != key.split("-",1)[1]:
        raise ValueError("LMHaze clear image length differs")
    foggy = [decode(raw) for raw in images[1:]]
    if any(image.size != clean.size for image in foggy):
        raise ValueError("LMHaze pair has differing RGB dimensions")
    pixels = [image_tensor(long_edge(im, min(1024, max(clean.size)))) for im in (clean, *foggy)]
    torch_save(path, {"clean": pixels[0], "frames": pixels[1:]})
    card = dict(file=str(path), SHA256=digest(path), members=names,
                bytes_SHA256=[sha256(v).hexdigest() for v in images], scene=key, native_size=list(clean.size))
    atomic_write(provenance, card)
    return card


def reference(card):
    pixels = torch.load(card["file"], map_location="cpu", weights_only=True)["clean"]
    return witness(card["scene"], Image.fromarray(pixels.permute(1,2,0).numpy()), card["bytes_SHA256"][0])


def calibration_groups(train):
    order = sorted(train, key=lambda s: sha256(("scene-utility-v31/"+s).encode()).digest())
    return set(order[:8])


def publish_training():
    """Release the checked training split as soon as its local cache is complete."""
    path, index = fetch_index("train")
    registered = scenes("train", index)
    calibration = calibration_groups(registered)
    items = [(k, STORE/f"scene-{k}.json") for k in registered]
    if not all(p.is_file() for _,p in items):
        raise FileNotFoundError("LMHaze training scenes are still staging")
    cards = [dict(read(p), role="calibrate" if k in calibration else "fit") for k,p in items]
    refs = {r["scene"]:reference(r) for r in cards}
    fit = [refs[r["scene"]] for r in cards if r["role"] == "fit"]
    cal = [refs[r["scene"]] for r in cards if r["role"] == "calibrate"]
    flags, nearest = overlap_check(fit+list(observed_refs()),cal)
    removed = {r["id"] for r in flags}
    valid = [r for r in cards if r["scene"] not in removed]
    result = dict(author=AUTHOR, paper=PAPER,index_SHA256=digest(path),scene_counts=dict(Counter(r["role"] for r in valid)),
                  scenes=valid, excluded=flags, nearest=nearest, finished_utc=now())
    if not TRAIN_CARD.exists():
        atomic_write(TRAIN_CARD,result)
    print(dict(ready_training_scenes=result["scene_counts"], excluded=sorted(removed)),flush=True)


def stage():
    ROOT.mkdir(exist_ok=True, parents=True)
    STORE.mkdir(exist_ok=True, parents=True)
    if CARD.exists():
        return read(CARD)
    sources = {split: fetch_index(split) for split in ARCHIVES}
    groups = {split: scenes(split, source) for split, (_, source) in sources.items()}
    members = {split: {r["name"]:r for r in source["members"]} for split, (_, source) in sources.items()}
    if groups["train"].keys() & groups["test"].keys():
        raise ValueError("Author train/test contain the same exact reference")
    cal_groups = calibration_groups(groups["train"])
    cards = {}
    with ThreadPoolExecutor(max_workers=14) as pool:
        tasks = {pool.submit(fetch_scene, key, pair, sources[split][1], members[split]): split
                 for split, subset in groups.items() for key, pair in subset.items()}
        for task in as_completed(tasks):
            row = task.result()
            group = row["scene"]
            split = tasks[task]
            row["role"] = "calibrate" if group in cal_groups else "fit" if split == "train" else "test"
            cards[group] = row
            if len(cards) % 6 == 0:
                print(dict(ready_lmhaze_scenes=len(cards), total=len(tasks)), flush=True)
    refs = {key: reference(card) for key, card in cards.items()}
    older = list(observed_refs())
    drop, nearest = {}, {}
    fit_refs = [refs[k] for k, v in cards.items() if v["role"] == "fit"]
    for role in ("calibrate", "test"):
        local = [refs[k] for k, v in cards.items() if v["role"] == role]
        controls = fit_refs if role == "calibrate" else [refs[k] for k, v in cards.items() if v["role"] != "test"]
        flags, found = overlap_check(controls + older, local)
        nearest[role] = found
        for result in flags:
            drop.setdefault(result["id"], []).append(result)
    valid = [cards[k] for k in sorted(cards.keys()-drop.keys())]
    if {v["role"] for v in valid} != {"fit", "calibrate", "test"}:
        raise ValueError("LMHaze grouped split became empty")
    result = dict(author=AUTHOR, paper=PAPER, frames_per_scene=FRAMES,
                  indices={k: dict(path=str(v[0]), SHA256=digest(v[0]), url=v[1]["url"]) for k, v in sources.items()},
                  selection="Group author PNG clear targets by matching ZIP length and CRC32. Sample center of each equal index bin. Reserve eight hashed official training scenes for calibration and all official test scenes for evaluation. Remove flagged near-duplicates from calibration or test.",
                  scene_counts=dict(Counter(v["role"] for v in valid)), exclusions=drop, nearest=nearest,
                  scenes=valid, finished_utc=now())
    atomic_write(CARD, result)
    print(dict(scene_counts=result["scene_counts"], excluded=list(drop)), flush=True)
    return result


def samples(roles):
    manifest = TRAIN_CARD if "test" not in roles else CARD
    for card in read(manifest)["scenes"]:
        if card["role"] not in roles:
            continue
        if digest(card["file"]) != card["SHA256"]:
            raise ValueError("LMHaze registered scene bytes changed")
        saved = torch.load(card["file"], map_location="cpu", weights_only=True)
        for name, pixels in zip(card["members"][1:], saved["frames"], strict=True):
            yield "lmhaze", PurePosixPath(name).stem, "LMHaze/"+card["scene"], pixels, saved["clean"], None, card["role"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-training",action="store_true")
    if parser.parse_args().register_training:
        publish_training()
    else:
        stage()

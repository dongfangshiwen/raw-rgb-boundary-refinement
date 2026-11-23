"""Reserve recording-disjoint rain frames from the official MUSES validation archives."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import requests
import torch

from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import verified_image
from scripts.stage_muses_v67 import ARCHIVES, SOURCE, archive_index, stage_item
from scripts.train_regional_guidance_v18 import atomic_write


DATA = REPO.parent / "benchmark_data"
ROOT = DATA / "muses-rain-v90"
INDEX = ROOT / "registration.json"
MANIFEST = ROOT / "held.json"
LIMIT = 56
FOG_MANIFESTS = tuple(DATA / name / f"{split}.json" for name, splits in (
    ("muses-v70", ("fit", "calibrate", "held")),
    ("muses-recording-v74", ("fit", "calibrate", "held")),
    ("muses-affinity-v75", ("held",)),
) for split in splits)


def prior():
    groups, rgb, identities = set(), set(), set()
    for file in FOG_MANIFESTS:
        for row in read(file)["pairs"]:
            groups.add(row["group"])
            rgb.add(row["frame_camera"]["SHA256"])
            identities.add(row["id"])
    return groups, rgb, identities


def order(value):
    return sha256(f"rain-v90/{value}".encode()).digest()


def rain_index(key):
    return archive_index(key, ("val",), ("rain",))


def register():
    signatures = {str(f): digest(f) for f in FOG_MANIFESTS}
    if INDEX.is_file():
        result = read(INDEX)
        if result["prior"] != signatures or digest(__file__) != result["script_SHA256"]:
            raise ValueError("The reserved rain index belongs to another protocol")
        return result
    prior_groups, _, previous = prior()
    with ThreadPoolExecutor(max_workers=2) as workers:
        source = list(workers.map(rain_index, ARCHIVES))
    archives, catalogs = {}, {}
    for kind, archive, candidates in source:
        archives[kind], catalogs[kind] = archive, candidates
    rgb, gt = catalogs["frame_camera"], catalogs["gt_semantic"]
    if gt.keys() != rgb.keys():
        raise ValueError("MUSES rain label and camera identifiers do not agree")
    by_group = defaultdict(list)
    for key, row in rgb.items():
        if row["group"] not in prior_groups and key not in previous:
            by_group[row["group"]].append(key)
    if not by_group:
        raise ValueError("No recording-disjoint MUSES rain labels available")
    groups = sorted(by_group, key=order)
    for values in by_group.values():
        values.sort(key=order)
    sequence = [by_group[group][i] for i in range(max(map(len, by_group.values()))) for group in groups
                if i < len(by_group[group])]
    chosen = {key: {kind: entries[key] for kind, entries in catalogs.items()} for key in sequence[:LIMIT]}
    result = dict(source=SOURCE, archives=archives, prior=signatures, script_SHA256=digest(__file__),
                  full_rain_validation=len(rgb), eligible_recordings=groups, cases=chosen,
                  selection="Fixed SHA order round-robin over official rain validation recordings unseen in the "
                            "project's fog manifests. Frame selection uses no class labels or model outcomes.",
                  utc=now())
    atomic_write(INDEX, result)
    print(dict(rain_candidates=len(rgb), eligible_recordings=groups, registered=len(chosen)), flush=True)
    return result


def stage_case(job):
    with requests.Session() as session:
        return stage_item(job, directory=ROOT / "frames", session=session)


def stage():
    source = register()
    jobs = [(key, files, source["archives"]) for key, files in source["cases"].items()]
    with ThreadPoolExecutor(max_workers=5) as workers:
        pairs = list(workers.map(stage_case, jobs))
    groups, old_rgb, used = prior()
    checksums, accepted = set(), []
    for row in pairs:
        checksum = row["frame_camera"]["SHA256"]
        if row["group"] in groups or row["id"] in used or checksum in old_rgb or checksum in checksums:
            raise ValueError(f"Unexpected overlap in the reserved rain frames: {row['id']}")
        checksums.add(checksum)
        accepted.append(row)
    coverage = Counter(cls for row in accepted for cls in row["observed_label_ids"] if cls < 19)
    lights = Counter(row["group"] for row in accepted if 6 in row["observed_label_ids"])
    atomic_write(MANIFEST, dict(registration_SHA256=digest(INDEX), pairs=accepted, class_image_counts=coverage,
                 class_six_recordings=lights, counts={"muses_rain": len(accepted)}, utc=now()))
    print(dict(registered=len(accepted), traffic_light_images=coverage[6], traffic_light_recordings=len(lights)), flush=True)


def cases():
    card = read(MANIFEST)
    if card["registration_SHA256"] != digest(INDEX):
        raise ValueError("Rain evaluation population differs from registration")
    for row in card["pairs"]:
        image, mask = row["frame_camera"], row["gt_semantic"]
        rgb = verified_image(image["file"], image["SHA256"], False)
        labels = verified_image(mask["file"], mask["SHA256"], True)
        yield "muses_rain", row["id"], row["group"], row["daytime"], rgb, torch.from_numpy(labels.copy()).cuda().long()


if __name__ == "__main__":
    stage()

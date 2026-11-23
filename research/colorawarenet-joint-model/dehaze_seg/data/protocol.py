"""Fixed, content-verified train/validation/test manifests for new experiments."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image

from .datasets import make_dataset
from .splits import canonical_root, clear_reference, file_digest, paired_directories
from ..utils import write_json


PARTITIONS = ("train", "val", "test")


def manifest_digest(manifest):
    """Exclude local paths, retaining sample membership and all input hashes."""
    payload = {key: manifest[key] for key in ("dataset", "records", *PARTITIONS)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_record(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        shape = rgb.size
        pixels = hashlib.sha256(str(shape).encode("ascii") + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        difference = small[:, 1:] > small[:, :-1]
        dhash = int.from_bytes(np.packbits(difference).tobytes(), "big")
    return pixels, shape, dhash


def audit_samples(dataset, root, ids):
    hazy_dir, clear_dir, mask_dir = paired_directories(dataset, root)
    from .common import find_by_stem
    records, image_cache, hashes = {}, {}, {}
    for stem in sorted(ids):
        clear = clear_reference(clear_dir, stem, dataset)
        hazy = find_by_stem(hazy_dir, stem)
        mask = find_by_stem(mask_dir, stem) if mask_dir is not None else None
        if clear is None or hazy is None or (mask_dir is not None and mask is None):
            raise ValueError(f"Missing paired files for {stem}")
        if clear not in image_cache:
            image_cache[clear] = image_record(clear)
        pixels, size, dhash = image_cache[clear]
        for path in (clear, hazy, mask):
            if path is None:
                continue
            if path not in hashes:
                hashes[path] = file_digest(path)
            with Image.open(path) as image:
                if image.size != size:
                    raise ValueError(f"Unaligned sizes for {stem}: {path.name} is {image.size}, clear is {size}")
        records[stem] = dict(reference=clear.name, clear_sha256=hashes[clear],
                             clear_pixels_sha256=pixels, hazy_sha256=hashes[hazy],
                             mask_sha256=hashes[mask] if mask else None,
                             width=size[0], height=size[1], dhash=f"{dhash:016x}")
    return records


def scene_groups(records, declared=None):
    """Union identical decoded references and explicitly declared scene IDs."""
    ids = sorted(records)
    if declared is not None and (set(declared) != set(ids) or
                                 any(not isinstance(v, str) or not v.strip() for v in declared.values())):
        raise ValueError("Scene-group JSON must map every sample ID to a nonempty scene string")
    parent = {stem: stem for stem in ids}

    def find(stem):
        while stem != parent[stem]:
            parent[stem] = parent[parent[stem]]
            stem = parent[stem]
        return stem

    def merge(first, second):
        a, b = sorted((find(first), find(second)))
        parent[b] = a

    pixels_seen, scenes_seen = {}, {}
    for stem in ids:
        pixels = records[stem]["clear_pixels_sha256"]
        if pixels in pixels_seen:
            merge(stem, pixels_seen[pixels])
        pixels_seen[pixels] = stem
        if declared is not None:
            scene = declared[stem]
            if scene in scenes_seen:
                merge(stem, scenes_seen[scene])
            scenes_seen[scene] = stem
    groups = defaultdict(list)
    for stem in ids:
        groups[find(stem)].append(stem)
    return dict(groups)


def three_way_split(groups, val_ratio=.15, test_ratio=.15, seed=42):
    if (not all(math.isfinite(v) and 0 < v < 1 for v in (val_ratio, test_ratio))
            or val_ratio + test_ratio >= 1):
        raise ValueError("Validation/test ratios must be finite, positive and sum to less than one")
    if len(groups) < 3:
        raise ValueError("A three-way split requires at least three independent scene groups")
    remaining = sorted(groups)
    random.Random(seed).shuffle(remaining)
    count = sum(len(v) for v in groups.values())
    result = {}
    for name, ratio, reserve in (("test", test_ratio, 2), ("val", val_ratio, 1)):
        target = max(1, round(count * ratio))
        chosen, used = [], 0
        for group in remaining:
            if len(remaining) - len(chosen) <= reserve:
                break
            size = len(groups[group])
            if abs(used + size - target) < abs(used - target):
                chosen.append(group)
                used += size
        if not chosen:
            chosen = [min(remaining, key=lambda group: (len(groups[group]), group))]
        result[name] = sorted(stem for group in chosen for stem in groups[group])
        remaining = [group for group in remaining if group not in chosen]
    result["train"] = sorted(stem for group in remaining for stem in groups[group])
    return result


def verify_manifest(manifest, dataset, root, ids):
    """Fail on altered data, partition membership, or overlapping scene groups."""
    if manifest.get("version") != 3 or manifest.get("dataset") != dataset:
        raise ValueError("Fixed training splits require a version-3 manifest for the selected dataset")
    membership = []
    for name in PARTITIONS:
        values = manifest.get(name)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
            raise ValueError(f"Manifest {name} must be a nonempty list of sample IDs")
        membership.extend(values)
    if len(membership) != len(set(membership)) or set(membership) != set(ids):
        raise ValueError("Manifest partitions must be disjoint and cover exactly the available sample IDs")
    records = manifest.get("records", {})
    if any(set(records.get(name, {})) != set(manifest[name]) for name in PARTITIONS):
        raise ValueError("Manifest records do not match partition IDs")
    if manifest.get("split_sha256") != manifest_digest(manifest):
        raise ValueError("Manifest fingerprint mismatch; regenerate the split instead of editing its contents")
    seen_pixels, seen_groups = set(), set()
    hazy_dir, clear_dir, mask_dir = paired_directories(dataset, root)
    from .common import find_by_stem
    for name in PARTITIONS:
        pixels, groups = set(), set()
        for stem in manifest[name]:
            saved = records[name][stem]
            reference = clear_reference(clear_dir, stem, dataset)
            for field, path in (("clear_sha256", reference), ("hazy_sha256", find_by_stem(hazy_dir, stem)),
                                ("mask_sha256", find_by_stem(mask_dir, stem) if mask_dir else None)):
                if path is None and field == "mask_sha256" and mask_dir is None:
                    continue
                if path is None or file_digest(path) != saved.get(field):
                    raise ValueError(f"Dataset content changed for {stem} ({field}); create a new audited manifest")
            pixels.add(saved["clear_pixels_sha256"])
            groups.add(saved["scene_group"])
        if pixels & seen_pixels or groups & seen_groups:
            raise ValueError("Manifest scene groups or clear references cross partitions")
        seen_pixels.update(pixels)
        seen_groups.update(groups)
    result = deepcopy(manifest)
    for name in PARTITIONS:
        result[name + "_root"] = canonical_root(root)
    return result


def load_manifest(path, dataset, root, ids):
    return verify_manifest(json.loads(Path(path).read_text(encoding="utf-8")), dataset, root, ids)


def prepare_split(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Split output already exists: {output}; choose a new --output")
    dataset = make_dataset(args.dataset, args.data_root, resize=None)
    records = audit_samples(args.dataset, args.data_root, dataset.ids)
    declared = json.loads(Path(args.scene_groups).read_text(encoding="utf-8")) if args.scene_groups else None
    groups = scene_groups(records, declared)
    partitions = three_way_split(groups, args.val_ratio, args.test_ratio, args.seed)
    for group, members in groups.items():
        for stem in members:
            records[stem]["scene_group"] = group
    manifest = dict(version=3, dataset=args.dataset, seed=args.seed, val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio, grouping="clear-rgb-sha256+declared-scene" if declared else "clear-rgb-sha256",
                    scene_review="provided-group-IDs; source-location provenance unverified" if declared else "near-duplicate-review-pending",
                    shared_clear_references=[], **partitions,
                    records={name: {stem: records[stem] for stem in partitions[name]} for name in PARTITIONS})
    for name in PARTITIONS:
        manifest[name + "_root"] = canonical_root(args.data_root)
    manifest["split_sha256"] = manifest_digest(manifest)
    verify_manifest(manifest, args.dataset, args.data_root, dataset.ids)
    duplicate_pixels = defaultdict(list)
    for stem, record in records.items():
        duplicate_pixels[record["clear_pixels_sha256"]].append(stem)
    candidates = []
    ids = sorted(records)
    membership = {stem: name for name in PARTITIONS for stem in partitions[name]}
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            if records[first]["scene_group"] == records[second]["scene_group"]:
                continue
            distance = (int(records[first]["dhash"], 16) ^ int(records[second]["dhash"], 16)).bit_count()
            if distance <= args.near_duplicate_distance:
                candidates.append(dict(first=first, second=second, dhash_distance=distance,
                                       crosses_partition=membership[first] != membership[second]))
    candidates.sort(key=lambda row: (row["dhash_distance"], row["first"], row["second"]))
    # Supply a conservative starting point for review, never silently treat it
    # as verified source-sequence metadata or as the active training partition.
    parent = {stem: stem for stem in ids}

    def find(stem):
        while stem != parent[stem]:
            parent[stem] = parent[parent[stem]]
            stem = parent[stem]
        return stem

    for members in groups.values():
        for stem in members:
            a, b = sorted((find(members[0]), find(stem)))
            parent[b] = a
    for pair in candidates:
        a, b = sorted((find(pair["first"]), find(pair["second"])))
        parent[b] = a
    candidate_mapping = {stem: find(stem) for stem in ids}
    report = dict(sample_count=len(ids), scene_group_count=len(groups),
                  counts={name: len(partitions[name]) for name in PARTITIONS},
                  identical_reference_groups=[v for v in duplicate_pixels.values() if len(v) > 1],
                  near_duplicate_candidate_count=len(candidates),
                  cross_partition_candidate_count=sum(row["crosses_partition"] for row in candidates),
                  near_duplicate_candidates=candidates, split_sha256=manifest["split_sha256"],
                  limitation="dHash candidates need visual/source-sequence review; hash isolation alone does not establish unseen locations")
    write_json(output / "split.json", manifest)
    write_json(output / "audit.json", report)
    write_json(output / "scene_groups.json", {stem: records[stem]["scene_group"] for stem in ids})
    write_json(output / "candidate_scene_groups.json", candidate_mapping)
    print(json.dumps({key: report[key] for key in ("sample_count", "scene_group_count", "counts", "identical_reference_groups")}, indent=2))
    return manifest

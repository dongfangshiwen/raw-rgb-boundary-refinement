"""Stage public real RGB pairs and compute a train-only affine haze bank."""
import argparse
import hashlib
import io
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from scripts.camvid_data_v15 import DATA
from scripts.projective_ledger_v12 import REPO, digest, now, read, write
from scripts.train_regional_guidance_v18 import torch_save


ROOT = REPO / "results/real-transfer-v22"
STORE = REPO.parent / "benchmark_data/real-haze-v22"
NH_URL = "https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/"
DENSE_URL = "https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/"
HEIGHT, WIDTH = 192, 256
PHASH_MAX = 6
NCC_MIN = 0.96


def member_index(archive):
    groups = {}
    for info in archive.infolist():
        parts = Path(info.filename).parts
        if info.is_dir() or any(p in ("__MACOSX", "..") for p in parts):
            continue
        match = re.fullmatch(r"(\d{1,3})[_-](GT|hazy)\.(?:png|jpe?g)", parts[-1], re.IGNORECASE)
        if match is None:
            continue
        pair = match.group(1).zfill(2)
        kind = "clean" if match.group(2).lower() == "gt" else "hazy"
        if kind in groups.setdefault(pair, {}):
            raise ValueError("Ambiguous RGB pair ID " + pair)
        groups[pair][kind] = info
    return groups


def read_member(archive, info):
    value = archive.read(info)
    with Image.open(io.BytesIO(value)) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    return im, dict(member=info.filename, CRC32=f"{info.CRC:08x}", SHA256=hashlib.sha256(value).hexdigest(), native=list(im.size))


def align(img):
    w, h = img.size
    factor = min(WIDTH / w, HEIGHT / h)
    x, y = round(w*factor), round(h*factor)
    left, top = (WIDTH-x)//2, (HEIGHT-y)//2
    rgb = np.array(img.resize((x, y), Image.Resampling.BILINEAR))
    result = np.pad(rgb, ((top, HEIGHT-y-top), (left, WIDTH-x-left), (0, 0)), mode="edge")
    rect = [left, top, left+x, top+y]
    return torch.from_numpy(result).permute(2, 0, 1).contiguous(), rect


def fingerprint(img):
    grey = np.array(img.convert("L").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64)
    x = np.arange(32) + .5
    cosine = np.cos(np.pi * np.arange(8)[:, None] * x[None] / 32)
    spectrum = (cosine @ grey @ cosine.T).ravel()[1:]
    bits = spectrum >= np.median(spectrum)
    centered = grey.ravel() - grey.mean()
    centered /= max(np.linalg.norm(centered), 1e-8)
    return bits, centered


def load_archived_pairs(filename, kind, permitted):
    frames, witnesses = [], []
    path = STORE / filename
    with ZipFile(path) as archive:
        index = member_index(archive)
        expected = set(permitted) if permitted is not None else set(index)
        if not expected.issubset(index) or any(set(index[p]) != {"hazy", "clean"} for p in expected):
            raise ValueError("Incomplete RGB references in " + filename)
        if kind == "dense" and len(expected) != 55:
            raise ValueError("The registered Dense-Haze download should contain 55 paired members")
        for pair in sorted(expected):
            hazy, source = read_member(archive, index[pair]["hazy"])
            clean, target = read_member(archive, index[pair]["clean"])
            if hazy.size != clean.size:
                raise ValueError(f"Native pair {kind}/{pair} has different geometry")
            x, roi = align(hazy)
            y, clean_roi = align(clean)
            if roi != clean_roi:
                raise ValueError("Pair coordinates disagree")
            frame = dict(id=pair, hazy=x, clean=y, roi=roi)
            frames.append(frame)
            witnesses.append(dict(id=pair, hazy=source, clean=target, roi=roi, signature=fingerprint(clean),
                                  normalized_clean_SHA256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    return frames, witnesses, dict(path=str(path), SHA256=digest(path), decoded_pairs=len(expected),
                                  remaining_ids=sorted(set(index)-expected))


def old_ohaze():
    points, witnesses = [], []
    rows = [pair for pair in read(DATA)["RGB_pairs"] if pair["dataset"] == "ohaze"]
    if len(rows) != 45:
        raise ValueError("Need inspected O-HAZE45 manifest")
    for row in sorted(rows, key=lambda r: r["id"]):
        images = {}
        for key in ("hazy", "clear"):
            meta = row[key]
            if digest(meta["path"]) != meta["SHA256"]:
                raise ValueError("An O-HAZE source image changed")
            with Image.open(meta["path"]) as im:
                images[key] = ImageOps.exif_transpose(im).convert("RGB")
        if images["hazy"].size != images["clear"].size:
            raise ValueError("O-HAZE pair geometry changed")
        x, roi = align(images["hazy"])
        y, roi2 = align(images["clear"])
        if roi != roi2:
            raise ValueError("O-HAZE pair alignment differs")
        points.append(dict(id=row["id"], hazy=x, clean=y, roi=roi))
        witnesses.append(dict(id=row["id"], hazy=row["hazy"], clean=row["clear"], roi=roi,
                              signature=fingerprint(images["clear"]),
                              normalized_clean_SHA256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    return points, witnesses


def overlap_check(train, other):
    flagged, nearest = [], []
    for row in other:
        candidates = []
        for src in train:
            a, b = row["signature"], src["signature"]
            distance = int(np.count_nonzero(a[0] != b[0]))
            similarity = float(a[1] @ b[1])
            same = (row["normalized_clean_SHA256"] == src["normalized_clean_SHA256"]
                    or row["clean"]["SHA256"] == src["clean"]["SHA256"])
            record = dict(id=row["id"], train_id=src["id"], phash_distance=distance,
                          thumbnail_correlation=similarity, exact_image=same)
            candidates.append(record)
            if same or (distance <= PHASH_MAX and similarity >= NCC_MIN):
                flagged.append(record)
        nearest.append(max(candidates, key=lambda v: v["thumbnail_correlation"]))
    return flagged, nearest


def mask_for(roi):
    left, top, right, bottom = roi
    use = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.bool)
    use[:, top:bottom, left:right] = True
    return use


def reflection_mean(value, size):
    radius = size//2
    return F.avg_pool2d(F.pad(value, (radius, radius, radius, radius), mode="replicate"), size, stride=1)


def affine_haze(hazy, clear):
    """Estimate a shared local slope and bounded RGB intercept in encoded RGB."""
    global_h = hazy.mean((-2, -1), keepdim=True)
    global_c = clear.mean((-2, -1), keepdim=True)
    dc, di = clear-global_c, hazy-global_h
    score = ((dc*di).sum((1, 2, 3), keepdim=True) / dc.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-6)).clamp(.12, 1.)
    h, c = reflection_mean(hazy, 17), reflection_mean(clear, 17)
    variance = (reflection_mean(clear.square(), 17)-c.square()).clamp_min(0).mean(1, keepdim=True)
    cov = (reflection_mean(hazy*clear, 17)-h*c).mean(1, keepdim=True)
    transmission = ((cov + .002*score)/(variance+.002)).clamp(.12, 1.)
    transmission = reflection_mean(transmission, 11).clamp(.12, 1.)
    bias = (h-transmission*c).clamp_min(0).minimum(1-transmission)
    pooled = F.adaptive_avg_pool2d(torch.cat((transmission, bias), dim=1).double(), (12, 16)).float()
    # The inequality is convex. Enforce it again after rounding in the pooled representation.
    low_transmission = pooled[:, :1].clamp(.12, 1)
    low_bias = pooled[:, 1:].clamp_min(0).minimum(1 - low_transmission)
    fields = torch.cat((low_transmission, low_bias), dim=1)
    # Both arms share the same donor.
    spatial = F.interpolate(fields, hazy.shape[-2:], mode="bilinear", align_corners=False)
    homogeneous = spatial.mean((-2, -1), keepdim=True)
    return fields, homogeneous, spatial


def field_bank(training):
    fields, summaries = [], []
    for row in training:
        clean = row["clean"][None].float() / 255
        hazy = row["hazy"][None].float() / 255
        field, homogen, spatial = affine_haze(hazy, clean)
        if not torch.isfinite(field).all() or not bool((field[:, 1:] <= 1-field[:, :1]+1e-6).all()):
            raise ValueError("Invalid fitted donor fields")
        valid = mask_for(row["roi"])
        def error(values):
            rendered = values[:, :1]*clean+values[:, 1:]
            return float((rendered-hazy).abs()[0, :, valid[0]].mean())
        summaries.append(dict(id=row["id"], homogeneous_MAE=error(homogen), spatial_MAE=error(spatial),
                              t_mean=float(field[:, :1].mean()), t_std=float(field[:, :1].std())))
        fields.append(field[0])
    return torch.stack(fields), summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data_manifest.json"
    train_cache, eval_cache = STORE / "training.pth", STORE / "evaluation.pth"
    if any(p.exists() for p in (path, train_cache, eval_cache)):
        raise ValueError("Preserve the registered dataset version")
    train, witnesses, nh = load_archived_pairs("NH-HAZE.zip", "nh", [f"{i:02d}" for i in range(1, 46)])
    dense, dense_rows, dense_archive = load_archived_pairs("Dense-Haze.zip", "dense", None)
    oh, oh_rows = old_ohaze()
    same_dense, near_dense = overlap_check(witnesses, dense_rows)
    same_oh, near_oh = overlap_check(witnesses, oh_rows)
    def ids_of(rows):
        return sorted({r["id"] for r in rows})
    dense = [r for r in dense if r["id"] not in ids_of(same_dense)]
    oh = [r for r in oh if r["id"] not in ids_of(same_oh)]
    if len(dense) < 20 or len(oh) < 20:
        raise ValueError("Too many suspected scene overlaps for this comparison")
    fields, fitted = field_bank(train)
    torch_save(train_cache, dict(training=train, fields=fields, ids=[r["id"] for r in train]))
    torch_save(eval_cache, dict(dense=dense, ohaze=oh))
    def without_array(rows):
        return [{k: v for k, v in row.items() if k!="signature"} for row in rows]
    value = dict(all_passed=True, training_archive=nh, dense_archive=dense_archive, nh_source=NH_URL,
                 dense_source=DENSE_URL, geometry=[WIDTH, HEIGHT], source_manifest_SHA256=digest(DATA),
                 training=dict(file=str(train_cache), SHA256=digest(train_cache), rows=without_array(witnesses)),
                 evaluation=dict(file=str(eval_cache), SHA256=digest(eval_cache), dense=without_array(dense_rows),
                                 ohaze=without_array(oh_rows), dense_eligible=len(dense), ohaze_eligible=len(oh)),
                 duplicate_policy=dict(max_PHash=PHASH_MAX, minimum_correlation=NCC_MIN, dense_hits=same_dense,
                                       ohaze_hits=same_oh, dense_highest_correlations=near_dense, ohaze_highest_correlations=near_oh,
                                       scope="Exact decoded values and conservative thumbnail check only; no independent-site claim"),
                 field_fits=fitted, split_note="NH IDs 01–45 preregistered as the only training donors. Later archive members are left undecoded at staging. The Dense download has 55 paired members, while its descriptive landing page discusses 33 outdoor scenes. We use the registered archive rather than claim an official challenge subset. Dense and O-HAZE are RGB-only evaluation.",
                 updated_utc=now())
    write(path, value)
    print(dict(nh=nh, dense=dense_archive, dense_eligible=len(dense), ohaze_eligible=len(oh),
               spatial_MAE=float(np.mean([r["spatial_MAE"] for r in fitted])),
               homogeneous_MAE=float(np.mean([r["homogeneous_MAE"] for r in fitted]))), flush=True)


def load_data(split):
    note = read(ROOT / "data_manifest.json")
    if split not in ("training", "evaluation") or not note["all_passed"]:
        raise ValueError("Request a registered real data role")
    path = note[split]["file"]
    if digest(path) != note[split]["SHA256"]:
        raise ValueError("Real haze staging artifact changed")
    return torch.load(path, map_location="cpu", weights_only=True)


if __name__ == "__main__":
    main()

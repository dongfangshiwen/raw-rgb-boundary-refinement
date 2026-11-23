"""Train on additional recording-disjoint positive masks and test restoration-aware supervision."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.utils import set_seed
from boundary_refinement.evaluation.controls.audit_dual_output import paired_iou_intervals
from boundary_refinement.models.segmentation.class_protected_residual import class_examples, soft_counts
from boundary_refinement.models.segmentation.coarse_semantics import blend, paint
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor, responses
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.evaluation.semantic.real_fog_correction import aggregate
from boundary_refinement.evaluation.semantic.resolution_interaction import (
    ROOT as PARENT, SEEDS, DetailHead, DualScaleFeatures, Views, add_global_cohort,
    batch, load as raw_models, packets, selection,
)
from boundary_refinement.models.segmentation.restoration_affinity import compress_scores
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.semantic.weatherreasonseg import (
    CLASSES, REVISION, ROOT as HELD, group_recall, open_rgb, parse as held_frames,
)


DATA = REPO.parent / "benchmark_data/weatherreasonseg-v86-train"
ROOT = REPO / "results/sparse-recovery-v86"
ARMS = ("fusion_dense", "raw_mixed", "fusion_mixed", "raw_uncertainty", "recovery_weighted",
        "reverse_weighted", "difference_mixed")
MAIN = "recovery_weighted"
STEPS = 480
TRAIN_SEQUENCES = ("GOPR0477", "GOPR0478", "GOPR0479", "GOPR0480")
PER_SEQUENCE = 27


def interval_sample(values, cap):
    values = sorted(values)
    if len(values) <= cap:
        return set(values)
    return {values[i] for i in np.linspace(0, len(values) - 1, cap, dtype=int)}


def sparse_rows():
    sys.path.insert(0, str(HELD / "deps"))
    import pyarrow.parquet as pq

    release = read(DATA / "release.json")
    if release["revision"] != REVISION or len(release["files"]) != len(TRAIN_SEQUENCES):
        raise ValueError("Unexpected sparse training release")
    population = defaultdict(dict)
    for row in release["files"]:
        path = DATA / Path(row["path"]).name
        sequence = path.stem.removeprefix("fog_")
        if sequence not in TRAIN_SEQUENCES or digest(path) != row["lfs"]["oid"]:
            raise ValueError(f"Unregistered training shard {path.name}")
        data = pq.ParquetFile(path)
        annotation_ids = defaultdict(set)
        for small in data.iter_batches(columns=["image_id", "ann_id"]):
            for record in small.to_pylist():
                key = record["image_id"]
                if not key.startswith(f"{sequence}_frame_") or not record["ann_id"].startswith(f"{key}_"):
                    raise ValueError("Training annotation frame and sequence do not agree")
                k = int(record["ann_id"].removeprefix(f"{key}_"))
                if k not in CLASSES:
                    raise ValueError(f"Unexpected sparse supervision class {k}")
                annotation_ids[key].add(k)
        choose = interval_sample([key for key, kinds in annotation_ids.items() if 6 in kinds], 9)
        choose |= interval_sample([key for key, kinds in annotation_ids.items() if 14 in kinds], 9)
        choose |= interval_sample(set(annotation_ids) - choose, PER_SEQUENCE - len(choose))
        for block in data.iter_batches(batch_size=9):
            for item in block.to_pylist():
                key, ann = item["image_id"], item["ann_id"]
                if key not in choose:
                    continue
                k = int(ann.removeprefix(f"{key}_"))
                shape = item["img_height"], item["img_width"]
                photo = item["image"]
                rgb_hash = sha256(photo).hexdigest()
                if key not in population[sequence]:
                    population[sequence][key] = (photo, rgb_hash, np.full(shape, 255, dtype=np.uint8), Counter())
                _, stored, target, annotation = population[sequence][key]
                if stored != rgb_hash or target.shape != shape or k in annotation:
                    raise ValueError(f"Unexpected sparse frame or duplicate class {ann}")
                with Image.open(BytesIO(item["mask"])) as mask:
                    binary = np.asarray(mask)
                if binary.shape != shape or not set(np.unique(binary)) <= {0, 1, 255}:
                    raise ValueError(f"Invalid training mask {ann}")
                valid = binary != 0
                if not valid.any() or np.any(target[valid] != 255):
                    raise ValueError(f"Overlapping or empty training mask {ann}")
                target[valid] = k
                annotation[k] = int(valid.sum())
        if set(population[sequence]) != choose:
            raise ValueError(f"The sparse training split is incomplete: {sequence}")
    return population


def thumbnail(photo):
    with Image.open(BytesIO(photo)) as image:
        rgb = image.convert("RGB").resize((32, 18), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.float32) / 255.


def register():
    groups = sparse_rows()
    public, _ = held_frames()
    val_hashes = {r[1] for r in public.values()}
    held_video = {key.split("_")[0][-4:] for key in public}
    probe = np.stack([thumbnail(r[0]) for r in public.values()])
    records, blocked, distribution = [], [], Counter()
    for sequence, samples in groups.items():
        if sequence[-4:] in held_video:
            raise ValueError("Sparse training crosses a held video")
        for key, (rgb, rgb_hash, mask, positives) in samples.items():
            thumb = thumbnail(rgb)
            distances = np.abs(probe - thumb).mean(axis=(1, 2, 3))
            if rgb_hash in val_hashes or distances.min() < .012:
                blocked.append(key)
                continue
            records.append(dict(id=key, group=sequence, rgb_SHA256=rgb_hash,
                                mask_SHA256=sha256(mask.tobytes()).hexdigest(), native=dict(positives)))
            distribution.update(positives.keys())
    if set(r["group"] for r in records) != set(TRAIN_SEQUENCES) or not distribution[6] or not distribution[14]:
        raise ValueError("Insufficient disjoint sparse training data")
    card = dict(source_SHA256=digest(DATA / "release.json"), code_SHA256=digest(__file__),
                held_SHA256=digest(HELD / "positive-coverage.json"), rows=sorted(records, key=lambda r: r["id"]),
                recordings=Counter(r["group"] for r in records), positive_frames=dict(distribution), excluded=blocked,
                scope="WeatherReasonSeg real train/fog shards on four different GoPro recording numbers. "
                      "Select up to nine time-spaced frames each for traffic light and truck and fill evenly by time "
                      "to 27 per recording. Remove exact and thumbnail-near duplicates with the separately released val images. "
                      "No annotation outside a published positive mask is assigned background.",
                utc=now())
    atomic_write(DATA / "fit-population.json", card)
    print(dict(frames=len(records), recordings=card["recordings"], positive_frames=dict(distribution), exclusions=blocked), flush=True)


@torch.no_grad()
def make_sparse_pool(engine, smoke):
    found = sparse_rows()
    ledger = read(DATA / "fit-population.json")
    if ledger["source_SHA256"] != digest(DATA / "release.json") or not matches_registered(ledger['code_SHA256'], digest(__file__)):
        raise ValueError("Sparse population changed after registration")
    rows = ledger["rows"][:8] if smoke else ledger["rows"]
    groups, support = defaultdict(list), torch.zeros(19, dtype=torch.int64)
    for i, r in enumerate(rows, 1):
        source, identity, labels, _ = found[r["group"]][r["id"]]
        if identity != r["rgb_SHA256"] or sha256(labels.tobytes()).hexdigest() != r["mask_SHA256"]:
            raise ValueError("Unexpected fit image or dense pixel data")
        y = torch.from_numpy(labels.copy()).cuda().float()
        view, _, _ = engine(open_rgb(source))
        label = F.interpolate(y[None, None], view.coarse.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
        if (label < 19).any():
            support += torch.bincount(label[label < 19].long(), minlength=19)
            groups[r["group"]].append((view.to("cpu", dtype=torch.float16), label))
        if i % 24 == 0:
            print(dict(encoded_sparse=i), flush=True)
    present = (support > 0) & (support <= .10 * support.sum())
    return tuple(groups.values()), class_examples(groups.values(), present), support


def protocol():
    return dict(code_SHA256=digest(__file__), data_SHA256=digest(DATA / "fit-population.json"),
                parent_SHA256=digest(PARENT / "completed.json"), seeds=list(SEEDS), arms=list(ARMS), steps=STEPS,
                design="Keep the same frozen raw anchor and two dense MUSES crops in every update. Add a third crop. "
                       "The dense control draws an identically budgeted MUSES crop instead of a sparse fog crop. "
                       "All other controls see the identical sparse labels. Train raw, restored fusion, raw-confidence "
                       "weighting, and same-architecture weighting by the improvement in the correct-vs-competing class "
                       "logit margin from a histogram match to the actual dehazing. Test reversed weighting and "
                       "prediction-space color subtraction. Normalize additional-crop weights to the same mass.",
                primary="Compare full mIoU against raw_mixed, fusion_mixed and raw_uncertainty in three dense cohorts. "
                        "Require agreement with recording-wise positive-label recall on the new partial test. "
                        "Check reversed weights and replacement of the inference-time restored view with matched color.",
                fixed_inference_RGB=True)


def weights_for(mode, views, start, target):
    if mode not in ("raw_uncertainty", MAIN, "reverse_weighted"):
        return torch.ones_like(target, dtype=start.dtype)
    y = target.clamp(0, 18)[:, None]
    deficit = 1. - start.softmax(1).gather(1, y)
    if mode == "raw_uncertainty":
        return (1. + 2. * deficit)[:, 0]
    best, second = start.topk(2, dim=1).indices.split(1, dim=1)
    competitor = torch.where(best == y, second, best)
    evidence = views.fine[:, 1] - views.fine[:, 2]
    improvement = (evidence.gather(1, y) - evidence.gather(1, competitor)) / 4.
    if mode == "reverse_weighted":
        improvement = -improvement
    return (1. + 2. * deficit * improvement.tanh().relu())[:, 0]


def edit(net, views, mode):
    if mode == "raw_mixed":
        form = "raw_extra"
    elif mode == "difference_mixed":
        form = "difference_tone"
    else:
        form = "plain_restored"
    return responses(net, views, form)


def fit_one(mode, seed, reference, dense, sparse, weights, smoke):
    path = ROOT / f"{mode}-{seed}.pth"
    ledger = path.with_suffix(".json")
    lock = digest(ROOT / "protocol.json") if not smoke else None
    if not smoke and ledger.is_file():
        old = read(ledger)
        if digest(path) != old["weights_SHA256"] or lock != old["protocol_SHA256"]:
            raise ValueError(f"Existing sparse checkpoint differs: {path.name}")
        return old
    set_seed(seed)
    net = DetailHead().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.00055, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.000055)
    gen = torch.Generator().manual_seed(86011 + seed)
    extra = torch.Generator().manual_seed(86031 + seed)
    n = 4 if smoke else STEPS
    for _ in range(n):
        primary, labels = batch(*dense, gen, "dense")
        batch_extra, extra_labels = batch(*(dense if mode == "fusion_dense" else sparse), extra, "extra")
        # Apply one additional crop in every arm, favouring the class-sampled crop.
        sample = batch_extra.cut(slice(1, 2))
        target = extra_labels[1:2]
        start = raw_anchor(reference, primary)
        update = edit(net, primary, mode)
        logits = start + update
        hit, union, count, _ = soft_counts(logits, labels)
        loss = F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
        loss += .8 * (1. - hit[count > 0] / union[count > 0].clamp_min(1e-5)).mean() + .015 * update.square().mean()
        original = raw_anchor(reference, sample)
        sparse_edit = edit(net, sample, mode)
        strength = weights_for(mode, sample, original, target)
        sparse_ce = F.cross_entropy(original + sparse_edit, target, weight=weights, ignore_index=255, reduction="none")
        mass = strength * weights[target.clamp(0, 18)] * (target < 19)
        loss += .5 * (strength * sparse_ce).sum() / mass.sum().clamp_min(1.)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    result = dict(method=mode, seed=seed, steps=n, params=sum(p.numel() for p in net.parameters()),
                  loss=float(loss.detach()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in net.state_dict().items()})
        result.update(weights_SHA256=digest(path), protocol_SHA256=lock)
        atomic_write(ledger, result)
    print(result, flush=True)
    return result


def fit(smoke):
    configure_cuda()
    cfg = protocol()
    lock = ROOT / "protocol.json"
    if not smoke:
        if lock.is_file() and read(lock) != cfg:
            raise ValueError("Sparse training scope changed")
        atomic_write(lock, cfg)
    raw = raw_models()
    engine = DualScaleFeatures()
    train, weights, rare = packets(engine, smoke)
    data, locations, support = make_sparse_pool(engine, smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    items = [fit_one(mode, seed, raw["raw_detail", seed], (train, rare), (data, locations), weights, smoke)
             for seed in seeds for mode in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=items, total_updates=sum(i["steps"] for i in items),
                     protocol_SHA256=digest(lock), sparse_grid_pixels=support.tolist(),
                     torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def load():
    lock = ROOT / "protocol.json"
    result = read(ROOT / "completed.json")
    if not matches_registered(read(lock), protocol()) or digest(lock) != result["protocol_SHA256"]:
        raise ValueError("Sparse experiment output is incompatible")
    states = {}
    for row in result["runs"]:
        key = row["method"], row["seed"]
        file = ROOT / f"{key[0]}-{key[1]}.pth"
        net = DetailHead().cuda().eval().requires_grad_(False)
        if row["steps"] != STEPS or digest(file) != row["weights_SHA256"] or row["params"] != sum(p.numel() for p in net.parameters()):
            raise ValueError(f"Bad frozen sparse model: {file.name}")
        net.load_state_dict(torch.load(file, weights_only=True, map_location="cpu"), strict=True)
        states[key] = net
    if set(states) != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("The sparse controls did not all finish")
    return states


def images(part):
    if part != "weather":
        for dataset, key, group, light, _, y, rgb in selection(part):
            yield dataset, key, group, light, rgb, y
        return
    cases, _ = held_frames()
    reg = read(HELD / "positive-coverage.json")
    for row in reg["frames"]:
        rgb, identity, labels, _ = cases[row["id"]]
        if identity != row["rgb_SHA256"] or sha256(labels.tobytes()).hexdigest() != row["mask_SHA256"]:
            raise ValueError("The sparse validation changed")
        yield "weatherreasonseg_fog", row["id"], row["recording"], "real fog", open_rgb(rgb), torch.from_numpy(labels).cuda().long()


@torch.inference_mode()
def score(part):
    configure_cuda()
    net, anchor = load(), raw_models()
    engine = DualScaleFeatures()
    rows, photos, population = [], {}, Counter()
    for ds, key, group, light, rgb, truth in images(part):
        cues, restored, author = engine(rgb)
        tone = Views(*(torch.stack((a[:, 0], a[:, 2], a[:, 2]), dim=1) for a in (cues.coarse, cues.fine)))
        for seed in SEEDS:
            start = raw_anchor(anchor["raw_detail", seed], cues)
            logits = {"author_raw": author, "raw_anchor": start}
            logits.update({a: start + edit(net[a, seed], cues, a) for a in ARMS})
            logits["recovery_color_at_test"] = start + edit(net[MAIN, seed], tone, MAIN)
            labels = {k: resize(v, truth.shape).argmax(1)[0] for k, v in logits.items()}
            rows.extend(dict(dataset=ds, id=key, group=group, method=k, seed=seed,
                             counts=confusion_matrix(v, truth, 19).tolist()) for k, v in labels.items())
            if seed == SEEDS[0] and (ds, group) not in photos and sum(k[0] == ds for k in photos) < 3:
                tags = ("raw_mixed", "fusion_mixed", "raw_uncertainty", MAIN)
                panel = [blend(rgb, labels[t].byte().cpu().numpy()) for t in tags]
                photos[ds, group] = (f"{key} / {light or ds}", [rgb, to_rgb(restored), *panel,
                                     paint(truth.byte().cpu().numpy())],
                                     ("Fog", "Restored RGB", "Raw + positive masks", "Fusion + positive masks",
                                      "Raw uncertainty", "Restoration weighted", "Labels"))
        population[ds] += 1
        if sum(population.values()) % 32 == 0:
            print(dict(part=part, sparse_trained_predictions=sum(population.values())), flush=True)
    total = sum(population.values())
    if total != {"held": 186, "external": 141, "weather": 100}[part]:
        raise ValueError(f"Unexpected label score population {population}")
    data = add_global_cohort(rows)
    contrasts = [(MAIN, a) for a in (*ARMS, "raw_anchor", "recovery_color_at_test") if a != MAIN]
    value = dict(positive_recall=group_recall(rows)) if part == "weather" else dict(
        metrics=aggregate(data), comparisons=paired_iou_intervals(data, contrasts, "method", "counts"))
    matrix, pic = ROOT / f"{part}-individual.json.gz", ROOT / f"{part}-predictions.png"
    compress_scores(matrix, rows)
    draw(pic, list(photos.values()))
    atomic_write(ROOT / f"{part}-test.json", dict(**value, counts=dict(population), rows_SHA256=digest(matrix),
                 preview_SHA256=digest(pic), weights_SHA256=digest(ROOT / "completed.json"), utc=now()))
    if part == "weather":
        print({m: round(v["mean_recall"], 4) for m, v in value["positive_recall"]["all"].items()}, flush=True)
    else:
        print({m: round(v["miou"], 4) for m, v in value["metrics"].items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("register", "smoke", "fit", "held", "external", "weather"))
    args = cli.parse_args()
    if args.stage == "register":
        register()
    elif args.stage in ("smoke", "fit"):
        fit(args.stage == "smoke")
    else:
        score(args.stage)

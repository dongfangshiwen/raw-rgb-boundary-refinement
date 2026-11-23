"""Score semantic donor colors on real adverse scenes and separate synthetic fog scenes."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.evaluation.semantic import evaluate_transport as earlier
from boundary_refinement.models.segmentation import matched_recall as roc
from boundary_refinement.models.operators import donor_colors as method
from boundary_refinement.data.staging import stage_vkitti_fog as virtual
from boundary_refinement.models.operators import subpixel_transport as baseline
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band, compress_scores
from boundary_refinement.models.segmentation.sparse_recovery import edit
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT, SEEDS = method.ROOT, method.SEEDS
REF = ("author_lidseg", "bilinear", "jbu", "dysample", "raw_free", "dual_free", "dual_mass", "known_negative")
METHODS = (*REF, *method.ARMS)
COMPARISONS = [(method.PRIMARY, a) for a in METHODS if a != method.PRIMARY] + [
    ("raw_core", "raw_free"), ("raw_free", "bilinear"), ("dual_free", "bilinear"),
    ("dual_free", "jbu"), ("dual_free", "dysample"), ("dual_mass", "dual_free")]


def grouped_scores(rows, key, common):
    grouped = defaultdict(dict)
    for row in rows:
        if key not in row:
            continue
        ds = row["dataset"]
        group = roc.capture_id(ds, row["group"])
        scopes = (ds, "foggy_driving_101") if key == "counts" and ds.startswith("foggy_driving_") else (ds,)
        for ds in scopes:
            if group not in grouped[ds]:
                grouped[ds][group] = np.zeros((len(METHODS), len(SEEDS), 19, 19), np.int64)
            grouped[ds][group][METHODS.index(row["method"]), SEEDS.index(row["seed"])] += row[key]
    out = {}
    for ds, bank in grouped.items():
        values = np.stack(list(bank.values()))
        n = len(bank)
        rng = np.random.default_rng(92033)
        draws = np.ones((1, n), dtype=np.int64)
        if n > 1:
            draws = np.vstack((draws, rng.multinomial(n, np.full(n, 1 / n), 2000)))
        scores, paired, seeds = {}, {}, {}
        for j, arm in enumerate(METHODS):
            groups = values[:, j].reshape(n, -1)
            stats = []
            for chunk in np.array_split(draws, max(1, len(draws) // 100)):
                cm = (chunk @ groups).reshape(-1, len(SEEDS), 19, 19)
                hit = cm.diagonal(axis1=-1, axis2=-2)
                union = cm.sum(-1) + cm.sum(-2) - hit
                iou = np.divide(100. * hit, union, out=np.zeros_like(hit, dtype=float), where=union > 0)
                stats.append(np.stack((iou[..., list(common)].mean(-1), iou[..., 6], iou[..., 7], iou[..., 14]), -1))
            per = np.concatenate(stats)
            seeds[arm] = per[0].tolist()
            paired[arm] = per.mean(1)
            scores[arm] = paired[arm][0].tolist()
        effects = {}
        for a, b in COMPARISONS:
            delta = paired[a] - paired[b]
            effects[f"{a}-vs-{b}"] = dict(diff=delta[0].tolist(), ci95=(
                np.quantile(delta[1:], [.025, .975], axis=0).T.tolist() if n > 1 else None))
        out[ds] = dict(scores=scores, by_seed=seeds, groups=list(bank), effects=effects,
                       gt_pixels=values[:, 0, 0].sum((0, 2)).tolist(), mean_classes=list(common))
    return out


def frozen_fields(z, guides, old, new, seed, names):
    stencil = baseline.guide_stencil(guides, z.argmax(1)[0])
    trust = {}
    for arm in names:
        if arm in method.ARMS:
            mode = arm if arm in ("center_core", "roll_core") else method.PRIMARY
            if mode not in trust:
                trust[mode] = method.color_weights(z, mode)
            cues = method.prepare(guides, z, arm, stencil, trust[mode])
            yield arm, baseline.prediction("dual_free", new[arm, seed], z, cues)
        elif arm == "jbu":
            yield arm, earlier.fixed_bilateral(guides, z)
        else:
            yield arm, baseline.prediction(arm, old[arm, seed], z, stencil)


def parent_rows(part):
    packed = baseline.ROOT / f"{part}-individual.json.gz"
    histfile = baseline.ROOT / f"{part}-histograms.npz"
    ledger = read(baseline.ROOT / f"{part}-test.json")
    if digest(packed) != ledger["rows_SHA256"] or digest(histfile) != ledger["hist_SHA256"]:
        raise ValueError("V92 comparison artifacts changed")
    with gzip.open(packed, "rt", encoding="utf-8") as stream:
        rows = [r for r in json.load(stream) if r["method"] in REF]
    with np.load(histfile, allow_pickle=False) as archive:
        keys, methods, counts = archive["names"].tolist(), archive["methods"].tolist(), archive["counts"]
    hist = {}
    for i, name in enumerate(keys):
        bank = np.zeros((len(METHODS), *counts.shape[2:]), dtype=np.uint32)
        for arm in REF:
            bank[METHODS.index(arm)] = counts[i, methods.index(arm)]
        hist[tuple(name)] = bank
    return rows, hist


def write_sample(directory, rgb, restored, truth, pred, key, ds):
    directory.mkdir(parents=True, exist_ok=True)
    photos = {"input": rgb, "restored": to_rgb(resize(restored, truth.shape)), "truth": truth.byte().cpu().numpy(), **pred}
    for name, arr in photos.items():
        Image.fromarray(arr).save(directory / f"{name}.png")
    atomic_write(directory / "example.json", dict(dataset=ds, id=key, shape=list(truth.shape), seed=SEEDS[0],
                 selection="First fine-labelled frame in iteration order with at least 5 traffic light pixels.",
                 files={name: digest(directory / f"{name}.png") for name in photos}))


@torch.inference_mode()
def evaluate(part):
    configure_cuda()
    sim = part == "virtual"
    rows, hist = ([], {}) if sim else parent_rows(part)
    old_rows = {(r["dataset"], r["id"], r["seed"]): r for r in rows if r["method"] == "bilinear"}
    iterator, expected, source = ((virtual.cases(), read(virtual.HELD)["counts"], virtual.HELD) if sim else earlier.cases(part))
    engine, raw, old, new = baseline.DualScaleFeatures(), baseline.anchors(), baseline.load(), method.load()
    previous = earlier.previous.load() if sim else None
    counts, shown = Counter(), set()
    for ds, key, group, _, rgb, truth in iterator:
        features, restored, author = engine(rgb)
        guides = baseline.render_guides(rgb, restored, features.fine.shape[-2:])
        edge = None if ds == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        want = edge is not None and ds not in shown and int((truth == 6).sum()) >= 5
        example = {}
        if (ds, group) not in hist:
            hist[ds, group] = np.zeros((len(METHODS), len(SEEDS), len(roc.CLASSES), 2, roc.BINS), dtype=np.uint32)
        bank = hist[ds, group]
        for trial, seed in enumerate(SEEDS):
            z = raw_anchor(raw["raw_detail", seed], features)
            native = resize(z, truth.shape)
            direct = native.argmax(1)[0]
            if not sim:
                check = old_rows[ds, key, seed]
                if confusion_matrix(direct, truth, 19).tolist() != check["counts"] or check["group"] != group:
                    raise ValueError("Replayed native raw predictions differ from V92")
            if want and trial == 0:
                example["bilinear"] = direct.byte().cpu().numpy()
            def outputs():
                if sim:
                    yield "bilinear", native
                    yield "author_lidseg", resize(author, truth.shape)
                    yield "known_negative", resize(z + edit(previous[earlier.previous.PRIMARY, seed], features, "fusion_mixed"), truth.shape)
                names = [s for s in (METHODS if sim else method.ARMS) if s not in ("bilinear", "known_negative", "author_lidseg")]
                if want and trial == 0 and not sim:
                    names += ["jbu", "dysample", "dual_free"]
                for name, delta in frozen_fields(z, guides, old, new, seed, names):
                    yield name, native + resize(delta, truth.shape)
            for arm, output in outputs():
                pred = output.argmax(1)[0]
                if want and trial == 0 and arm in ("jbu", "dysample", "dual_free", method.PRIMARY):
                    example[arm] = pred.byte().cpu().numpy()
                if not sim and arm in REF:
                    continue
                record = dict(dataset=ds, group=group, id=key, method=arm, seed=seed, counts=confusion_matrix(pred, truth, 19).tolist())
                if edge is not None:
                    record["edge"] = confusion_matrix(pred, edge, 19).tolist()
                rows.append(record)
                bank[METHODS.index(arm), trial] += roc.pixel_histograms(output, truth).astype(np.uint32)
        if want:
            write_sample(ROOT / "examples" / ds, rgb, restored, truth, example, key, ds)
            shown.add(ds)
        counts[ds] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(part=part, frames=sum(counts.values())), flush=True)
    if dict(counts) != expected or len(rows) != len(METHODS) * len(SEEDS) * sum(counts.values()):
        raise ValueError("Incomplete semantic subpixel evaluation")
    matrix, histfile = ROOT / f"{part}-individual.json.gz", ROOT / f"{part}-histograms.npz"
    compress_scores(matrix, rows)
    roc.save_histograms(histfile, hist, METHODS)
    ids = virtual.CLASSES if sim else range(19)
    full, band = grouped_scores(rows, "counts", ids), grouped_scores(rows, "edge", ids)
    atomic_write(ROOT / f"{part}-test.json", dict(population=dict(counts), methods=list(METHODS), full=full, edge=band,
                 recall=roc.summarize(histfile, COMPARISONS), hist_SHA256=digest(histfile), rows_SHA256=digest(matrix),
                 source_SHA256=digest(source), model_SHA256=digest(ROOT / "completed.json"),
                 old_SHA256=digest(baseline.ROOT / "completed.json"), evaluator_SHA256=digest(__file__), utc=now()))
    for ds, data in band.items():
        print(ds, {m: round(v[0], 3) for m, v in data["scores"].items()}, flush=True)


@torch.inference_mode()
def predict(image, output, arm, seed):
    configure_cuda()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty prediction folder")
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image) as photo:
        rgb = np.array(photo.convert("RGB"))
    views, restored, _ = baseline.DualScaleFeatures()(rgb)
    z = raw_anchor(baseline.anchors()["raw_detail", seed], views)
    g = baseline.render_guides(rgb, restored, views.fine.shape[-2:])
    ((_, delta),) = frozen_fields(z, g, baseline.load(), method.load(), seed, [arm])
    mask = (resize(z, rgb.shape[:2]) + resize(delta, rgb.shape[:2])).argmax(1)[0].cpu().byte().numpy()
    Image.fromarray(to_rgb(resize(restored, rgb.shape[:2]))).save(output / "restored.png")
    Image.fromarray(mask).save(output / "mask_train_ids.png")
    atomic_write(output / "prediction.json", dict(image_SHA256=digest(image), model=arm, seed=seed, size=list(mask.shape),
                 RGB_SHA256=digest(output / "restored.png"), mask_SHA256=digest(output / "mask_train_ids.png")))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    sub.add_parser("score").add_argument("parts", nargs="+", choices=("external", "rain", "snow", "virtual"))
    dec = sub.add_parser("predict")
    dec.add_argument("image", type=Path)
    dec.add_argument("output", type=Path)
    dec.add_argument("--arm", choices=(*method.ARMS, "raw_free", "dual_free", "dysample", "jbu"), default=method.PRIMARY)
    dec.add_argument("--seed", choices=SEEDS, type=int, default=SEEDS[0])
    args = cli.parse_args()
    if args.action == "score":
        for part in args.parts:
            evaluate(part)
    else:
        predict(args.image, args.output, args.arm, args.seed)

"""Evaluate the frozen subpixel heads, author DySample and fixed bilateral interpolation."""
import argparse
from collections import Counter, defaultdict
import gzip
import json

import numpy as np
from PIL import Image
import torch

from dehaze_seg.metrics import confusion_matrix
from scripts import false_positive_constraint_v87 as previous
from scripts import matched_recall_v88 as roc
from scripts import stage_muses_rain_v90 as rain
from scripts import stage_muses_snow_v92 as snow
from scripts import subpixel_transport_v92 as train
from scripts.contrastive_response_v85 import raw_anchor
from scripts.cross_scale_rgb_v44 import resize
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import digest, now, read
from scripts.restoration_affinity_v75 import boundary_band, compress_scores
from scripts.sparse_recovery_v86 import edit, images
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


ROOT, SEEDS = train.ROOT, train.SEEDS
METHODS = ("author_lidseg", "bilinear", "jbu", "known_negative", *train.ARMS)
COMPARISONS = [(train.PRIMARY, a) for a in METHODS if a != train.PRIMARY] + [
    ("raw_mass", "raw_free"), ("dual_free", "raw_free"), ("raw_free", "bilinear"), ("dysample", "bilinear")]


def fixed_bilateral(guides, logits):
    # Direct evaluation of the Gaussian spatial and range kernels over the whole 3x3 footprint.
    h, w = logits.shape[-2:]
    pixels = train.pack(guides[0], h, w).float() / 255.
    low = pixels.mean(1)
    z = logits[0].permute(1, 2, 0).reshape(-1, 19)
    base, offsets, distance = train.geometry(logits.device)
    parts = []
    for block in torch.arange(h * w, device=logits.device).split(4096):
        neighbor = ((block[:, None] // w + offsets[:, 0]).clamp(0, h - 1) * w +
                    (block[:, None] % w + offsets[:, 1]).clamp(0, w - 1))
        color = (pixels[block, :, None] - low[neighbor][:, None]).square().mean(-1)
        weight = (-.5 * (distance / .65 ** 2 + color / .05 ** 2)).softmax(-1)
        parts.append((weight - base) @ z[neighbor])
    return train.unpack(torch.cat(parts), h, w)


def score_table(rows, edge):
    by_group = defaultdict(dict)
    for row in rows:
        if edge and "edge" not in row:
            continue
        ds, capture = row["dataset"], roc.capture_id(row["dataset"], row["group"])
        scopes = (ds, "foggy_driving_101") if not edge and ds.startswith("foggy_driving_") else (ds,)
        for ds in scopes:
            group = by_group[ds].setdefault(capture, np.zeros((len(METHODS), len(SEEDS), 19, 19), dtype=np.int64))
            group[METHODS.index(row["method"]), SEEDS.index(row["seed"])] += row["edge" if edge else "counts"]
    table = {}
    for ds, groups in by_group.items():
        counts = np.stack(list(groups.values()))
        rng = np.random.default_rng(92033)
        n = len(groups)
        draws = np.ones((1, n), dtype=np.int64)
        if n > 1:
            draws = np.vstack((draws, rng.multinomial(n, np.ones(n) / n, 2000)))
        metrics = {}
        by_seed = {}
        for j, arm in enumerate(METHODS):
            estimates = []
            for block in np.array_split(draws, max(1, len(draws) // 100)):
                matrix = (block @ counts[:, j].reshape(n, -1)).reshape(-1, 3, 19, 19)
                hit = matrix.diagonal(axis1=-2, axis2=-1)
                union = matrix.sum(-2) + matrix.sum(-1) - hit
                iou = np.divide(100. * hit, union, out=np.zeros_like(hit, dtype=float), where=union > 0)
                estimates.append(np.stack((iou.mean(-1), iou[..., 6], iou[..., 7], iou[..., 14]), axis=-1))
            data = np.concatenate(estimates)
            metrics[arm] = data.mean(1)
            by_seed[arm] = data[0].tolist()
        effects = {}
        def record(name, delta):
            effects[name] = dict(diff=delta[0].tolist(), ci95=np.percentile(delta[1:], (2.5, 97.5), axis=0).T.tolist()
                                 if n > 1 else None)
        for a, b in COMPARISONS:
            record(f"{a}-vs-{b}", metrics[a] - metrics[b])
        record("conservation_x_restoration", metrics["dual_mass"] - metrics["dual_free"] -
               metrics["raw_mass"] + metrics["raw_free"])
        table[ds] = dict(groups=list(groups), endpoints=["miou19", "traffic_light_IoU", "traffic_sign_IoU", "truck_IoU"],
                         scores={m: v[0].tolist() for m, v in metrics.items()}, by_seed=by_seed, effects=effects)
    return table


def cases(part):
    if part == "rain":
        return rain.cases(), read(rain.MANIFEST)["counts"], rain.MANIFEST
    if part == "snow":
        return snow.cases(), read(snow.MANIFEST)["counts"], snow.MANIFEST
    file = previous.ROOT / "external-test.json"
    return images("external"), read(file)["counts"], file


def save_example(ds, key, rgb, restored, truth, masks):
    directory = ROOT / "examples" / ds
    directory.mkdir(parents=True, exist_ok=True)
    photos = {"input": rgb, "restored": to_rgb(resize(restored, truth.shape)), "truth": truth.cpu().byte().numpy(), **masks}
    for name, arr in photos.items():
        Image.fromarray(arr).save(directory / f"{name}.png")
    atomic_write(directory / "example.json", dict(dataset=ds, id=key, source_shape=list(truth.shape), seed=SEEDS[0],
                 selection="First fine-labelled frame in iterator with at least five class-6 pixels.",
                 files={k: digest(directory / f"{k}.png") for k in photos}))


@torch.inference_mode()
def evaluate(part):
    configure_cuda()
    nets, raw, old = train.load(), train.anchors(), previous.load()
    engine = train.DualScaleFeatures()
    rows, hist, counts, shown = [], {}, Counter(), set()
    iterator, total, source = cases(part)
    for ds, key, group, _, rgb, truth in iterator:
        views, restored, author = engine(rgb)
        guides = train.render_guides(rgb, restored, views.fine.shape[-2:])
        edge = None if ds == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        want = edge is not None and ds not in shown and int((truth == 6).sum()) >= 5
        masks = {}
        bucket = hist.setdefault((ds, group), None)
        if bucket is None:
            bucket = hist[ds, group] = np.zeros((len(METHODS), 3, len(roc.CLASSES), 2, roc.BINS), dtype=np.uint32)
        for trial, seed in enumerate(SEEDS):
            z = raw_anchor(raw["raw_detail", seed], views)
            native = resize(z, truth.shape)
            stencil = train.guide_stencil(guides, z.argmax(1)[0])
            for j, arm in enumerate(METHODS):
                if arm == "author_lidseg":
                    output = resize(author, truth.shape)
                elif arm == "bilinear":
                    output = native
                elif arm == "known_negative":
                    output = resize(z + edit(old[previous.PRIMARY, seed], views, "fusion_mixed"), truth.shape)
                else:
                    delta = fixed_bilateral(guides, z) if arm == "jbu" else train.prediction(arm, nets[arm, seed], z, stencil)
                    output = native + resize(delta, truth.shape)
                pred = output.argmax(1)[0]
                cm = confusion_matrix(pred, truth, 19)
                record = dict(dataset=ds, id=key, group=group, seed=seed, method=arm, counts=cm.tolist())
                if edge is not None:
                    record["edge"] = confusion_matrix(pred, edge, 19).tolist()
                rows.append(record)
                bucket[j, trial] += roc.pixel_histograms(output, truth).astype(np.uint32)
                if want and trial == 0 and arm in ("bilinear", "dysample", train.PRIMARY):
                    masks[arm] = pred.byte().cpu().numpy()
        if want:
            save_example(ds, key, rgb, restored, truth, masks)
            shown.add(ds)
        counts[ds] += 1
        if sum(counts.values()) % 16 == 0:
            print(dict(part=part, frames=sum(counts.values())), flush=True)
    if dict(counts) != total:
        raise ValueError(f"Population differs from registered manifest: {counts}")
    histfile, matrix = ROOT / f"{part}-histograms.npz", ROOT / f"{part}-individual.json.gz"
    roc.save_histograms(histfile, hist, METHODS)
    compress_scores(matrix, rows)
    full, band = score_table(rows, False), score_table(rows, True)
    curves = roc.summarize(histfile, COMPARISONS)
    atomic_write(ROOT / f"{part}-test.json", dict(population=dict(counts), methods=list(METHODS), full=full, edge=band,
                 recall=curves, rows_SHA256=digest(matrix), hist_SHA256=digest(histfile), model_SHA256=digest(ROOT / "completed.json"),
                 source_SHA256=digest(source), scorer_SHA256=digest(__file__), utc=now()))
    for ds, table in band.items():
        print(ds, {m: round(v[0], 3) for m, v in table["scores"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", nargs="+", choices=("external", "rain", "snow"))
    for item in parser.parse_args().part:
        evaluate(item)

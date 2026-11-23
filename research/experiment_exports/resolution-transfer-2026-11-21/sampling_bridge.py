"""Replay unchanged predictions with dense and sparse reference sampling, and test score placement."""
from hashlib import sha256

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch

from cohort import OUT, ROOT, digest, read, write
from scale_training import SEEDS, image_at
import evaluate as dense
import stf_score as fog
from stf_data import TO_CITY
import gate
from scripts.score_path_v104 import metrics

ARMS = ("released", "paper", "native_paper", "fixed", "native_fixed")
REGIONS = ("dense19", "dense13", "sampled_all", "sampled_dense_band", "sampled_near12")
CLASSES = sorted(TO_CITY.values())
PAIRS = {"paper": ("released", "fixed"),
         "native_paper": ("released", "paper", "native_fixed"),
         "native_fixed": ("released", "fixed")}


def lock():
    rows = read(OUT / "new-test.json")["rows"]
    fog_rows = read(OUT / "stf-manifest.json")["rows"]
    def ranked(rows, tag):
        return sorted(rows, key=lambda v: sha256((tag + v["id"]).encode()).digest())
    template = ranked(fog_rows, "bridge-template/")
    pairs = {r["id"]: template[i % len(template)]["id"]
             for i, r in enumerate(ranked(rows, "bridge-frame/"))}
    card = dict(stage="Diagnostic designed after reading both primary results. No parameter fitting or weather selection.",
                weights=digest(OUT / "training.json"), dense=digest(OUT / "results.json"),
                fog=digest(OUT / "stf-results.json"), code=digest(__file__), pairs=pairs,
                arms=list(ARMS), regions=list(REGIONS), main="768 long edge; three archived manuscript seeds.",
                placement="Add exactly the same colour residual to the released native backbone scores or to enlarged coarse scores. Both paths are algebraically identical for SegFormer-B1, which is an exact control.",
                sampling="Transfer only the XY coordinates of matched-class, visible, projected returns. Assign templates in cyclic hash order without consulting masks. Map pixel centers by normalized image width/height and de-duplicate. Obtain class and six-pixel native dense band from the BDD mask. For the sparse-near12 endpoint use two measured BDD classes within 12 pixels in canonical 1920 x 1024 coordinates. The original ground labels from the template do not transfer.",
                ontology="For matched endpoints use 13 Cityscapes classes and exclude other reference labels. All 19 model outputs remain eligible, including wrong predictions into excluded classes. Macro class IoU with zero for zero union. Full dense19 uses all original classes.",
                resampling="5000 paired draws of BDD prefix / SemanticSTF capture day, RNG 11329, seed mean after pooling. Diagnostic intervals are descriptive.",
                scope="A sparse sample of a camera mask retains camera geometry. This test does not simulate lidar measurement error or repeat captures in the same weather, and cannot assign causality to any domain factor.")
    path = OUT / "sampling-bridge-protocol.json"
    write(path, card, True)
    return digest(path), pairs, rows, fog_rows


def sparse_bdd(row, template, y, edge):
    with np.load(OUT / "measured-points/mask2former" / (template + ".npz")) as blob:
        xy = blob["xy"].astype(float)
        src_h, src_w = blob["shape"]
    h, w = y.shape
    scale = np.array((src_w / w, src_h / h))
    mapped = np.floor((xy + .5) / scale).astype(int)
    flat = np.unique(mapped[:, 1] * w + mapped[:, 0])
    assert np.all((flat >= 0) & (flat < w * h))
    gt = y.reshape(-1)[flat]
    valid = np.isin(gt, CLASSES)
    flat, gt = flat[valid], gt[valid]
    coordinates = (np.stack((flat % w, flat // w), 1) + .5) * scale
    near = np.zeros(len(gt), bool)
    for label in np.unique(gt):
        keep = gt == label
        if (~keep).any():
            other = cKDTree(coordinates[~keep])
            near[keep] = other.query(coordinates[keep], distance_upper_bound=12.000001)[0] <= 12
    return flat, gt, (np.ones(len(gt), bool), edge.reshape(-1)[flat], near)


def native_labels(rgb, target, model, paper, fixed):
    shape = rgb.shape[:2]
    image = image_at(rgb, 768)
    scores, base = model(image, shape)
    coarse = gate.ev.resize(scores, shape)
    if target == "segformer":
        assert torch.equal(base, coarse)
    guide = image.mul(255).round().byte()
    stencil = gate.ev.legacy.guide_stencil(guide[[0, 0]], scores.argmax(1)[0])
    def labels(tensor):
        return tensor.argmax(1)[0].byte().cpu().numpy()
    yield "released", None, labels(base)
    residual = gate.ev.resize(fixed.correction(scores, gate.ev.prepare(scores, guide)), shape)
    yield "fixed", None, labels(coarse + residual)
    yield "native_fixed", None, labels(base + residual)
    for j, seed in enumerate(SEEDS):
        d = gate.ev.legacy.prediction("dual_free", paper["dual_core", seed], scores, stencil)
        residual = gate.ev.resize(d, shape)
        yield "paper", j, labels(coarse + residual)
        yield "native_paper", j, labels(base + residual)


def hist(y, pred):
    return np.bincount(19 * y.astype(int) + pred, minlength=361).reshape(19, 19)


def gated_dense(y, edge, guess):
    keep = edge & (y < 19)
    cm = hist(y[keep], guess[keep])
    matched = cm.copy()
    matched[~np.isin(np.arange(19), CLASSES)] = 0
    return cm, matched


@torch.inference_mode()
def run():
    sig, pairs, bdd_rows, fog_rows = lock()
    gate.ev.setup.configure_cuda()
    paper = gate.ev.original.load()
    fixed = gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    root = OUT / "bridge-counts"
    for target in dense.TARGETS:
        model = gate.ev.Target(target)
        for dataset, rows in (("bdd", bdd_rows), ("stf", fog_rows)):
            directory = root / target / dataset
            directory.mkdir(exist_ok=True, parents=True)
            for i, r in enumerate(rows):
                file = directory / (r["id"] + ".npz")
                if file.exists():
                    with np.load(file) as f:
                        assert f["lock"].item() == sig
                    continue
                if dataset == "bdd":
                    rgb, truth = dense.prior_scores.load_case(r)
                    y = truth.cpu().numpy().astype(np.uint8)
                    edge = gate.boundary_band(truth).cpu().numpy()
                    flat, point_y, masks = sparse_bdd(r, pairs[r["id"]], y, edge)
                    n = len(REGIONS)
                else:
                    with Image.open(ROOT / r["paths"][0]) as im:
                        rgb = np.array(im.convert("RGB"))
                    with np.load(OUT / "measured-points" / target / (r["id"] + ".npz")) as f:
                        xy, point_y, bands = f["xy"].astype(int), f["truth"], f["band"]
                        old = f["pred"][0]
                    flat = xy[:, 0] + rgb.shape[1] * xy[:, 1]
                    masks = (np.ones(len(flat), bool), bands <= 1)
                    n = len(masks)
                measured = np.zeros((n, len(ARMS), len(SEEDS), 19, 19), np.int32)
                projected = np.empty((len(ARMS), len(SEEDS), len(flat)), np.uint8)
                for name, seed, guess in native_labels(rgb, target, model, paper, fixed):
                    ss = slice(None) if seed is None else seed
                    at = guess.ravel()[flat]
                    table = [hist(point_y[q], at[q]) for q in masks]
                    if dataset == "bdd":
                        table = list(gated_dense(y, edge, guess)) + table
                    else:
                        if name in ("released", "fixed", "paper"):
                            np.testing.assert_array_equal(at, old[fog.ARMS.index(name), seed or 0])
                    table = np.stack(table)
                    measured[:, ARMS.index(name), ss] = table[:, None] if seed is None else table
                    projected[ARMS.index(name), ss] = at
                if dataset == "bdd":
                    with np.load(OUT / "counts" / target / "new_bdd" / (r["id"] + ".npz")) as f:
                        old_cm = f["counts"][1, 1]
                    for name in ("released", "fixed", "paper"):
                        np.testing.assert_array_equal(measured[0, ARMS.index(name)], old_cm[dense.ARMS.index(name)])
                if target == "segformer":
                    for a, b in (("paper", "native_paper"), ("fixed", "native_fixed")):
                        np.testing.assert_array_equal(measured[:, ARMS.index(a)], measured[:, ARMS.index(b)])
                assert np.all(measured.sum(-1) == measured.sum(-1)[:, :1, :1])
                dense.save(file, counts=measured, point_pred=projected, truth=point_y, near=masks[-1], lock=sig)
                if i == 0 or (i + 1) % 32 == 0 or i + 1 == len(rows):
                    print("Sampling bridge", target, dataset, i + 1, "/", len(rows), flush=True)
        del model
        torch.cuda.empty_cache()


def summarize():
    sig, _, bdd_rows, stf_rows = lock()
    out = {}
    for dataset, rows in (("bdd", bdd_rows), ("stf", stf_rows)):
        regions = REGIONS if dataset == "bdd" else ("sampled_all", "sampled_near12")
        ids = np.array([r["group"] if dataset == "bdd" else r["day"] for r in rows])
        groups = sorted(set(ids))
        out[dataset] = {}
        for target in dense.TARGETS:
            cms = []
            transitions = np.zeros((len(ARMS), 3, 3), np.int64)
            for r in rows:
                with np.load(OUT / "bridge-counts" / target / dataset / (r["id"] + ".npz")) as f:
                    assert f["lock"].item() == sig
                    cm, y, pred, use = f["counts"], f["truth"], f["point_pred"], f["near"]
                cms.append(cm)
                p = pred[:, :, use]
                ok = p == y[use]
                orig = ok[ARMS.index("released")]
                transitions += np.stack(((ok & ~orig).sum(-1), (~ok & orig).sum(-1),
                                         ((p != p[:1]) & ~ok & ~orig).sum(-1)), -1)
                for a, s in np.ndindex(len(ARMS), 3):
                    np.testing.assert_array_equal(cm[-1, a, s], hist(y[use], p[a, s]))
            arr = np.stack(cms)
            group_cm = np.stack([arr[ids == group].sum(0, dtype=np.int64) for group in groups])
            gt = group_cm.sum(-1)
            hit = group_cm.diagonal(axis1=-1, axis2=-2)
            union = gt + group_cm.sum(-2) - hit
            all_stats = np.stack((hit, union, gt), -1)
            out[dataset][target] = {}
            for k, reg in enumerate(regions):
                classes = list(range(19)) if reg == "dense19" else CLASSES
                stats = np.take(all_stats[:, k], classes, axis=-2)[:, None]
                result = metrics(stats, groups, list(ARMS), [target], PAIRS, 11329)
                out[dataset][target][reg] = dict(**result["targets"][target],
                    groups=result["groups"], points=sum(result["class_support_pixels"]))
            out[dataset][target]["changed_near_points"] = dict(
                description="Per seed: corrected, newly incorrect, changed incorrect to another incorrect relative to released.",
                counts={name: transitions[j].tolist() for j, name in enumerate(ARMS)})
            print(dataset, target, {r: round(out[dataset][target][r]["contrasts"]["paper - released"]["pp"], 3)
                                   for r in regions}, flush=True)
    write(OUT / "sampling-bridge-results.json", dict(lock=sig, datasets=out))


if __name__ == "__main__":
    run()
    summarize()

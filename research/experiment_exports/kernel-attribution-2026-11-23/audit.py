"""Frozen 2 x 2 parameter substitution on the manuscript's 24 IDD-AW fog drives."""
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
PREVIOUS = HERE.parent / "decoder-resolution-2026-11-21"
sys.path.insert(0, str(WORK / "colorawarenet-joint-model"))

from scripts import audit_fixed_ranges_v106 as factors
from scripts import eval_range_v101 as ev

ARMS = ("study_reference", "no_correction", "fixed", "colour_only", "spatial_only", "both", "paper")
NET_NAMES = {"colour_only": "fit_range", "spatial_only": "fit_spatial", "both": "both_v101"}
SEEDS = (101, 211, 307)
TARGETS = ("mask2former", "segformer")
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18]
PAIR_NAMES = ("fixed_minus_no_correction", "colour_only_minus_fixed", "spatial_only_minus_fixed",
              "both_minus_fixed", "paper_minus_fixed")


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf8")
    temp.replace(path)


def protocol():
    preceding = ev.read(PREVIOUS / "protocol.json")
    rows = {r["id"]: r for r in ev.read(ev.IDD / "evaluation.json")["rows"] if "fog" in r["cohorts"]}
    assert len(rows) == len(preceding["frames"]) == 146
    manifest = []
    for item in preceding["frames"]:
        row = rows[item["id"]]
        assert all(row[k] == item[k] for k in ("id", "group", "rgb_sha", "mask_sha"))
        manifest.append(row)
    assert len(set(r["group"] for r in manifest)) == 24
    plan = {
        "status": "Retrospective mechanism diagnostic. Cohort, aggregate full-vs-fixed outcomes, and related ablations have already been examined. No target tuning, no new independent data, no training.",
        "question": "At the manuscript's 512-pixel input long edge, which part of the fixed-to-source-fitted boundary change comes from the four colour amplitudes and which from the bilinear spatial exponent?",
        "fixed_values": {"ranges": [0.02, 0.05, 0.10, 0.20], "amplitude_per_range": "6*sigmoid(-0.5)", "spatial_exponent": 1},
        "matched_2x2": "Use the same RangeDecoder in all four cells, raw RGB, transition mask, donors, base scores and native grid. Transplant source parameters; these interventions are not separate training runs.",
        "arms": list(ARMS), "targets": list(TARGETS), "seeds": list(SEEDS), "classes": CLASSES,
        "primary": "Pointwise paired 95% intervals for colour-only minus fixed on native radius-six boundary mIoU, shown for both frozen targets. Also report spatial-only, fully fitted and their interaction.",
        "secondary": "Full and outside-band mIoU, change counts, and the directly paired difference between the two target effects. Use the previously archived 1024 results as a sensitivity check without retuning.",
        "statistics": "Cluster the 146 frames by their 24 drive IDs. Pool class counts before computing macro IoU; zero for missing union. Average the three source seeds as repeated predictions. Use the same 5000 multinomial drive replicates and seed 11221 as the earlier resolution audit. Pointwise two-sided percentile intervals; no familywise claim. Pixel changes average over seeds, never treat pixels or seeds as independent samples.",
        "validation": "Recompute native confusion matrices of study reference, no correction, fixed and manuscript for every frame and seed; require exact agreement with the archived 512 run. Track rounding differences between the merged-parameter implementation and original paper outputs. For each mask substitution validate changed correct/incorrect pixel counts against native confusion traces.",
        "old_protocol_sha256": ev.digest(PREVIOUS / "protocol.json"),
        "old_count_sha256": {n: ev.digest(PREVIOUS / f"counts-{n}.npz") for n in TARGETS},
        "checkpoint_hashes": preceding["fitted_decoder_sha256"],
        "backbone_hashes": preceding["backbone_weights_sha256"],
        "sha256_of_code": {Path(path).name: ev.digest(path) for path in
                           (__file__, ev.__file__, factors.__file__, ev.legacy.__file__,
                            sys.modules[ev.RangeDecoder.__module__].__file__)},
        "dataset": "Same ordered frames and image/label hashes as old protocol, unchanged physical images. Native 2048 x 1536.",
    }
    path = HERE / "protocol.json"
    if path.exists():
        assert ev.read(path) == plan, "Protocol changed; do not mix results."
    else:
        write_json(path, plan)
    return manifest, ev.digest(path)


class ScoreFrame:
    def __init__(self, truth):
        metric = ev.ev.Counts(truth)
        self.valid = metric.valid
        self.key = metric.base
        self.y = (self.key // 19 % 19).astype(np.uint8)
        self.region = (self.key // (19 * 19)).astype(np.uint8)

    def labels(self, scores):
        return scores.argmax(1)[0].byte().cpu().numpy().ravel()[self.valid]

    def confusion(self, labels):
        two = np.bincount(self.key + labels, minlength=2*19*19).reshape(2, 19, 19)
        return np.stack((two[0]+two[1], two[1])).astype(np.int32)

    def flips(self, before, after):
        changed = np.flatnonzero(before != after)
        new_correct = after[changed] == self.y[changed]
        old_correct = before[changed] == self.y[changed]
        # Indices 0, 1, 2 denote corrected, spoiled and still incorrect.
        event = np.where(new_correct, 0, np.where(old_correct, 1, 2))
        cells = np.bincount(3*self.region[changed] + event, minlength=6).reshape(2, 3)
        return np.stack((cells[0]+cells[1], cells[1]))


def one_frame(model, fits, nets, row):
    rgb, truth = ev.scene(row, "idd")
    score = ScoreFrame(truth)
    size = rgb.shape[:2]
    assert size == (1536, 2048)
    img = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
    img = ev.resize(img, (384, 512), antialias=True).contiguous()
    z, reference = model(img, size)
    guide = img.clamp(0, 1).mul(255).round().byte()
    st = ev.prepare(z, guide)
    original_stencil = ev.legacy.guide_stencil(guide[[0, 0]], z.argmax(1)[0])
    counts = np.empty((2, len(ARMS), len(SEEDS), 19, 19), dtype=np.int32)
    flips = np.empty((2, len(PAIR_NAMES), len(SEEDS), 3), dtype=np.int64)
    rounding = np.zeros((2, len(SEEDS)), dtype=np.int64)

    counts[:, 0] = score.confusion(score.labels(reference))[:, None]
    del reference
    base = ev.resize(z, size)
    uncorrected = score.labels(base)
    counts[:, 1] = score.confusion(uncorrected)[:, None]

    def from_delta(delta):
        return score.labels(base + ev.resize(delta, size))

    fixed = from_delta(nets["fixed4", 0].correction(z, st))
    counts[:, 2] = score.confusion(fixed)[:, None]
    flips[:, 0] = score.flips(uncorrected, fixed)[:, None]
    for j, seed in enumerate(SEEDS):
        for k, arm in enumerate(("colour_only", "spatial_only", "both"), start=3):
            labels = from_delta(nets[NET_NAMES[arm], seed].correction(z, st))
            counts[:, k, j] = score.confusion(labels)
            flips[:, k-2, j] = score.flips(fixed, labels)
        merged_prediction = labels  # Last factorial cell has both sets of source parameters.
        original = from_delta(ev.legacy.prediction("dual_free", fits["dual_core", seed], z, original_stencil))
        counts[:, 6, j] = score.confusion(original)
        flips[:, 4, j] = score.flips(fixed, original)
        rounding[:, j] = score.flips(merged_prediction, original).sum(-1)
    trace = counts.diagonal(axis1=-2, axis2=-1).sum(-1)
    net_gains = trace[:, [2, 3, 4, 5, 6]] - trace[:, [1, 2, 2, 2, 2]]
    np.testing.assert_array_equal(flips[..., 0] - flips[..., 1], net_gains)
    assert np.all(counts.sum(-1) == counts[:, :1, :1].sum(-1))
    assert np.all(counts[1] <= counts[0]) and np.all(flips[1] <= flips[0])
    return counts, flips, rounding


def save_pack(path, arrays, ids, groups, done, lock):
    part = path.with_suffix(".tmp")
    with part.open("wb") as f:
        np.savez_compressed(f, **arrays, ids=ids, groups=groups, seeds=SEEDS, arms=ARMS,
                            pairs=PAIR_NAMES, complete=done, protocol=lock)
    part.replace(path)


@torch.inference_mode()
def run():
    ev.setup.configure_cuda()
    assert torch.cuda.is_available()
    rows, lock = protocol()
    ids = [r["id"] for r in rows]
    groups = [r["group"] for r in rows]
    fits = ev.original.load()
    nets, settings = factors.decoders(fits)
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "source_parameters": settings,
               "protocol": lock}
    write_json(HERE / "run.json", runtime)
    timer = monotonic()
    for target in TARGETS:
        path = HERE / f"counts-{target}.npz"
        n = len(rows)
        if path.exists():
            with np.load(path, allow_pickle=False) as p:
                assert p["protocol"].item() == lock and p["ids"].tolist() == ids
                assert p["groups"].tolist() == groups and p["seeds"].tolist() == list(SEEDS)
                arrays = {k: p[k] for k in ("counts", "flips", "rounding")}
                start = int(p["complete"])
        else:
            arrays = {"counts": np.zeros((n, 2, 7, 3, 19, 19), dtype=np.int32),
                      "flips": np.zeros((n, 2, 5, 3, 3), dtype=np.int64),
                      "rounding": np.zeros((n, 2, 3), dtype=np.int64)}
            start = 0
        if start == n:
            continue
        model = ev.Target(target)
        with np.load(PREVIOUS / f"counts-{target}.npz", allow_pickle=False) as p:
            assert p["ids"].tolist() == ids and p["groups"].tolist() == groups and int(p["finished"]) == n
            assert p["arms"].tolist() == ["released", "bilinear_grid", "fixed_four", "paper_fit"]
            assert p["seeds"].tolist() == list(SEEDS)
            archived = p["counts"][:, 0]
        for i in range(start, n):
            counts, flips, rounding = one_frame(model, fits, nets, rows[i])
            np.testing.assert_array_equal(counts[:, [0, 1, 2, 6]], archived[i], err_msg=f"{target} {ids[i]}")
            arrays["counts"][i], arrays["flips"][i], arrays["rounding"][i] = counts, flips, rounding
            if i == start or (i+1) % 16 == 0 or i+1 == n:
                save_pack(path, arrays, ids, groups, i + 1, lock)
                print(f"{target} {i+1}/{n}, elapsed {monotonic()-timer:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()

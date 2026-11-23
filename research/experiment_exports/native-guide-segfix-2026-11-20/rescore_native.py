"""Recompute native mIoUs and paired cluster intervals from integer matrices. NumPy only."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = ("mask2former", "segformer")
METRICS = ("full", "boundary")
CLASSES = {"bdd": list(range(19)), "fog": [k for k in range(19) if k not in (9, 16)]}
COMPARISONS = {
    "fixed_native": ("fixed_lowpass", "paper_512", "bilinear", "released", "segfix", "jbu_native",
                     "matched_gaussian_fixed", "matched_gaussian_fit"),
    "class_gap": ("fixed_native", "class_gap_lowpass", "scalar", "rgb_gap", "paper_512", "released",
                  "segfix", "jbu_native", "matched_gaussian_fixed", "matched_gaussian_fit"),
    "segfix": ("released", "bilinear"),
    "paper_512": ("released", "segfix", "jbu_native", "matched_gaussian_fixed", "matched_gaussian_fit"),
    "matched_gaussian_fit": ("matched_gaussian_fixed", "paper_512", "released", "segfix"),
    "matched_gaussian_fixed": ("paper_512", "released", "segfix"),
    "jbu_native": ("paper_512", "released", "segfix"),
}

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def get_groups(rows, cohort, receipt, packet_root=HERE):
    by_group = {}
    directory = packet_root / "counts" / cohort
    paths = sorted(directory.glob("*.npz"))
    assert len(paths) == len(rows) == {"bdd": 128, "fog": 62}[cohort]
    arms = receipt["arms"]
    for path in paths:
        with np.load(path, allow_pickle=False) as packet:
            key = packet["id"].item()
            group = packet["group"].item()
            assert key in rows and rows.pop(key)["group"] == group
            assert packet["lock"].item() == receipt["lock"]
            assert packet["targets"].tolist() == list(TARGETS)
            assert packet["seeds"].tolist() == [101, 211, 307] and packet["arms"].tolist() == arms
            count = packet["counts"]
            assert count.shape == (2, 2, len(arms), 3, 19, 19) and count.dtype == np.int32
            assert (count >= 0).all() and (count[:, 1] <= count[:, 0]).all()
            support = count.sum(-1)
            assert (support == support[:1, :, :1, :1]).all()
            assert count[0, 1, 0, 0].sum() == packet["edge_pixels"].item()
            hit = count.diagonal(axis1=-2, axis2=-1)
            union = count.sum(-2) + count.sum(-1) - hit
            suff = np.stack([hit[..., CLASSES[cohort]], union[..., CLASSES[cohort]]], -1)
            if group not in by_group:
                by_group[group] = np.zeros_like(suff, dtype=np.int64)
            by_group[group] += suff
    assert not rows and len(by_group) == {"bdd": 128, "fog": 16}[cohort]
    return np.stack([by_group[g] for g in sorted(by_group)])

def analyze(groups, arms, comparisons=COMPARISONS):
    n = len(groups)
    packed = groups.reshape(n, -1).astype(np.float64)
    def summarize(weight):
        pools = (weight @ packed).reshape(len(weight), *groups.shape[1:])
        hit, union = pools[..., 0], pools[..., 1]
        iou = np.divide(hit, union, out=np.zeros_like(hit), where=union > 0)
        return 100 * iou.mean(axis=-1).mean(axis=-1)
    point = summarize(np.ones((1, n)))[0]
    leave_one = summarize(np.ones((n, n)) - np.eye(n))
    draws = np.random.default_rng(10211).multinomial(n, np.full(n, 1 / n), size=5000)
    boot = np.concatenate([summarize(w) for w in np.array_split(draws, 100)])
    out = {}
    for t, target in enumerate(TARGETS):
        scores = {a: dict(zip(METRICS, point[t, :, i].tolist())) for i, a in enumerate(arms)}
        contrast = {}
        for a, others in comparisons.items():
            i = arms.index(a)
            for b in others:
                j = arms.index(b)
                contrast[a + " minus " + b] = delta = {}
                for k, metric in enumerate(METRICS):
                    sample = boot[:, t, k, i] - boot[:, t, k, j]
                    remove = leave_one[:, t, k, i] - leave_one[:, t, k, j]
                    delta[metric] = dict(pp=float(point[t, k, i] - point[t, k, j]),
                                         ci95=np.quantile(sample, [.025, .975]).tolist(),
                                         leave_one_out=[float(remove.min()), float(remove.max())])
        out[target] = dict(scores=scores, contrasts=contrast)
    return out

def main():
    receipt = read(HERE / "evaluation.json")
    manifest = read(HERE / "manifest.json")
    assert manifest["selection"] == digest(HERE / "selection.json") and receipt["files"] == 190
    rows = defaultdict(dict)
    for row in manifest["rows"]:
        assert row["id"] not in rows[row["cohort"]]
        rows[row["cohort"]][row["id"]] = row
    allfits = []
    for name in ("training.json", "gaussian-training.json"):
        allfits += read(HERE / name)["records"]
    assert len({(r["mode"], r["seed"]) for r in allfits}) == len(allfits) == 12
    for fit in allfits:
        assert fit["steps"] == len(fit["loss"]) == 320
        assert np.isfinite(fit["loss"]).all() and digest(HERE / fit["weights"]) == fit["sha256"]
    result = dict(confusion_lock=receipt["lock"], frames=receipt["files"], fits=len(allfits),
                  resampling="5000 paired percentile intervals, seed 10211, fixed classes and weights",
                  scores_in_percent=True, contrasts_in_percentage_points=True, cohorts={})
    for cohort in CLASSES:
        frames = len(rows[cohort])
        groups = get_groups(rows[cohort], cohort, receipt)
        result["cohorts"][cohort] = dict(frames=frames, groups=len(groups), classes=CLASSES[cohort],
                                         targets=analyze(groups, receipt["arms"]))
    dest = HERE / "native-results.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified", result["frames"], "images and", result["fits"], "three-seed fits.", dest.name)

if __name__ == "__main__":
    main()

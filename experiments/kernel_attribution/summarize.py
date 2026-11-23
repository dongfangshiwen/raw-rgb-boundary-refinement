"""Rebuild all intervals and the pixel-change diagnostic from the saved integer counts (CPU)."""
from boundary_refinement.paths import WORKSPACE
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = (WORKSPACE / "experiment_exports" / "kernel-attribution-2026-11-23")
PREVIOUS = HERE.parent / "decoder-resolution-2026-11-21"
ARMS = ("study_reference", "no_correction", "fixed", "colour_only", "spatial_only", "both", "paper")
REGIONS = ("whole", "boundary", "outside")
PAIRS = ("fixed_minus_no_correction", "colour_only_minus_fixed", "spatial_only_minus_fixed",
         "both_minus_fixed", "paper_minus_fixed")
CONTRASTS = {
    "fixed_minus_no_correction": {"fixed": 1, "no_correction": -1},
    "fixed_minus_study_reference": {"fixed": 1, "study_reference": -1},
    "colour_only_minus_fixed": {"colour_only": 1, "fixed": -1},
    "spatial_only_minus_fixed": {"spatial_only": 1, "fixed": -1},
    "both_minus_fixed": {"both": 1, "fixed": -1},
    "both_minus_spatial_only": {"both": 1, "spatial_only": -1},
    "both_minus_colour_only": {"both": 1, "colour_only": -1},
    "factor_interaction": {"both": 1, "spatial_only": -1, "colour_only": -1, "fixed": 1},
    "paper_minus_fixed": {"paper": 1, "fixed": -1},
    "paper_minus_study_reference": {"paper": 1, "study_reference": -1},
}
TARGETS = ("mask2former", "segformer")


def read(path):
    return json.loads(path.read_text("utf8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def extended(values):
    # Input is already aggregated by drive; counts have 64-bit precision.
    return np.concatenate((values, values[:, 0:1]-values[:, 1:2]), axis=1)


def group_sum(values, groups, names):
    return np.stack([values[groups == g].sum(axis=0, dtype=np.int64) for g in names])


def bootstrap(raw, classes, weights):
    # [drive, region, arm, seed, truth, pred] -> pooled macro IoU in percent.
    hits = raw.diagonal(axis1=-1, axis2=-2)
    union = raw.sum(-1)+raw.sum(-2)-hits
    sufficient = np.stack((hits[..., classes], union[..., classes]), axis=-1)
    mat = sufficient.reshape(len(raw), -1).astype(np.float64)
    def scores(w):
        sums = (w @ mat).reshape(len(w), *sufficient.shape[1:])
        h, u = sums[..., 0], sums[..., 1]
        return 100*np.divide(h, u, out=np.zeros_like(h), where=u>0).mean(axis=-1).mean(axis=-1)
    point = scores(np.ones((1, len(raw))))[0]
    reps = np.concatenate([scores(batch) for batch in np.array_split(weights, 50)])
    return point, reps


def ci(point, replicates):
    q = np.quantile(replicates, (0.025, 0.975))
    return {"pp": float(point), "ci95": [float(q[0]), float(q[1])]}


def dot_contrast(scores, coeff):
    return sum(value*scores[..., ARMS.index(name)] for name, value in coeff.items())


def earlier_samples():
    old = read(PREVIOUS/"protocol.json")["frames"]
    old_ids, old_groups = {r["id"] for r in old}, {r["group"] for r in old}
    other_ids, other_groups = set(), set()
    for f in (HERE.parent/"fixed-range-audit-v106"/"counts"/"idd").glob("*.npz"):
        with np.load(f, allow_pickle=False) as packet:
            other_ids.add(packet["id"].item())
            other_groups.add(packet["group"].item())
    assert len(other_ids) == 62 and len(other_groups) == 16
    return {"earlier_factorial_frames": len(other_ids), "earlier_factorial_drives": len(other_groups),
            "shared_frames": len(other_ids & old_ids), "shared_drives": len(other_groups & old_groups)}


def main():
    plan = read(HERE/"protocol.json")
    lock = digest(HERE/"protocol.json")
    historical = read(PREVIOUS/"results.json")
    targets, condition_effects = {}, {}
    group_order = None
    for name in TARGETS:
        path = HERE / f"counts-{name}.npz"
        with np.load(path, allow_pickle=False) as p, np.load(PREVIOUS/path.name, allow_pickle=False) as old:
            assert p["protocol"].item() == lock and p["complete"].item() == old["finished"].item() == 146
            for k in ("ids", "groups", "seeds"):
                np.testing.assert_array_equal(p[k], old[k])
            assert list(p["arms"]) == list(ARMS) and list(p["pairs"]) == list(PAIRS)
            groups = p["groups"]
            ordered = np.unique(groups)
            assert len(ordered)==24
            if group_order is None:
                group_order = ordered
            np.testing.assert_array_equal(group_order, ordered)
            counts = p["counts"]
            old_counts = old["counts"]
            np.testing.assert_array_equal(counts[:, :, [0, 1, 2, 6]], old_counts[:, 0])
            grouped = extended(group_sum(counts, groups, ordered))
            events = extended(group_sum(p["flips"], groups, ordered)).sum(0).mean(axis=2)
            rounding = p["rounding"].sum(0)
        # Reuse exactly the earlier resolution audit's paired 24-drive draws for all arms and targets.
        weights = np.random.default_rng(11221).multinomial(24, np.full(24, 1/24), size=5000)
        point, reps = bootstrap(grouped, plan["classes"], weights)
        valid_pixels = grouped.sum(0)[:, 0, 0].sum((-1, -2))
        by_region = {}
        for i, reg in enumerate(REGIONS):
            contrasts = {k: ci(dot_contrast(point[i], cf), dot_contrast(reps[:, i], cf))
                         for k, cf in CONTRASTS.items()}
            change_counts = {}
            for j, pair in enumerate(PAIRS):
                rescued, harmed, wrong = events[i, j]
                change_counts[pair] = {
                    "corrected_mean": float(rescued), "spoiled_mean": float(harmed),
                    "wrong_to_other_wrong_mean": float(wrong),
                    "valid_labels_changed_pct": float(100*(rescued+harmed+wrong)/valid_pixels[i]),
                    "pixel_accuracy_delta_pp": float(100*(rescued-harmed)/valid_pixels[i]),
                }
            by_region[reg] = {"valid_pixels": int(valid_pixels[i]), "miou_percent": dict(zip(ARMS, point[i].tolist())),
                              "contrasts": contrasts, "pixel_changes": change_counts}
        old512 = historical["targets"][name]["scales"]["512"]
        for i, (now, old_reg) in enumerate((("whole","full"), ("boundary","boundary"))):
            for k, k0 in (("fixed","fixed_four"), ("paper","paper_fit"), ("study_reference","released"),
                          ("no_correction","bilinear_grid")):
                np.testing.assert_allclose(point[i, ARMS.index(k)], old512[old_reg]["miou"][k0], rtol=0, atol=1e-10)
            reference = old512[old_reg]["contrasts"]["paper_minus_fixed_four"]
            current = by_region[now]["contrasts"]["paper_minus_fixed"]
            np.testing.assert_allclose([current["pp"],*current["ci95"]], [reference["pp"],*reference["ci95"]],
                                       rtol=0, atol=1e-10)
        targets[name] = {"sha256": digest(path), "by_region": by_region, "rounding_disagreements_by_seed":
                         {"whole": rounding[0].tolist(), "boundary": rounding[1].tolist()},
                         "boundary_share_of_valid_pct": float(100*valid_pixels[1]/valid_pixels[0]),
                         "prior_1024_paper_minus_fixed_boundary": historical["targets"][name]["scales"]["1024"]["boundary"]["contrasts"]["paper_minus_fixed_four"]}
        condition_effects[name] = (point, reps)
    effects = {}
    for i, region in enumerate(REGIONS):
        effects[region] = {}
        for contrast in ("colour_only_minus_fixed","spatial_only_minus_fixed","both_minus_fixed","paper_minus_fixed"):
            coefficients = CONTRASTS[contrast]
            pm, rm = condition_effects["mask2former"]
            ps, rs = condition_effects["segformer"]
            effects[region][contrast] = ci(dot_contrast(ps[i]-pm[i], coefficients),
                                           dot_contrast(rs[:, i]-rm[:, i], coefficients))
    result = {"protocol_sha256": lock, "cohort": {"images": 146, "drives": 24, **earlier_samples()},
              "bootstrap": {"paired_drive_draws": 5000, "seed": 11221, "interval": "pointwise percentile 95%"},
              "primary_input_long_edge": 512, "targets": targets, "segformer_minus_mask2former_effect": effects}
    path = HERE / "results.json"
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if path.exists():
        assert read(path) == result, "Recomputation differs from previous output"
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf8")
        tmp.replace(path)
    for n, t in targets.items():
        print(n)
        for metric in ("whole", "boundary", "outside"):
            d = t["by_region"][metric]
            print(metric, " ".join(f"{k}={d['contrasts'][k]['pp']:+.3f}" for k in
                                   ("fixed_minus_no_correction", "colour_only_minus_fixed",
                                    "spatial_only_minus_fixed", "both_minus_fixed")))
    print("Reproduced previously archived masks' confusion matrices and paired summary endpoints.")


if __name__ == "__main__":
    main()

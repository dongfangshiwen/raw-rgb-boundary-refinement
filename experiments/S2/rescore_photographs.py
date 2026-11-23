"""Independently check the six photographic comparisons using CPU NumPy."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("input", "paper_curve", "public_mct", "public_pgl", "scalar", "conditional")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def run():
    rows = load("photo-individual.json")
    published = load("photo-results.json")["sets"]
    for dataset, expected_images, expected_groups in (("bedde", 208, 23), ("revide_final", 16, 8)):
        frames = defaultdict(dict)
        for row in rows:
            if row["ds"] == dataset:
                frame = frames[(row["group"], row["key"])]
                assert row["arm"] not in frame
                frame[row["arm"]] = (row["psnr"], row["deltaE"])
        assert len(frames) == expected_images and all(set(f) == set(METHODS) for f in frames.values())
        groups = sorted({g for g, _ in frames})
        assert len(groups) == expected_groups
        group_means = np.array([
            np.mean([[measurements[m] for m in METHODS]
                     for (g, _), measurements in frames.items() if g == group], axis=0)
            for group in groups
        ])
        means = group_means.mean(axis=0)
        rng = np.random.default_rng(100481)
        draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=5000)
        boot = np.einsum("bg,gmk->bmk", draws / len(groups), group_means)
        saved = published[dataset]
        assert saved["images"] == len(frames) and saved["groups"] == len(groups)
        for j, method in enumerate(METHODS):
            np.testing.assert_allclose(means[j], saved["scores"][method], atol=1e-9, rtol=0)
        for label, archived in saved["contrasts"].items():
            a, b = (METHODS.index(name) for name in label.split(" minus "))
            np.testing.assert_allclose(means[a] - means[b], archived["estimate"], atol=1e-9, rtol=0)
            ci = np.quantile(boot[:, a] - boot[:, b], [.025, .975], axis=0).T
            np.testing.assert_allclose(ci, archived["ci95"], atol=1e-9, rtol=0)
        score = means[METHODS.index("scalar"), 0]
        print(f"{dataset}: {len(frames)} photographs, {len(groups)} groups, scalar PSNR {score:.3f} dB")


if __name__ == "__main__":
    run()

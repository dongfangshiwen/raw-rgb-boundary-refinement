"""Measure semantic false positives at matched recall on the original label grids."""
import argparse
from collections import Counter
import re

import numpy as np
import torch

from dehaze_seg.metrics import confusion_matrix
from scripts import false_positive_constraint_v87 as projection
from scripts.contrastive_response_v85 import raw_anchor
from scripts.cross_scale_rgb_v44 import resize
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import aggregate
from scripts.resolution_interaction_v84 import DualScaleFeatures, SEEDS, add_global_cohort
from scripts.restoration_affinity_v75 import compress_scores
from scripts.sparse_recovery_v86 import ROOT as SPARSE, edit, images, load as sparse_weights, raw_models
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


ROOT = REPO / "results/matched-recall-v88"
CLASSES = (6, 7, 14)
RECALL = (.80, .85, .90, .95)
SCALE = 64
LIMIT = 32
BINS = 2 * LIMIT * SCALE + 2
LEGACY = ("author_raw", "raw_two_scale", "raw_sparse", "before_adaptation", *projection.ARMS)
PRIMARY = projection.PRIMARY
BOOTSTRAP = 2000


def bucket(score):
    return (score.mul(SCALE).floor().long() + LIMIT * SCALE + 1).clamp_(0, BINS - 1)


def pixel_histograms(logits, truth):
    mask = truth != 255
    label = truth[mask].long()
    scores, indices = logits[0].topk(2, dim=0)
    best, runner_up = scores[:, mask].unbind()
    winner = indices[0, mask]
    output = []
    for cls in CLASSES:
        margin = logits[0, cls, mask] - torch.where(winner == cls, runner_up, best)
        order = bucket(margin)
        histogram = torch.bincount(order + (label == cls).long() * BINS, minlength=2 * BINS)
        output.append(histogram.reshape(2, BINS))
    return torch.stack(output).cpu().numpy()


def save_histograms(target, populations, methods):
    data = np.stack([v for v in populations.values()])
    file = target.with_suffix(".part.npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(file, names=np.asarray(list(populations)), methods=np.asarray(methods), counts=data)
        file.replace(target)
    finally:
        file.unlink(missing_ok=True)


def curve_at_recall(hist, samples, recall):
    """Solve each monotone positive CDF and interpolate within the selected bin."""
    hist = hist[..., ::-1]
    neg, pos = np.cumsum(hist, axis=-1, dtype=np.int64).transpose(1, 0, 2)
    total_pos = samples @ pos[:, -1]
    total_neg = samples @ neg[:, -1]
    valid = (total_pos > 0) & (total_neg > 0)
    target = total_pos[:, None] * np.asarray(recall)
    left, right = np.zeros_like(target, dtype=np.int32), np.full(target.shape, BINS - 1, dtype=np.int32)

    def at(cumulative, bins):
        values = cumulative[:, bins]
        return np.einsum("gd,dgk->gk", samples, values)

    for _ in range((BINS - 1).bit_length()):
        middle = (left + right) // 2
        high = at(pos, middle) >= target
        left = np.where(high, left, middle + 1)
        right = np.where(high, middle, right)
    prior = np.maximum(left - 1, 0)
    before_pos = np.where(left > 0, at(pos, prior), 0)
    before_neg = np.where(left > 0, at(neg, prior), 0)
    local_pos = at(pos, left) - before_pos
    local_neg = at(neg, left) - before_neg
    fraction = np.divide(target - before_pos, np.maximum(local_pos, 1)).clip(0., 1.)
    false_rate = 1e6 * (before_neg + fraction * local_neg) / np.maximum(total_neg[:, None], 1)
    false_rate[~valid] = np.nan
    return false_rate, left


def table_for(counts, methods, part, group_names, comparisons):
    # The same group draws are reused across methods and across optimization seeds.
    rng = np.random.default_rng(88033)
    groups = counts.shape[0]
    samples = np.ones((1, groups), dtype=np.int64)
    if groups > 1:
        samples = np.vstack((samples, rng.multinomial(groups, np.full(groups, 1. / groups), size=BOOTSTRAP)))
    out = {"groups": group_names, "classes": {}}
    for i, cls in enumerate(CLASSES):
        support = counts[:, 0, 0, i].sum(-1)
        if not support.sum(axis=0).min():
            out["classes"][str(cls)] = dict(positive=int(support[:, 1].sum()), negative=int(support[:, 0].sum()))
            continue
        results, computed = {}, {}
        for j, method in enumerate(methods):
            runs = [curve_at_recall(counts[:, j, s, i], samples, RECALL) for s in range(len(SEEDS))]
            estimates = np.stack([v for v, _ in runs])
            curve = estimates.mean(axis=0)
            if any(np.any((b[0] == 0) | (b[0] == BINS - 1)) for _, b in runs):
                raise ValueError(f"The target recall for {part}/{method}/class {cls} landed in an overflow bin")
            computed[method] = curve
            results[method] = dict(per_million=curve[0].tolist(), each_seed=estimates[:, 0].tolist())
        intervals = {}
        for term, control in comparisons:
            delta = computed[term] - computed[control]
            draws = delta[1:]
            valid = np.isfinite(draws).all(axis=1)
            interval = np.percentile(draws[valid], (2.5, 97.5), axis=0).T.tolist() if valid.any() else None
            intervals[f"{term}-vs-{control}"] = dict(diff=delta[0].tolist(), ci95=interval, valid_draws=int(valid.sum()))
        out["classes"][str(cls)] = dict(positive=int(support[:, 1].sum()), negative=int(support[:, 0].sum()),
                                         positive_groups=int((support[:, 1] > 0).sum()), methods=results, effects=intervals)
    return out


def capture_id(dataset, group):
    if dataset == "foggy_zurich_testv2" and re.fullmatch(r"(?:GOPR|GP[0-9]{2})[0-9]{4}", group):
        # A GOPR file and GP01/GP02/... files with the same suffix are chapters of one video.
        return f"GoPro-{group[-4:]}"
    return group


def capture_banks(names, counts):
    scopes = {}
    for (dataset, file_group), values in zip(names, counts, strict=True):
        group = capture_id(dataset, file_group)
        cohorts = (dataset, "foggy_driving_101") if dataset.startswith("foggy_driving_") else (dataset,)
        for cohort in cohorts:
            bank = scopes.setdefault(cohort, {})
            bank.setdefault(group, np.zeros_like(values))
            bank[group] += values
    return scopes


def summarize(histogram, comparisons):
    with np.load(histogram, allow_pickle=False) as archive:
        names = archive["names"].tolist()
        methods = archive["methods"].tolist()
        counts = archive["counts"]
    return {dataset: table_for(np.stack(list(bank.values())), methods, dataset, list(bank), comparisons)
            for dataset, bank in capture_banks(names, counts).items()}


@torch.inference_mode()
def score(part, variant):
    configure_cuda()
    methods, nets = LEGACY, {}
    comparisons = [(PRIMARY, a) for a in ("total_loss_projection", "plain_sparse", "dense_only", "raw_sparse", "raw_two_scale")]
    trials = {}
    if variant in ("ranking", "recall"):
        from scripts import ranked_negatives_v89 as ranking

        nets.update(ranking.load())
        trials["ranking"] = digest(ranking.ROOT / "completed.json")
        methods += ranking.ARMS
        comparisons += [(ranking.PRIMARY, a) for a in (*ranking.ARMS, *projection.ARMS, "raw_two_scale")
                        if a != ranking.PRIMARY]
    if variant == "recall":
        from scripts import recall_constraint_v91 as constraint

        nets.update(constraint.load())
        trials["recall"] = digest(constraint.ROOT / "completed.json")
        methods += constraint.ARMS
        comparisons += [(constraint.PRIMARY, a) for a in (*constraint.ARMS, *projection.ARMS, "rank_margin",
                        "raw_two_scale", "before_adaptation") if a != constraint.PRIMARY]
    if part == "rain":
        from scripts.stage_muses_rain_v90 import MANIFEST as reference, cases

        iterator = cases()
    else:
        reference = projection.ROOT / f"{part}-test.json"
        iterator = images(part)
    ref, anchors, prior = projection.load(), raw_models(), sparse_weights()
    engine = DualScaleFeatures()
    seen, hist, records = Counter(), {}, []
    for dataset, key, group, _, rgb, truth in iterator:
        features, _, author = engine(rgb)
        for seed_idx, seed in enumerate(SEEDS):
            start = raw_anchor(anchors["raw_detail", seed], features)
            logits = {"author_raw": author, "raw_two_scale": start,
                      "raw_sparse": start + edit(prior["raw_mixed", seed], features, "raw_mixed"),
                      "before_adaptation": start + edit(prior["fusion_dense", seed], features, "fusion_mixed")}
            logits.update({arm: start + edit(ref[arm, seed], features, "fusion_mixed") for arm in projection.ARMS})
            logits.update({arm: start + edit(net, features, "fusion_mixed") for (arm, s), net in nets.items() if s == seed})
            keygroup = dataset, group
            if keygroup not in hist:
                hist[keygroup] = np.zeros((len(methods), len(SEEDS), len(CLASSES), 2, BINS), np.uint32)
            for idx, arm in enumerate(methods):
                native = resize(logits[arm], truth.shape)
                hist[keygroup][idx, seed_idx] += pixel_histograms(native, truth).astype(np.uint32)
                record = confusion_matrix(native.argmax(1)[0], truth, 19)
                records.append(dict(dataset=dataset, id=key, group=group, seed=seed, method=arm, counts=record.tolist()))
        seen[dataset] += 1
        if sum(seen.values()) % 24 == 0:
            print(dict(scored=part, variant=variant, frames=sum(seen.values())), flush=True)
    if dict(seen) != read(reference)["counts"]:
        raise ValueError(f"Scored population changed: {seen}")
    scope = ROOT / variant
    histogram, matrix = (scope / f"{part}-{ext}" for ext in ("histograms.npz", "individual.json.gz"))
    save_histograms(histogram, hist, methods)
    compress_scores(matrix, records)
    scores = summarize(histogram, comparisons)
    merged = add_global_cohort(records)
    results = aggregate(merged)
    atomic_write(scope / f"{part}-test.json",
                 dict(population=dict(seen), methods=list(methods), recall=list(RECALL), primary_recall=.85,
                      bin_width=1. / SCALE, histogram_SHA256=digest(histogram), rows_SHA256=digest(matrix),
                      models_SHA256=digest(projection.ROOT / "completed.json"),
                      challenger_SHA256=trials,
                      reference_SHA256=digest(reference) if reference.is_file() else None,
                      scorer_SHA256=digest(__file__), mIoU=results, matched_recall=scores,
                      interpretation="Descriptive ROC interpolation after scoring. Matched recall within each dense test "
                                     "cohort does not select a deployed threshold or update any model. Bootstrap resamples "
                                     "registered groups and averages the three fixed optimization seeds. Unknown pixels "
                                     "are excluded. Positive-only labels do not cover all negative categories.",
                      utc=now()))
    for name, table in scores.items():
        light = table["classes"]["6"]
        if "methods" in light:
            print(name, {m: round(x["per_million"][1], 2) for m, x in light["methods"].items()}, flush=True)


def check_math():
    values = np.zeros((2, 2, BINS), dtype=np.int64)
    values[0, 1, 202:205] = (20, 60, 20)
    values[0, 0, 200:205] = (40, 40, 15, 5, 0)
    values[1] = np.roll(values[0], 170, axis=-1)
    samples = np.array([[1, 0], [0, 1]], dtype=np.int64)
    area, chosen = curve_at_recall(values, samples, (.20, .50, .80))
    np.testing.assert_allclose(area[0], [0, 25000, 50000])
    np.testing.assert_array_equal(area[0], area[1])
    assert np.all(chosen[0] - chosen[1] == 170)
    logits = torch.zeros((1, 19, 1, 4))
    logits[:, 6] = torch.tensor([[2., -1., 1., 8.]])
    truth = torch.tensor([[6, 7, 0, 255]])
    hist = pixel_histograms(logits, truth)
    assert hist[0].sum() == 3 and hist[0, 1, bucket(torch.tensor(2.)).item()] == 1
    assert capture_id("foggy_zurich_testv2", "GOPR0229") == capture_id("foggy_zurich_testv2", "GP010229")
    print("Matched recall, constant offset invariance, and ignore masks passed.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("part", choices=("check", "external", "held", "rain"))
    cli.add_argument("--variant", choices=("legacy", "ranking", "recall"), default="legacy")
    args = cli.parse_args()
    if args.part == "check":
        check_math()
    else:
        score(args.part, args.variant)

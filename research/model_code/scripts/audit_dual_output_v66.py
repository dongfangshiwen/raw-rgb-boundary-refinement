"""Measure both tasks on paired road scenes and evaluate the official LiDSeg release."""
import argparse
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from statistics import mean

import numpy as np
from PIL import Image
import torch

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from scripts.coarse_semantics_v39 import blend, native_labels, paint
from scripts.edge_projection_v47 import draw
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, load_pairs, resize_image
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_shared_haze_field_v40 import quality, to_rgb
from scripts.factorial_restoration_v65 import Restoration, predictions, trained
from scripts.kitti_fog_v40 import MANIFEST, load as kitti
from scripts.parallel_utility_v32 import capture_group
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.tonal_agreement_v66 import COHORTS, ROOT, SEEDS, rendered_samples
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


LIDSEG = REPO.parent / "benchmark_data/lidseg-author-v66"
AUTHOR_WEIGHTS = LIDSEG / "lidseg_pidnetl_uber.pth"
AUTHOR_HASH = "62b6fc6c75f66884efb9cc01391ab21016018622def1c8598132bcd355d2c1c7"
AUTHOR_RELEASE = "https://github.com/yobsanb/LiDSeg/releases/tag/pidnetl-acdc"
KITTI_ARMS = ("mcaf", "utility", "tone", "utility_tone", "direct_tone", "published_mct")


@torch.inference_mode()
def paired():
    configure_cuda()
    restorer, bank = Restoration(), trained()
    rows, counts, groups = [], Counter(), defaultdict(lambda: defaultdict(list))
    for row in kitti():
        if row["role"] != "held":
            continue
        clear = row["rgb"][None].cuda().float() / 255.
        views, _ = rendered_samples(row)
        for data, fog in zip(("kitti_light", "kitti_dense"), views, strict=True):
            item = restorer(fog)
            counts[data] += 1
            for seed in SEEDS:
                images = predictions(item, bank, seed)
                values = {tag: quality(image, clear) for tag, image in {"raw": fog, **images}.items()}
                for tag, metric in values.items():
                    rows.append(dict(dataset=data, id=row["id"], group=row["drive"], method=tag, seed=seed, **metric))
                for tag in KITTI_ARMS:
                    groups[data, tag][row["drive"]].append(values[tag]["psnr"] - values["mcaf"]["psnr"])
    if counts != {key: value for key, value in COHORTS.items() if key.startswith("kitti_")}:
        raise ValueError("Incomplete paired KITTI trial")
    recorded = ROOT / "paired_kitti_individual.json"
    atomic_write(recorded, rows)
    metrics = defaultdict(list)
    for row in rows:
        metrics[row["dataset"], row["method"]].append(row["psnr"])
    result = {"/".join(key): mean(items) for key, items in metrics.items()}
    intervals = {"/".join(key): bootstrap({g: mean(items) for g, items in values.items()}) for key, values in groups.items()}
    atomic_write(ROOT / "paired_kitti.json", dict(counts=dict(counts), psnr=result, change_vs_mcaf=intervals,
                 kitti_SHA256=digest(MANIFEST), table_SHA256=digest(recorded), score="PSNR on the original clear image grid. "
                 "Same held KITTI identities and fog renders as 19-class semantic evaluation. The semantic adapter changes only the mask.",
                 finished_utc=now()))
    print({key: round(v, 3) for key, v in result.items()}, flush=True)


def author_loader():
    if digest(AUTHOR_WEIGHTS) != AUTHOR_HASH:
        raise ValueError("Unexpected official LiDSeg weight checksum")
    source = LIDSEG / "archive/LiDSeg-pidnetl-acdc/tools/infer.py"
    spec = spec_from_file_location("published_lidseg_infer", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_model(AUTHOR_WEIGHTS, torch.device("cuda")).requires_grad_(False), digest(source)


@torch.inference_mode()
def published():
    configure_cuda()
    author, model, source_hash = author_loader()
    rows, counts, panels = [], Counter(), []
    for dataset in ("foggy_zurich_testv2", "foggy_driving_fine"):
        for key, group, picture, truth in load_pairs(dataset):
            low, _ = resize_image(picture)
            native = torch.from_numpy(truth.astype(np.int64, copy=False)).cuda()
            for name, image in (("1024_long_side", to_rgb(low)), ("author_native", picture)):
                # The author loader may receive unusual aspect ratios. Replicate pad to the backbone stride and crop before scoring.
                h, w = image.shape[:2]
                pad = np.pad(image, ((0, (-h) % 32), (0, (-w) % 32), (0, 0)), mode="edge")
                result = model.predict(author.to_input(pad)[None].cuda()).float()[..., :h, :w]
                if not bool(result.isfinite().all()) or result.shape[-2:] != (h, w):
                    raise ValueError(f"Invalid author output {dataset}/{key}")
                label = native_labels(result, native.shape)
                cm = confusion_matrix(label, native, 19)
                rows.append(dict(dataset=dataset, id=key, group=group or capture_group(key), method=name, counts=cm.tolist()))
                if not counts[dataset] and name == "author_native":
                    panels.append((f"{dataset}_{key}", author.PALETTE[label.cpu().numpy()]))
            counts[dataset] += 1
        print(dict(dataset=dataset, samples=counts[dataset]), flush=True)
    if counts != {key: value for key, value in COHORTS.items() if key.startswith("foggy_")}:
        raise ValueError("The public-model evaluation is incomplete")
    sums = defaultdict(lambda: np.zeros((19, 19), dtype=np.int64))
    for row in rows:
        sums[row["dataset"], row["method"]] += np.asarray(row["counts"], dtype=np.int64)
    scores = {"/".join(key): 100. * segmentation_metrics(torch.from_numpy(value))["miou"] for key, value in sums.items()}
    dest = ROOT / "lidseg"
    individual = dest / "individual.json"
    atomic_write(individual, rows)
    for name, panel in panels:
        Image.fromarray(panel).save(dest / f"{name}.png")
    atomic_write(dest / "test.json", dict(miou=scores, counts=dict(counts), table_SHA256=digest(individual),
                 author=dict(release=AUTHOR_RELEASE, weights_SHA256=AUTHOR_HASH, source_SHA256=source_hash,
                             source_archive_SHA256=digest(LIDSEG / "tag-source.zip"), parameters=sum(p.numel() for p in model.parameters())),
                 data_SHA256={p.name: digest(p) for p in (ZURICH, DRIVING)},
                 design="Official ACDC-fine-tuned PIDNet-L model. Same evaluation masks, native scoring grids and Cityscapes IDs. "
                        "Long-1024 compares input size. Native variant uses the original RGB size. Training data and budget differ.", finished_utc=now()))
    print(scores, flush=True)


def paired_iou_intervals(rows, pairs, method_key, counts_key):
    scores = defaultdict(list)
    for row in rows:
        scores[row["dataset"], row["group"], row[method_key], row["seed"]].append(row[counts_key])
    intervals = {}
    for d in sorted({r["dataset"] for r in rows}):
        groups = sorted({r["group"] for r in rows if r["dataset"] == d})
        random = np.random.default_rng(66007)
        draws = random.multinomial(len(groups), np.full(len(groups), 1. / len(groups)), size=3000)
        draws = np.concatenate((np.ones((1, len(groups)), dtype=np.int64), draws))
        boot = {}
        for arm in sorted({mode for pair in pairs for mode in pair}):
            seeds = sorted({r["seed"] for r in rows if r["dataset"] == d and r[method_key] == arm})
            by_group = np.asarray([[np.sum(scores[d, group, arm, seed], axis=0, dtype=np.int64) for seed in seeds]
                                   for group in groups])
            corr = (draws @ by_group.reshape(len(groups), -1)).reshape(len(draws), len(seeds), 19, 19)
            hit = np.diagonal(corr, axis1=-2, axis2=-1)
            union = corr.sum(-2) + corr.sum(-1) - hit
            ratios = np.divide(hit, union, out=np.zeros_like(hit, dtype=float), where=union != 0)
            boot[arm] = 100. * ratios.mean(axis=(1, 2))
        for active, control in pairs:
            delta = boot[active] - boot[control]
            intervals[f"{d}/{active}-vs-{control}"] = dict(diff=float(delta[0]), ci_95=np.quantile(delta[1:], (.025, .975)).tolist(),
                                                           groups=len(groups))
    return intervals


@torch.inference_mode()
def external():
    from scripts.road_transition_v68 import muses_cases
    from scripts.stage_muses_v67 import MANIFEST as EXTERNAL_DATA
    from scripts.tonal_agreement_v66 import DualOutput, guidance, trained as semantic_weights

    configure_cuda()
    system, fitted = DualOutput(), semantic_weights()
    author, published_model, _ = author_loader()
    channel_mean, channel_std = (torch.from_numpy(v).cuda()[None, :, None, None] for v in (author.MEAN, author.STD))
    modes = ("raw_adapter", "concat", "agreement", "shifted")
    rows, seen, panels = [], Counter(), []
    for d, key, group, x, truth, image in muses_cases():
        item, _, restored = system(x)
        reference = published_model.predict((x - channel_mean) / channel_std).float()
        methods = {("raw_b1", 0): item.raw, ("lidseg", 0): reference}
        for name in modes:
            cue = guidance(item, name)
            for seed in SEEDS:
                methods[name, seed] = fitted[name, seed](item.spatial, cue, item.raw)
        labels = {name: native_labels(logits, truth.shape) for name, logits in methods.items()}
        for (name, seed), mask in labels.items():
            rows.append(dict(dataset=d, id=key, group=group, method=name, seed=seed,
                             counts=confusion_matrix(mask, truth, 19).tolist()))
        if not seen[group]:
            predictions_to_show = (labels["raw_b1", 0], labels["agreement", SEEDS[0]], labels["lidseg", 0])
            overlays = [blend(image, v.byte().cpu().numpy()) for v in predictions_to_show]
            panels.append((group + " / " + key, [image, to_rgb(restored), *overlays, paint(truth.byte().cpu().numpy())],
                           ("Real fog", "Restored RGB", "Raw B1", "Tone agreement", "Author LiDSeg", "Reference")))
        seen[group] += 1
    expected = read(EXTERNAL_DATA)["groups"]
    if seen != expected:
        raise ValueError("The held MUSES recording cohort changed")
    intervals = paired_iou_intervals(rows, [("agreement", control) for control in ("raw_b1", "raw_adapter", "concat", "shifted", "lidseg")],
                                     "method", "counts")
    values = defaultdict(list)
    for row in rows:
        values[row["method"], row["seed"]].append(row["counts"])
    by_method = defaultdict(dict)
    for (method, seed), group in values.items():
        by_method[method][str(seed)] = 100. * segmentation_metrics(torch.from_numpy(np.sum(group, axis=0, dtype=np.int64)))["miou"]
    target, picture = ROOT / "muses-individual.json", ROOT / "muses-predictions.png"
    atomic_write(target, rows)
    draw(picture, panels)
    metrics = {name: dict(miou=mean(v.values()), by_seed=v) for name, v in by_method.items()}
    atomic_write(ROOT / "muses-test.json", dict(groups=dict(seen), metrics=metrics, intervals=intervals,
                 dataset_SHA256=digest(EXTERNAL_DATA), table_SHA256=digest(target), picture_SHA256=digest(picture), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in metrics.items()}, flush=True)


def verify():
    from scripts.factorial_restoration_v65 import ROOT as FACTORIAL
    from scripts.road_transition_v68 import ROOT as ROAD, semantic_metrics
    from scripts.tonal_agreement_v66 import summarize, tone_intersection

    a = torch.tensor([3., -4., 2., 0., -3.])
    b = torch.tensor([2., -6., -6., 4., 1.])
    torch.testing.assert_close(tone_intersection(a, b), torch.tensor([2., -4., 0., 0., 0.]))
    note = read(ROOT / "test.json")
    table = ROOT / "individual.json"
    if digest(table) != note["table_SHA256"]:
        raise ValueError("The semantic scores were modified")
    rows = read(table)
    scores, ci = summarize(rows)
    if note["metrics"] != scores or note["contrasts"] != ci:
        raise ValueError("Pooled semantic metrics did not reproduce")
    paired_note = read(ROOT / "paired_kitti.json")
    rgb_table = ROOT / "paired_kitti_individual.json"
    if digest(rgb_table) != paired_note["table_SHA256"]:
        raise ValueError("The jointly labelled RGB predictions changed")
    by_method = defaultdict(list)
    for row in read(rgb_table):
        by_method[f'{row["dataset"]}/{row["method"]}'].append(row["psnr"])
    if {key: mean(values) for key, values in by_method.items()} != paired_note["psnr"]:
        raise ValueError("Paired KITTI metrics did not reproduce")
    corrected = paired_iou_intervals(rows, [("agreement", c) for c in ("raw_b1", "raw_adapter", "concat", "shifted")], "method", "counts")
    factorial = read(FACTORIAL / "test.json")
    factorial_rows = FACTORIAL / "individual.json"
    if digest(factorial_rows) != factorial["table_SHA256"]:
        raise ValueError("The factorial PSNR table changed")
    grouped = defaultdict(list)
    for row in read(factorial_rows):
        grouped[row["dataset"], row["arm"]].append(row["psnr"])
    if {"/".join(k): mean(v) for k, v in grouped.items()} != factorial["psnr"]:
        raise ValueError("The factorial PSNR could not be reproduced")
    road = {}
    for name in ("development", "muses"):
        summary = read(ROAD / f"{name}-test.json")
        individual = ROAD / f"{name}-individual.json"
        if digest(individual) != summary["table_SHA256"]:
            raise ValueError("Road transition evaluation rows changed")
        cases = read(individual)
        if semantic_metrics(cases) != summary["metrics"]:
            raise ValueError("Road transition mIoU could not be reproduced")
        comparisons = [("transition", c) for c in ("raw", "generic", "shifted", "utility_tone")]
        road[name] = paired_iou_intervals(cases, comparisons, "arm", "confusion")
    body = dict(convention="Keep the denominator at 19 classes in every pooled bootstrap replicate. "
                "Each capture group is resampled as a whole. This supersedes the interval convention in the V66 development test JSON.",
                agreement=corrected, transition=road)
    atomic_write(ROOT / "fixed19-intervals.json", body)
    result = dict(agreement_math=True, reproduced_miou=True, reproduced_psnr=True, road_scores_checked=True,
                  factorial_rows=factorial["rows"], semantic_rows=note["rows"], kitti_rgb_rows=sum(map(len, by_method.values())), finished_utc=now())
    atomic_write(ROOT / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("paired", "published", "external", "verify"))
    {"paired": paired, "published": published, "external": external, "verify": verify}[parser.parse_args().stage]()

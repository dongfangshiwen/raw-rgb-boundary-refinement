"""Evaluate fixed selectors on a whole held-out development video."""
import numpy as np
import torch

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.data.camvid_data import DATA
from boundary_refinement.data.provenance import digest
from boundary_refinement.data.regional_data import VIEWS, boundary_band, validation_view
from boundary_refinement.training.regional_guidance import (
    CLASSES, CONTROLS, apply_choice, choose_tiles, selector_features, tile_gain,
)


METRIC_NAMES = ("psnr", "boundary_psnr", "mse", "boundary_mse", "regret", "positive_tiles")


def psnr(error):
    return -10 * error.clamp_min(1e-10).log10()


def mean_over_pixels(error, mask):
    count = mask.sum((-2, -1))
    return (error * mask).sum((-2, -1)) / count.clamp_min(1)


def image_measures(restored, reference, band):
    error = (restored - reference).square().mean(dim=1)
    mse = error.mean((-2, -1))
    boundary_mse = mean_over_pixels(error, band)
    return mse, boundary_mse


@torch.inference_mode()
def evaluate(model, selectors, data, fold, seed, coverage):
    counts = torch.bincount(data["mask"][fold["indices"]["validation"]].flatten().long(), minlength=256)[:CLASSES]
    rows, summaries = [], {}
    max_identity_error = 0.0
    model.eval()
    for view in VIEWS:
        cm = torch.zeros(CLASSES, CLASSES, dtype=torch.int64)
        condition = []
        indices = fold["indices"]["validation"]
        for start in range(0, len(indices), 8):
            batch_indices = indices[start:start + 8]
            reference = data["raw"][batch_indices].cuda().float() / 255
            targets = data["mask"][batch_indices].cuda().long()
            band = boundary_band(targets)
            has_band = band.flatten(1).any(1).cpu().tolist()
            image = validation_view(reference, view)
            anchor, rgb, _, guided, logits, probabilities = model(image)
            cm += confusion_matrix(logits.argmax(dim=1), targets, CLASSES)
            features, confidence = selector_features(image, anchor, rgb, guided, probabilities)
            gain = tile_gain(rgb, guided, reference)
            oracle_tiles = choose_tiles(gain, coverage)
            oracle_value = (oracle_tiles * gain).flatten(1).mean(1)
            scores = {"entropy": confidence} | {key: block(features) for key, block in selectors.items()}
            scores["oracle"] = gain
            input_mse, _ = image_measures(image, reference, band)
            rgb_mse, _ = image_measures(rgb, reference, band)
            predictions = {"anchor": (anchor, None), "rgb": (rgb, torch.zeros_like(gain)),
                           "semantic": (guided, torch.ones_like(gain))}
            for name, score in scores.items():
                if not torch.isfinite(score).all():
                    raise ValueError("Nonfinite regional score")
                selected = choose_tiles(score, coverage)
                predictions[name] = (apply_choice(rgb, guided, selected), selected)
            for method, (output, selected) in predictions.items():
                mse, band_mse = image_measures(output, reference, band)
                regret = positive = fill = [None] * len(reference)
                if selected is not None:
                    value = (selected * gain).flatten(1).mean(1)
                    error = (rgb_mse - mse - value).abs().max().item()
                    max_identity_error = max(max_identity_error, error)
                    if error > 2e-7:
                        raise ValueError("Tile utility and actual restored-pixel error disagree")
                    n = selected.flatten(1).sum(1)
                    fill = selected.flatten(1).mean(1).tolist()
                    if method in scores:
                        expected_count = max(1, round(gain[0].numel() * coverage))
                        if not bool((n == expected_count).all()):
                            raise ValueError("Tile coverage differs across selectors")
                        if bool((value > oracle_value + 1e-7).any()):
                            raise ValueError("Fixed-budget oracle fell below a selector")
                        regret = (oracle_value - value).clamp_min(0).tolist()
                        positive = (((gain > 0) * selected).flatten(1).sum(1) / n).tolist()
                for i, (m, p, bm, bp, input_p, base_p) in enumerate(zip(
                    mse.tolist(), psnr(mse).tolist(), band_mse.tolist(), psnr(band_mse).tolist(),
                    psnr(input_mse).tolist(), psnr(rgb_mse).tolist(),
                )):
                    condition.append(dict(held_video=fold["held_video"], seed=seed, condition=view, method=method,
                                          image=data["ids"][batch_indices[i]], mse=m, psnr=p, input_psnr=input_p,
                                          delta_rgb_psnr=p - base_p, boundary_psnr=bp if has_band[i] else None,
                                          boundary_mse=bm if has_band[i] else None, coverage=fill[i],
                                          regret=regret[i], positive_tiles=positive[i]))
        if not torch.equal(cm.sum(1), counts):
            raise ValueError("Validation confusion matrix and source label support disagree")
        methods = {}
        for name in (*CONTROLS, "oracle"):
            entries = [row for row in condition if row["method"] == name]
            result = {"images": len(entries)}
            for metric in METRIC_NAMES:
                values = [row[metric] for row in entries if row[metric] is not None]
                result[metric] = float(np.mean(values)) if values else None
            result["fraction_worse_than_rgb"] = float(np.mean([row["delta_rgb_psnr"] < -1e-6 for row in entries]))
            methods[name] = result
        summaries[view] = dict(methods=methods, segmentation=segmentation_metrics(cm),
                               cm=cm.tolist(), input_psnr=float(np.mean([r["input_psnr"] for r in condition if r["method"] == "rgb"])))
        rows += condition
    return dict(seed=seed, held_video=fold["held_video"], source_SHA256=digest(DATA),
                conditions=summaries, max_tile_identity_error=max_identity_error), rows

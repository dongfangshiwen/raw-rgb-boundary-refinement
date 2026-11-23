"""Evaluate the prior amplitude and the new ways to estimate local RGB utility."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations, chain
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import torch

from scripts.author_pgl_v25 import pgl_rgb
from scripts.coarse_semantics_v39 import blend, native_labels, paint
from scripts.decision_overlap_v55 import MANIFEST, ROOT as PREVIOUS, action, models, pair_index
from scripts.dual_view_inference_v43 import two_views
from scripts.edge_projection_v47 import draw
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_orthogonal_risk_v52 import reference_models
from scripts.evaluate_overlap_quadratic_v53 import POPULATION, samples
from scripts.evaluate_real_semantics_v22 import load_segmenter, resize_image
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.fresh_real_v53 import CARD, RR, image, registration
from scripts.overlap_quadratic_v53 import SEEDS, decoded, geometry, label, references
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_detail_data_v24 import image_tensor
from scripts.real_haze_data_v22 import fingerprint
from scripts.relative_utility_v56 import ROOT, fitted, proposals, radial_update, relative_geometry, residual_mixture
from scripts.single_anchor_student_v50 import area_psnr
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


HELD = ROOT / "rrshid-unused-pairs.json"
HELD_DOMAIN = "rrshid_extra"
LEVELS = ("thin_fog", "moderate_fog", "thick_fog")
N_HELD = 45
OLD_AMPLITUDES = (.5, .75, 1.)
SSIM = ("anchor", "mct", "global", "local_direct", "residual_direct", "local_utility")


def holdout():
    source, _ = registration()
    inputs = dict(archive_SHA256=source["SHA256"], previous_test_SHA256=digest(CARD),
                  previously_selected_train_val_SHA256=digest(MANIFEST), script_SHA256=digest(__file__))
    if HELD.is_file():
        packet = read(HELD)
        if packet["inputs"] != inputs:
            raise ValueError("The extra RRSHID evaluation set has changed")
        return packet
    used = read(MANIFEST)["pairs"] + [r for r in read(CARD)["records"] if r["dataset"] == "rrshid"]
    numbers = {r["id"].rsplit("/", 1)[-1] for r in used}
    selected = []
    with ZipFile(RR / "RRSHID.zip") as data:
        fit_records = pair_index(data, {"train"})
        signatures = [fingerprint(image(data, r["clean"])) for r in used]
        for level in LEVELS:
            entries = [n for role, d, n in fit_records if role == "train" and d == level and n not in numbers]
            entries.sort(key=lambda n: sha256(f"v56/holdout/{level}/{n}".encode()).digest())
            accepted = 0
            for n in entries:
                if n in numbers:
                    continue
                row = fit_records["train", level, n]
                pair = [image(data, row[k]) for k in ("haze", "clean")]
                if any(v.size != (256, 256) for v in pair):
                    raise ValueError("An unused RRSHID pair does not align")
                phash, thumb = fingerprint(pair[1])
                if any(np.count_nonzero(phash != other) <= 8 and float(thumb @ grid) >= .96
                       for other, grid in signatures):
                    continue
                key = level + "/" + n
                selected.append(dict(id=key, condition=level, group=key, **row))
                numbers.add(n)
                signatures.append((phash, thumb))
                accepted += 1
                if accepted == N_HELD:
                    break
            if accepted != N_HELD:
                raise ValueError("Insufficient unused RRSHID pairs in a fixed severity stratum")
    packet = dict(inputs=inputs, records=selected, counts=dict(Counter(r["condition"] for r in selected)),
                  rule="Draw an additional cohort from the original training archive by a filename hash. Exclude "
                       "numeric IDs and near-duplicate clear images shared with the registered train, validation, "
                       "test or already accepted extra images. This is an in-domain extra holdout for this project.",
                  locked_utc=now())
    atomic_write(HELD, packet)
    print(dict(extra_rrshid=packet["counts"]), flush=True)
    return packet


def additional_samples(packet):
    with ZipFile(RR / "RRSHID.zip") as data:
        for row in packet["records"]:
            x, y = (image_tensor(image(data, row[k])) for k in ("haze", "clean"))
            yield HELD_DOMAIN, row["id"], row["group"], x, y, None, row["condition"]


VERTICES = np.vstack((np.zeros((1, 3)), np.eye(3)))
FACES = tuple(face for n in range(1, 5) for face in combinations(range(4), n))


def optimum_on_hull(gram, rhs):
    """Find the convex four-candidate global optimum by minimizing on all simplex faces."""
    champion, score = VERTICES[0], 0.
    for face in FACES:
        points = VERTICES[list(face)]
        start = points[0]
        span = (points[1:] - start).T
        if span.shape[1]:
            config = span.T @ gram @ span
            target = span.T @ (rhs - gram @ start)
            weight, _, _, _ = np.linalg.lstsq(config, target, rcond=1e-10)
            if (weight < -1e-9).any() or weight.sum() > 1. + 1e-9:
                continue
            weight = weight.clip(0.)
            convex = start + span @ (weight / max(1., weight.sum()))
        else:
            convex = start
        loss = convex @ gram @ convex - 2. * (rhs @ convex)
        if loss < score:
            champion, score = convex, loss
    return champion


def oracle(geom, truth, roi):
    dot, _ = label(geom, truth)
    if roi is None:
        mass = geom.partition.mass / geom.partition.mass.sum()
        gram = (geom.gram * mass[..., None, None]).sum(dim=(0, 1, 2))
        rhs = (dot * mass[..., None]).sum(dim=(0, 1, 2))
    else:
        mask = roi[None].expand(3, -1, -1).bool()
        directions = geom.deltas[:, mask]
        gram = (directions @ directions.T) / mask.sum()
        rhs = (directions @ (truth - geom.base)[0, mask]) / mask.sum()
    cfg = optimum_on_hull(gram.double().cpu().numpy(), rhs.double().cpu().numpy())
    center = geom.base.new_tensor(cfg).reshape(1, 1, 1, 3)
    offsets, _, variance = relative_geometry(geom.gram, center)
    gradient = dot - (geom.gram @ center.unsqueeze(-1)).squeeze(-1)
    slope = (offsets @ gradient.unsqueeze(-1)).squeeze(-1)
    radial = residual_mixture(center, offsets, radial_update(slope, variance, smooth=False))
    return area_psnr(decoded(geom, center.expand_as(radial)), truth, roi), area_psnr(decoded(geom, radial), truth, roi)


def paired_stats(rows):
    by_image = defaultdict(dict)
    sums = defaultdict(list)
    for row in rows:
        d, item, group, m, s = (row[k] for k in ("dataset", "id", "group", "method", "seed"))
        scopes = [d]
        if row["condition"] in LEVELS:
            scopes.append(f"{d}/{row['condition']}")
        for scope in scopes:
            sums[scope, m].append(row)
            by_image[scope, item, group][m, s] = row
    stats = {"/".join(k): {field: mean(r[field] for r in v if r[field] is not None)
                           for field in ("psnr", "ssim") if any(r[field] is not None for r in v)}
             for k, v in sums.items()}
    contrasts = (("local_utility", "global"), ("local_utility", "local_direct"), ("local_utility", "residual_direct"),
                 ("local_utility", "mct"), ("utility_full", "utility_diagonal"),
                 ("residual_direct", "global"), ("v55_full_1.00", "v55_full_0.50"))
    paired = defaultdict(lambda: defaultdict(list))
    for (scope, _, group), m in by_image.items():
        for left, right in contrasts:
            scores_left = [r["psnr"] for (name, _), r in m.items() if name == left]
            scores_right = [r["psnr"] for (name, _), r in m.items() if name == right]
            paired[scope, f"{left}-vs-{right}"][group].append(mean(scores_left) - mean(scores_right))
    return stats, {"/".join(k): bootstrap({g: mean(v) for g, v in groups.items()}) for k, groups in paired.items()}


@torch.inference_mode()
def evaluate():
    configure_cuda()
    extra = holdout()
    counts_expected = {**POPULATION, HELD_DOMAIN: len(extra["records"])}
    plan = dict(source_SHA256=digest(__file__), fit_SHA256=digest(ROOT / "completed.json"),
                old_fit_SHA256=digest(PREVIOUS / "completed.json"), extra_sha256=digest(HELD),
                population=counts_expected, v55_strengths=list(OLD_AMPLITUDES),
                hypothesis="Improve on global weights, a same-size direct local gate and a same-size residual direct gate "
                           "with shared global center. Compare the fixed utility fit with both diagonal and full solvers. "
                           "Use a no-shrink configuration for V56 and the frozen amplitude sweep for V55. "
                           "Quantitative authors' output comparisons are public pretrained checkpoints.",
                ssims="SSIM compares each public standalone output and the first preregistered seed for learned models.",
                oracle="Ground-truth diagnostic only. Compare the exact global RGB convex hull optimum and local line choices.")
    lock = ROOT / "evaluation_protocol.json"
    if lock.exists() and read(lock) != plan:
        raise ValueError("The V56 evaluation is fixed")
    if not lock.exists():
        atomic_write(lock, plan)
    rgb = references()
    pgl, rs, author = reference_models()
    bank, previous = fitted(), models()
    counts, rows, panels = Counter(), [], []
    oracle_rows, relative = defaultdict(list), defaultdict(list)
    demo_domains = ("lmhaze", "bedde", "nt_haze", "rrshid", HELD_DOMAIN, "realhyper_csnc")
    for d, item, group, haze, clear, roi, condition in chain(samples(), additional_samples(extra)):
        x, y = as_float(haze), as_float(clear)
        inside = roi.cuda() if roi is not None else None
        ctx = geometry(x, rgb)
        candidate = ctx.base + ctx.deltas[1:2]
        target = {("anchor", 0): ctx.base, ("mct", 0): candidate,
                  ("mix", 0): .5 * (ctx.base + candidate),
                  ("pgl_author", 0): pgl_rgb(pgl, x).clamp(0., 1.)}
        if d.startswith("rrshid") or d.startswith("realhyper"):
            target["dformer_rs_maxnorm", 0] = author_rgb(rs, x / (x.amax() + 1e-8)).clamp(0., 1.)
        points = proposals(ctx, bank)
        target.update({tag: decoded(ctx, weights) for tag, weights in points.items()})
        for seed in SEEDS:
            net = previous["full_image", seed]
            weights, _ = action(net, ctx.features, ctx.gram, ctx.scale, "full_image")
            target.update({(f"v55_full_{scale:.2f}", seed): decoded(ctx, weights, scale)
                           for scale in OLD_AMPLITUDES})
        scores = {}
        for (name, seed), out in target.items():
            quality = area_psnr(out, y, inside)
            ssim = float(gaussian_ssim(out, y)[0]) if name in SSIM and seed in (0, SEEDS[0]) else None
            scores[name, seed] = quality
            rows.append(dict(dataset=d, id=item, group=group, condition=condition, method=name, seed=seed,
                             psnr=quality, ssim=ssim))
        for seed in SEEDS:
            edit = (target["local_utility", seed] - target["global", seed]).square().mean().sqrt().item()
            relative[d].append((edit >= 1/255, scores["local_utility", seed] <= scores["global", seed] - .1,
                                float(points["global", seed][0, 0, 0, 1])))
        before, after = oracle(ctx, y, inside)
        oracle_rows[d].append((before, after))
        if d in demo_domains and not counts[d]:
            listing = (x, ctx.base, candidate, target["global", SEEDS[0]], target["residual_direct", SEEDS[0]],
                       target["local_utility", SEEDS[0]], y)
            panels.append((f"{d} / {item}", [to_rgb(v) for v in listing],
                           ("Hazy input", "Initial RGB", "Published MCT", "Global mixture", "Local residual gate",
                            "Local predicted utility", "Clear reference")))
        counts[d] += 1
        if counts[d] == counts_expected[d]:
            print(d, counts[d], {m: round(mean(r["psnr"] for r in rows if r["dataset"] == d and r["method"] == m), 4)
                                  for m in ("anchor", "mct", "global", "local_direct", "residual_direct", "local_utility",
                                            "utility_diagonal", "utility_full", "v55_full_0.50", "v55_full_1.00")}, flush=True)
    if counts != counts_expected:
        raise ValueError(f"The V56 population is incomplete: {counts}")
    stats, compared = paired_stats(rows)
    individual, panel = ROOT / "individual.json", ROOT / "predictions.png"
    atomic_write(individual, rows)
    draw(panel, panels)
    atomic_write(ROOT / "test.json", dict(counts=dict(counts), rows=len(rows), means=stats, paired=compared,
        oracle_only={d: dict(global_exact=mean(a), radial_local=mean(b), difference=mean(b)-mean(a))
                     for d, cohort in oracle_rows.items() for a, b in [tuple(zip(*cohort))]},
        diagnostics={d: dict(edited=mean(e), harmed=mean(h), average_global_mct_weight=mean(m))
                     for d, cohort in relative.items() for e, h, m in [tuple(zip(*cohort))]},
        author=author, releases=rgb[2], protocol_SHA256=digest(lock), individual_SHA256=digest(individual),
        panel_SHA256=digest(panel), finished_utc=now()))


@torch.inference_mode()
def infer(path, directory):
    configure_cuda()
    output_names = ("raw", "anchor", "mct", "global_mix", "restored", "mask", "color_mask", "overlay", "comparison")
    if path.resolve() in {(directory / f"{name}.png").resolve() for name in output_names}:
        raise ValueError("Output must differ from the input")
    with Image.open(path) as input_file:
        pixels = np.array(ImageOps.exif_transpose(input_file).convert("RGB"))
    x = as_float(torch.from_numpy(pixels.copy()).permute(2, 0, 1))
    ctx = geometry(x, references())
    points = proposals(ctx, fitted())
    _, _, segmentation = two_views(*load_segmenter()[:3], resize_image(pixels)[0])
    mask = native_labels(segmentation, pixels.shape[:2]).byte().cpu().numpy()
    result = dict(raw=pixels, anchor=to_rgb(ctx.base), mct=to_rgb(ctx.base + ctx.deltas[1:2]),
                  global_mix=to_rgb(decoded(ctx, points["global", SEEDS[0]])),
                  restored=to_rgb(decoded(ctx, points["local_utility", SEEDS[0]])),
                  mask=mask, color_mask=paint(mask), overlay=blend(pixels, mask))
    if mask.shape != result["restored"].shape[:2] or not np.isin(mask, range(19)).all():
        raise ValueError("The two outputs differ in size or label space")
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        Image.fromarray(frame).save(directory / f"{name}.png")
    draw(directory / "comparison.png",
         ((path.name, [result[k] for k in ("raw", "mct", "global_mix", "restored")],
           ("Hazy input", "Published MCT", "Global mixture", "Local predicted utility")),
          ("19-class segmentation", [result[k] for k in ("raw", "color_mask", "overlay")],
           ("Segmentation input", "Predicted class mask", "Predicted overlay"))))
    atomic_write(directory / "metadata.json", dict(native=list(mask.shape[::-1]), input=str(path.resolve()),
        input_SHA256=digest(path), seed=SEEDS[0], files={f"{f}.png": digest(directory / f"{f}.png")
        for f in (*result, "comparison")}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("holdout")
    commands.add_parser("test")
    demo = commands.add_parser("infer")
    demo.add_argument("--input", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.command == "holdout":
        holdout()
    elif args.command == "test":
        evaluate()
    else:
        infer(args.input, args.output)

"""Score frozen V22 fits, published references and the two output paths."""
from collections import defaultdict
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import write_csv
from boundary_refinement.models.restoration.appearance_refiner import RECIPES
from boundary_refinement.data.depth_correlated_fog import fixed_view, load_proxy
from boundary_refinement.evaluation.photographic.evaluate_author_mct import load_author_reference, OFFICIAL
from boundary_refinement.evaluation.photographic.evaluate_parallel_rgb import bind as bind_author
from boundary_refinement.evaluation.semantic.evaluate_regional_guidance import psnr
from boundary_refinement.inference.infer_real_transfer import load_model, restore, semantic_logits
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.baselines.published_runtime import dehazeformer_t
from boundary_refinement.data.real_haze_data import HEIGHT, ROOT, WIDTH, load_archived_pairs, load_data, overlap_check
from boundary_refinement.data.regional_data import prepare_grouped_data, validation_view
from boundary_refinement.training.train_boundary_flux import ALL_VIEWS
from boundary_refinement.training.train_real_transfer import AUTHOR, author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda
from boundary_refinement.training.fit_sg_aod import METHODS as SG_METHODS, ROOT as SG_ROOT, author_factory, construct, hints


METRICS = ("psnr", "ssim", "luma_bias", "abs_luma_bias", "channel_bias", "gradient_mae")
SEG_INPUTS = ("hazy_input", "v21_rgb", "paired_free", "paired_detail", "affine_spatial")
FULL = (0, 0, WIDTH, HEIGHT)
REAL = ("nh_holdout", "dense", "ohaze")
SPATIAL = ("spatial", "spatial_alt", "aligned_dense")


def verify_weights():
    path = ROOT / "plan.json"
    plan, sha = read(path), digest(path)
    for name, expected in plan["source_SHA256"].items():
        if digest(name) != expected:
            raise ValueError("Training dependency changed: " + name)
    runs = read(ROOT / "completed.json")
    author = read(ROOT / "author_completed.json")
    sg = read(SG_ROOT / "completed.json")
    group = lambda rows: {(row["video"], row["seed"]) for row in rows}
    if (not runs["all_complete"] or not author["all_complete"] or not sg["all_registered_runs_complete"]
            or group(runs["runs"]) != group(plan["fixed_base"]) or group(author["runs"]) != group(sg["runs"])
            or group(author["runs"]) != {(video, seed) for video, seed in group(runs["runs"]) if seed == 42}
            or runs["plan_SHA256"] != sha or author["plan_SHA256"] != sha):
        raise ValueError("Training must be complete across registered groups")
    for row in runs["runs"] + author["runs"] + sg["runs"]:
        if digest(row["checkpoint"]) != row["checkpoint_SHA256"]:
            raise ValueError("A final checkpoint changed")
    return plan, runs, author, sg


def prepare_evaluation():
    real = load_data("evaluation")
    manifest = read(ROOT / "data_manifest.json")
    ids = manifest["training_archive"]["remaining_ids"]
    if ids != [f"{i:02d}" for i in range(46, 56)]:
        raise ValueError("Wrong NH-HAZE holdout IDs")
    samples, proof, archive = load_archived_pairs("NH-HAZE.zip", "nh_holdout", ids)
    _, training, src = load_archived_pairs("NH-HAZE.zip", "nh_training", [r["id"] for r in manifest["training"]["rows"]])
    if archive["SHA256"] != src["SHA256"] or archive["SHA256"] != manifest["training_archive"]["SHA256"]:
        raise ValueError("NH archive changed since registration")
    flagged, nearest = overlap_check(training, proof)
    excluded = {row["id"] for row in flagged}
    real["nh_holdout"] = [row for row in samples if row["id"] not in excluded]
    if len(real["nh_holdout"]) < 5:
        raise ValueError("Too few group-independent NH samples left after image duplicate checks")
    record = dict(remaining_ids=ids, exclusions=flagged, closest_thumbnail_match=nearest,
                  input_geometry=[WIDTH, HEIGHT], subsets={key: [r["id"] for r in real[key]] for key in REAL},
                  training_manifest_SHA256=digest(ROOT / "data_manifest.json"), archive=archive)
    atomic_write(ROOT / "held_out_images.json", record)
    return real


def gaussian_ssim(image, target):
    kernel = torch.arange(-5, 6, device=image.device, dtype=image.dtype).square().mul(-1/(2*1.5**2)).exp()
    kernel = (kernel/kernel.sum())[None, None].expand(3, 1, 11)
    x, y = kernel.unsqueeze(2).contiguous(), kernel.unsqueeze(3).contiguous()
    def blur(value):
        return F.conv2d(F.conv2d(value, x, groups=3), y, groups=3)
    a, b = blur(image), blur(target)
    av, bv, cov = blur(image.square()) - a.square(), blur(target.square()) - b.square(), blur(image*target) - a*b
    score = (2*a*b+.01**2)*(2*cov+.03**2)/((a.square()+b.square()+.01**2)*(av+bv+.03**2))
    return score.mean((1, 2, 3))


def quality(image, target):
    if image.shape != target.shape:
        raise ValueError("Prediction and reference geometry differ")
    error = image - target
    reconstruct = psnr(error.square().mean((1, 2, 3)))
    per_channel = error.mean((-1, -2))
    luma = per_channel @ error.new_tensor([.2126, .7152, .0722])
    gradient = .5*(error.diff(dim=-1).abs().mean((1, 2, 3)) + error.diff(dim=-2).abs().mean((1, 2, 3)))
    scores = torch.stack((reconstruct, gaussian_ssim(image, target), luma, luma.abs(),
                          per_channel.abs().mean(1), gradient), dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite quality score")
    return scores.tolist()


def add_metrics(rows, outputs, clean, ids, rects, dataset, condition, video, seed):
    regions = defaultdict(list)
    for i, roi in enumerate(rects):
        regions[tuple(roi)].append(i)
    for (x1, y1, x2, y2), index in regions.items():
        target = clean[index, :, y1:y2, x1:x2].contiguous()
        for method, image in outputs.items():
            rgb = image[index, :, y1:y2, x1:x2].contiguous()
            for i, values in zip(index, quality(rgb, target)):
                rows.append(dict(dataset=dataset, condition=condition, video=video, seed=seed, method=method,
                                 image=ids[i], **dict(zip(METRICS, values))))


def real_batches(samples):
    for begin in range(0, len(samples), 8):
        group = samples[begin:begin+8]
        hazy = torch.stack([v["hazy"] for v in group]).cuda().float()/255
        clean = torch.stack([v["clean"] for v in group]).cuda().float()/255
        yield hazy, clean, [v["id"] for v in group], [v["roi"] for v in group]


def camvid_batches(data, fold, condition):
    indices = fold["indices"]["validation"]
    for begin in range(0, len(indices), 8):
        index = indices[begin:begin+8]
        target = data["raw"][index].cuda().float()/255
        if condition == "aligned_dense":
            image, _ = fixed_view(target, data["proxy_distance"][index].cuda().float(), condition)
        else:
            image = (255*validation_view(target, condition)).round()/255
        yield image, target, data["mask"][index].cuda().long(), [data["ids"][i] for i in index]


def model_outputs(core, anchor, heads, transformer, sg, image):
    outputs = dict(hazy_input=image, **restore(core, anchor, heads, image))
    if transformer is not None:
        outputs[AUTHOR] = author_rgb(transformer, image).clamp(0, 1)
    if sg:
        guide = hints(core, image)
        outputs.update({name: model(image, guide[name]).float().clamp(0, 1) for name, model in sg.items()})
    return outputs


def load_transformer(run):
    state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    if set(state["models"]) != {AUTHOR} or any(state[k] != run[k] for k in ("seed", "video", "plan_SHA256")):
        raise ValueError("Published matched transformer checkpoint metadata changed")
    model = dehazeformer_t()
    model.load_state_dict(state["models"][AUTHOR], strict=True)
    return model.cuda().eval()


def load_sg(run, core_path):
    state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    anchor = torch.load(core_path, map_location="cpu", weights_only=True)
    if (state["format"] != "sg-aod-camvid-v21" or Path(state["base"]) != Path(anchor["base"])
            or state["base_SHA256"] != anchor["base_SHA256"]):
        raise ValueError("Published SG-AOD baseline and test segmentation checkpoints disagree")
    factory = author_factory()
    output = {}
    for name in SG_METHODS:
        module = construct(factory, run["seed"])
        module.load_state_dict(state["models"][name], strict=True)
        output[name] = module.eval()
    return output


@torch.inference_mode()
def score_group(candidate, transformer, sg, data, fold, real):
    core, anchor, heads, info = load_model(candidate["checkpoint"])
    if any(info[k] != candidate[k] for k in info):
        raise ValueError("V22 score group does not match final weights")
    packet = torch.load(candidate["checkpoint"], map_location="cpu", weights_only=True)
    live_author = load_transformer(transformer) if transformer else None
    live_sg = load_sg(sg, packet["base"]["checkpoint"]) if sg else {}
    rows, semantics = [], []
    video, seed = candidate["video"], candidate["seed"]
    for subset, samples in real.items():
        for image, clean, ids, rects in real_batches(samples):
            outputs = model_outputs(core, anchor, heads, live_author, live_sg, image)
            add_metrics(rows, outputs, clean, ids, rects, subset, "paired", video, seed)
    for view in ALL_VIEWS:
        cms = {name: torch.zeros((11, 11), dtype=torch.int64) for name in SEG_INPUTS}
        truth = torch.zeros(11, dtype=torch.int64)
        for image, clean, labels, ids in camvid_batches(data, fold, view):
            outputs = model_outputs(core, anchor, heads, live_author, live_sg, image)
            add_metrics(rows, outputs, clean, ids, [FULL]*len(image), "camvid_dev", view, video, seed)
            truth += torch.bincount(labels.flatten(), minlength=256)[:11].cpu()
            for name in SEG_INPUTS:
                logits = semantic_logits(core, outputs[name])
                cms[name] += confusion_matrix(logits.argmax(1), labels, 11)
        for name, cm in cms.items():
            if not torch.equal(cm.sum(1), truth):
                raise ValueError("CamVid class totals changed between image interventions")
            semantics.append(dict(video=video, seed=seed, condition=view, input=name,
                                  images=len(fold["indices"]["validation"]), metrics=segmentation_metrics(cm),
                                  confusion=cm.tolist()))
    return rows, semantics


@torch.inference_mode()
def score_mct(real):
    bind_author()
    model, registration = load_author_reference()
    rows = []
    for name, samples in real.items():
        for image, clean, ids, rects in real_batches(samples):
            predictions = author_rgb(model, image).clamp(0, 1)
            add_metrics(rows, {"dehazeformer_mct": predictions}, clean, ids, rects, name, "paired", "external", -1)
    return rows, registration


def averages(rows):
    keyed = defaultdict(list)
    for row in rows:
        keyed[row["dataset"], row["condition"], row["method"], row["video"], row["seed"]].append(row)
    mean = {key: {metric: float(np.mean([r[metric] for r in group])) for metric in METRICS} for key, group in keyed.items()}
    grouped = defaultdict(list)
    for (dataset, view, method, _, _), value in mean.items():
        grouped[dataset, view, method].append(value)
    return {f"{dataset}/{view}/{method}": {metric: float(np.mean([r[metric] for r in samples])) for metric in METRICS}
            for (dataset, view, method), samples in grouped.items()}


def real_contrasts(rows, contrasts):
    index = defaultdict(dict)
    for row in rows:
        if row["dataset"] in REAL:
            index[row["dataset"]][row["image"], row["video"], row["seed"], row["method"]] = row
    output = {}
    rng = np.random.default_rng(622)
    for dataset, records in index.items():
        comparisons = {}
        for name, (method, control) in contrasts.items():
            paired = defaultdict(list)
            groups = defaultdict(list)
            for (image, video, seed, current), row in records.items():
                if current != method or (image, video, seed, control) not in records:
                    continue
                base = records[image, video, seed, control]
                changes = [row[metric]-base[metric] for metric in METRICS]
                paired[image].append(changes)
                groups[video, seed].append(changes)
            if not paired:
                raise ValueError("No matched entries for " + name)
            by_image = np.array([np.mean(measurements, axis=0) for measurements in paired.values()])
            draws = rng.integers(len(by_image), size=(10000, len(by_image)))
            ci = np.percentile(by_image[draws].mean(1), [2.5, 97.5], axis=0).T
            comparisons[name] = dict(left=method, right=control, images=len(by_image), groups=len(groups),
                                     delta=dict(zip(METRICS, by_image.mean(0).tolist())),
                                     bootstrap95={k: [float(a), float(b)] for k, (a, b) in zip(METRICS, ci)},
                                     delta_PSNR_by_group={f"{video}/{seed}": float(np.mean(values, axis=0)[0])
                                                          for (video, seed), values in groups.items()},
                                     fraction_images_improved_PSNR=float((by_image[:, 0] > 0).mean()))
        output[dataset] = comparisons
    return output


def main():
    configure_cuda()
    plan, candidates, author, sg = verify_weights()
    plan_file = ROOT / "evaluation_plan.json"
    program = Path(__file__)
    registration = dict(protocol="Final epoch only; first decode NH holdout after the 39 registered final fits. All reference scores recomputed on the same 256×192 letterbox; PSNR, Gaussian SSIM and other RGB scores exclude padding. Gray-value statistics use encoded RGB.",
                        columns=list(METRICS), source_views=list(ALL_VIEWS), segmentation_inputs=list(SEG_INPUTS),
                        sources={str(p): digest(p) for p in (program, Path(load_model.__code__.co_filename),
                                                             ROOT / "plan.json", ROOT / "completed.json",
                                                             ROOT / "author_completed.json", SG_ROOT / "completed.json",
                                                             OFFICIAL / "reference_registration.json")})
    if plan_file.exists() and read(plan_file) != registration:
        raise ValueError("Evaluation source or training record changed")
    if not plan_file.exists():
        atomic_write(plan_file, registration)
    real = prepare_evaluation()
    data, folds = prepare_grouped_data()
    load_proxy(data)
    by_video = {f["held_video"]: f for f in folds}
    key = lambda record: (record["video"], record["seed"])
    authors = {key(run): run for run in author["runs"]}
    older = {key(run): run for run in sg["runs"]}
    rows, segmentation = [], []
    for run in candidates["runs"]:
        card, masks = score_group(run, authors.get(key(run)), older.get(key(run)), data, by_video[run["video"]], real)
        rows.extend(card)
        segmentation.extend(masks)
        print(dict(video=run["video"], seed=run["seed"], rgb_rows=len(card), mask_results=len(masks)), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    extra, pretraining = score_mct(real)
    rows.extend(extra)
    combined = averages(rows)
    comparisons = dict(**plan["contrasts"], paired_vs_v21=["paired_free", "v21_rgb"],
                       spatial_vs_synthetic=["affine_spatial", "synthetic_free"],
                       matched_author=["paired_free", AUTHOR], author_sg_semantic=["sg_semantic", "sg_rgb"])
    counterfactuals = real_contrasts(rows, comparisons)
    table = ROOT / "rgb_by_image.csv"
    write_csv(table, rows)
    atomic_write(ROOT / "semantics_camvid.json", segmentation)
    data_sizes = {name: len(real[name]) for name in REAL}
    summary = dict(completed_utc=now(), training_fits=candidates["fits"]+author["fits"],
                   training_updates=candidates["updates"]+author["updates"], rgb_rows=len(rows),
                   image_counts=data_sizes, held_camvid_videos=[f["held_video"] for f in folds],
                   mean_equal_video_seed=combined, bootstrap_sampling_unit="An RGB pair, averaged over matched fixed model groups. Intervals do not model new site, new source video or retraining uncertainty.",
                   real_paired_changes=counterfactuals, rgb_table_SHA256=digest(table),
                   segmentation_SHA256=digest(ROOT / "semantics_camvid.json"), mct=pretraining)
    atomic_write(ROOT / "scores.json", summary)
    print(dict(real_counts=data_sizes, rgb_rows=len(rows), segmentation_rows=len(segmentation),
               trained=summary["training_fits"], CUDA_updates=summary["training_updates"]), flush=True)


if __name__ == "__main__":
    main()

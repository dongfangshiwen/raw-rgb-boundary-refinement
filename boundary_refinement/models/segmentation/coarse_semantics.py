"""Compare boundary supervision on a fixed real-fog segmentation split."""
import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.data.cohorts.audit_real_fog import NAMES, city_train_ids
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.evaluation.photographic.evaluate_paired_basis import load_blend
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import (
    DRIVING, ZURICH, load_pairs, load_segmenter, resize_image, verified_image,
)
from boundary_refinement.inference.infer_mct_segformer import CITY_COLORS
from boundary_refinement.training.parallel_utility import PROTOCOL as SPLIT, capture_group
from boundary_refinement.training.point_strength import ROOT as RGB_ROOT, state as rgb_state
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.structure_transfer import SEEDS
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/fog-segmentation-v39"
ARMS = ("hard", "relaxed", "single_scale", "two_scale")
MAIN = "two_scale"
TESTS = ("foggy_zurich_testv2", "foggy_driving_fine")
COUNTS = dict(zip(TESTS, (40, 33), strict=True))
STEPS, WINDOW, BATCH = 192, 72, 4
MIX, CONFIDENCE = .35, .50
EXAMPLES = (0, 10)


def partitions():
    source = read(SPLIT)
    if source["dataset_manifest"] != digest(DRIVING) or source["zurich_manifest"] != digest(ZURICH):
        raise ValueError("Previously audited Foggy Driving split differs")
    manifest = read(DRIVING)["rows"]
    fit = set(source["data"]["fit_ids"])
    val = set(source["data"]["validation_ids"])
    ids = {r["id"]: r for r in manifest if r["annotation_kind"] == "gtCoarse"}
    groups = lambda selected: {capture_group(x) for x in selected}
    fine = {r["id"] for r in manifest if r["annotation_kind"] == "gtFine"}
    if (len(fit), len(val)) != (50, 6) or not (fit | val) <= ids.keys():
        raise ValueError("Expected the registered coarse train and validation pools")
    if groups(fit) & groups(val) or groups(fit | val) & groups(fine):
        raise ValueError("A registered capture group crosses the fit and held datasets")
    return [ids[k] for k in sorted(fit)], [ids[k] for k in sorted(val)]


def coarse_pair(row):
    rgb = verified_image(row["rgb"], row["rgb_SHA256"], False)
    mask = city_train_ids(verified_image(row["annotation"], row["mask_SHA256"], True))
    if rgb.shape[:2] != mask.shape:
        raise ValueError("Coarse RGB and label grids differ")
    return rgb, mask


def head_features(model, x, mean, std):
    cached = []
    hook = model.decode_head.classifier.register_forward_pre_hook(lambda _, args: cached.append(args[0]))
    try:
        output = model(pixel_values=(x - mean) / std).logits.float()
    finally:
        hook.remove()
    if len(cached) != 1:
        raise ValueError("SegFormer classifier feature tap changed")
    return cached[0], output


def valid_neighbors(labels):
    valid = labels != 255
    one_hot = F.one_hot(labels.clamp(0, 18), 19).permute(2, 0, 1)[None].float()
    return F.max_pool2d(one_hot * valid, 5, 1, 2)[0].bool() & valid


def choose_alternatives(first_logits, second_logits, labels, allowed):
    """Keep an identical number of changed pixels per coarse class and image."""
    scores, alternative = first_logits.softmax(1)[0].max(0)
    short_scores, short_labels = second_logits.softmax(1)[0].max(0)
    neighbor = allowed.gather(0, alternative[None])[0]
    plausible = (labels != 255) & neighbor & (labels != alternative)
    trusted = plausible & (scores >= CONFIDENCE)
    paired = trusted & (alternative == short_labels) & (short_scores >= CONFIDENCE)
    single = torch.zeros_like(paired)
    for category in labels[paired].unique().tolist():
        count = int((paired & (labels == category)).sum())
        domain = (trusted & (labels == category)).flatten().nonzero().flatten()
        best = scores.flatten()[domain].argsort(descending=True, stable=True)[:count]
        single.flatten()[domain[best]] = True
    if int(paired.sum()) != int(single.sum()):
        raise ValueError("Single-scale and two-scale soft target budgets differ")
    return alternative.to(torch.uint8), single, paired


@torch.no_grad()
def cache(model, mean, std, rows):
    items = []
    for pos, row in enumerate(rows, 1):
        rgb, labels = coarse_pair(row)
        x, _ = resize_image(rgb)
        features, high = head_features(model, x, mean, std)
        size = tuple(round(v * .70) for v in x.shape[-2:])
        x_short = F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=True)
        low = model(pixel_values=(x_short - mean) / std).logits.float()
        low = F.interpolate(low, high.shape[-2:], mode="bilinear", align_corners=False)
        labels = torch.from_numpy(labels.copy())[None, None].cuda().float()
        y = F.interpolate(labels, high.shape[-2:], mode="nearest-exact")[0, 0].long()
        allowed = valid_neighbors(y)
        alternative, single, paired = choose_alternatives(high, low, y, allowed)
        items.append(dict(id=row["id"], features=features[0].half(),
                          labels=y.byte(), allowed=allowed, alternate=alternative,
                          single_scale=single, two_scale=paired))
        if pos % 10 == 0 or pos == len(rows):
            print(dict(cached=pos, total=len(rows)), flush=True)
    return items


def train_batches(items, seed):
    rng = torch.Generator().manual_seed(seed + 39041)
    for _ in range(STEPS):
        indices = torch.randint(len(items), (BATCH,), generator=rng).tolist()
        batch = defaultdict(list)
        for idx in indices:
            row = items[idx]
            h, w = row["labels"].shape
            crop_h, crop_w = min(h, WINDOW), min(w, WINDOW)
            top = int(torch.randint(h - crop_h + 1, (), generator=rng))
            left = int(torch.randint(w - crop_w + 1, (), generator=rng))
            bounds = (..., slice(top, top + crop_h), slice(left, left + crop_w))
            pad = (0, WINDOW - crop_w, 0, WINDOW - crop_h)
            for name in ("features", "labels", "allowed", "alternate", "single_scale", "two_scale"):
                batch[name].append(F.pad(row[name][bounds], pad, value=255 if name == "labels" else 0))
        yield {k: torch.stack(v) for k, v in batch.items()}


def objective(logits, batch, arm):
    y = batch["labels"].long()
    valid = y != 255
    logp = logits.log_softmax(1)
    hard = -logp.gather(1, y.clamp_max(18)[:, None])[:, 0]
    if arm == "relaxed":
        eligible = batch["allowed"] | (~valid[:, None])
        soft = -logp.masked_fill(~eligible, -100.).logsumexp(1)
        losses = hard.lerp(soft, MIX)
    elif arm in ("single_scale", MAIN):
        soft = -logp.gather(1, batch["alternate"].long()[:, None])[:, 0]
        losses = hard.lerp(soft, MIX * batch[arm])
    else:
        losses = hard
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def fit(smoke):
    configure_cuda()
    rows, val = partitions()
    model, mean, std, registration = load_segmenter()
    parent = model.decode_head.classifier.requires_grad_(False)
    count = sum(p.numel() for p in parent.parameters())
    protocol = dict(
        purpose="Compare a same-capacity two-scale choice of soft boundary targets against ordinary classifier tuning, boundary label relaxation and a matched per-class single-scale choice.",
        seed=list(SEEDS), arms=list(ARMS), candidate=MAIN, trainable=count, crop=WINDOW, steps=STEPS,
        official_test_counts=COUNTS, coarse_fit_ids=[r["id"] for r in rows],
        coarse_validation_ids=[r["id"] for r in val], frozen_model=registration,
        hashes={str(p): digest(p) for p in (SPLIT, DRIVING, ZURICH, Path(__file__))},
        candidate_rule="A trusted alternative must be within two stride-four cells in the original coarse labels and be the frozen teacher top class at both 1024- and 717-pixel long edges, each at >=0.50 confidence. The single-scale control keeps exactly as many changes of each coarse class in each image. Each alternative receives mass 0.35. All arms exclude 255 from the loss.",
        optimization="Tune only the existing SegFormer classifier initialized with the author's weights. Shared cached feature patches, AdamW, LR 0.0005 with cosine decay to 0.00005, CUDA BF16. RGB branch separate. Validate mIoU and boundary-band IoU at native label resolution. All methods use fixed final update weights.",
        scope="Use the earlier photographed coarse/fine split and exact confidence-free held label grids. Test photographs have been evaluated earlier by the project. Dense overlays are exported at native dimensions; test labels are used for scoring and display only.",
    )
    path = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if path.exists() and read(path) != protocol:
            raise ValueError("The segmentation training protocol changed")
        if not path.exists():
            atomic_write(path, protocol)
    features = cache(model, mean, std, rows[:4] if smoke else rows)
    possible = sum(int(v["two_scale"].sum()) for v in features)
    valid = sum(int((v["labels"] != 255).sum()) for v in features)
    if possible == 0 or possible >= valid:
        raise ValueError("No usable real-fog soft label alternative")
    print(dict(soft_pixels=possible, annotated_pixels=valid), flush=True)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for arm in ARMS:
            weights = ROOT / f"{arm}-{seed}.pth"
            receipt = weights.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["checkpoint_SHA256"] != digest(weights) or row["protocol_SHA256"] != digest(path):
                    raise ValueError("A segmentation training checkpoint changed")
                ledger.append(row)
                continue
            set_seed(seed)
            head = deepcopy(parent).requires_grad_(True)
            optimizer = torch.optim.AdamW(head.parameters(), lr=.0005, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=5e-5)
            start = time.perf_counter()
            for step, batch in enumerate(train_batches(features, seed), 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = head(batch["features"])
                loss = objective(logits.float(), batch, arm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                if smoke and step == 2:
                    break
            if not smoke and step != STEPS:
                raise ValueError("Unequal segmentation training budgets")
            result = dict(arm=arm, seed=seed, updates=step, trainable=count, loss=loss.item(),
                          seconds=round(time.perf_counter() - start, 2))
            print(result, flush=True)
            if not smoke:
                torch_save(weights, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                row = dict(result, file=str(weights), checkpoint_SHA256=digest(weights),
                           protocol_SHA256=digest(path), finished_utc=now())
                atomic_write(receipt, row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, updates=sum(r["updates"] for r in ledger),
                     soft_pixels=possible, annotated_pixels=valid, protocol_SHA256=digest(path),
                     gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now()))


def load_heads(initial):
    card = read(ROOT / "completed.json")
    if card["protocol_SHA256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Segmentation training provenance changed")
    if {(r["arm"], r["seed"]) for r in card["runs"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Not all segmentation comparators were trained")
    result = {}
    for row in card["runs"]:
        if digest(row["file"]) != row["checkpoint_SHA256"]:
            raise ValueError("Segmentation weights changed")
        head = deepcopy(initial)
        head.load_state_dict(torch.load(row["file"], map_location="cpu", weights_only=True), strict=True)
        result[row["arm"], row["seed"]] = head.eval().requires_grad_(False)
    return result


def boundary_ids(labels):
    valid = labels != 255
    dx = (labels[:, 1:] != labels[:, :-1]) & (valid[:, 1:] & valid[:, :-1])
    dy = (labels[1:, :] != labels[:-1, :]) & (valid[1:, :] & valid[:-1, :])
    contour = F.pad(dx, (1, 0)) | F.pad(dx, (0, 1)) | F.pad(dy, (0, 0, 1, 0)) | F.pad(dy, (0, 0, 0, 1))
    radius = max(1, round(3 * max(labels.shape) / 1024))
    band = F.max_pool2d(contour[None, None].float(), 2 * radius + 1, 1, radius)[0, 0].bool()
    return labels.masked_fill(~band, 255)


def native_labels(logits, shape):
    return F.interpolate(logits, shape, mode="bilinear", align_corners=False).argmax(1)[0]


def paint(mask):
    palette = np.zeros((256, 3), np.uint8)
    palette[:19] = CITY_COLORS
    return palette[mask]


def blend(rgb, mask):
    return np.rint(.57 * rgb + .43 * paint(mask)).astype(np.uint8)


def rgb_checkpoint():
    restorer = load_pgl()[0]
    row = next(r for r in rgb_state() if r["seed"] == SEEDS[0])
    return load_blend(restorer, row, 1.), row["checkpoint_SHA256"]


def save_demo(directory, rgb, restored, truth, labels, hard, candidate, scene):
    directory.mkdir(parents=True, exist_ok=True)
    pictures = {
        "input": rgb, "restored": restored, "ground_truth": paint(truth),
        "mask": candidate, "color_mask": paint(candidate), "overlay": blend(rgb, candidate),
        "frozen_overlay": blend(rgb, labels), "hard_overlay": blend(rgb, hard),
    }
    for name, value in pictures.items():
        Image.fromarray(value).save(directory / f"{name}.png")
    names = ("Hazy input", "Restored RGB", "Frozen SegFormer", "Hard-label tuning", "Two-scale tuning", "Fine annotation")
    views = ("input", "restored", "frozen_overlay", "hard_overlay", "overlay", "ground_truth")
    width, height, pad = 390, 273, 10
    panel = Image.new("RGB", (3 * width, 2 * height + 40), "#f4f7fa")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=17)
    title_font = ImageFont.load_default(size=20)
    draw.text((pad, 11), scene, font=title_font, fill="#1f2a37")
    for i, (name, key) in enumerate(zip(names, views, strict=True)):
        x, y = (i % 3) * width, 40 + (i // 3) * height
        draw.text((x + pad, y + 4), name, font=font, fill="#1f2a37")
        image = Image.fromarray(pictures[key])
        image.thumbnail((width - 2 * pad, height - 38), Image.Resampling.LANCZOS)
        panel.paste(image, (x + (width - image.width) // 2, y + 28 + (height - 38 - image.height) // 2))
    panel.save(directory / "comparison.png")
    return {v.name: digest(v) for v in directory.glob("*.png")}


def summarize(matrices):
    report = defaultdict(list)
    details = {}
    for (dataset, arm, seed), values in matrices.items():
        measures = {metric: segmentation_metrics(mat) for metric, mat in values.items()}
        pair = dict(miou=100 * measures["full"]["miou"], boundary_miou=100 * measures["boundary"]["miou"],
                    per_class_iou=[100 * v for v in measures["full"]["class_iou"]])
        report[dataset, arm].append(pair)
        details[f"{dataset}/{arm}/{seed}"] = pair
    combined = {"/".join(k): {metric: float(np.mean([r[metric] for r in values]))
                               for metric in ("miou", "boundary_miou")} for k, values in report.items()}
    return combined, details


@torch.inference_mode()
def test():
    configure_cuda()
    _, val = partitions()
    model, mean, std, registration = load_segmenter()
    heads = load_heads(model.decode_head.classifier)
    restorer, restorer_hash = rgb_checkpoint()
    matrices = defaultdict(lambda: {
        m: torch.zeros((19, 19), dtype=torch.int64) for m in ("full", "boundary")
    })
    records, exports, counts = [], {}, {}
    for dataset in (*TESTS, "coarse_validation"):
        examples = load_pairs(dataset) if dataset in TESTS else (
            (r["id"], capture_group(r["id"]), *coarse_pair(r)) for r in val
        )
        count = 0
        for index, (scene, group, rgb, label) in enumerate(examples):
            if rgb.shape[:2] != label.shape or not np.isin(np.unique(label), [*range(19), 255]).all():
                raise ValueError("Unexpected native 19-class evaluation grid")
            target = torch.from_numpy(label.astype(np.int64)).cuda()
            edge_target = boundary_ids(target)
            expected = torch.bincount(target.flatten(), minlength=256)[:19].cpu()
            expected_edge = torch.bincount(edge_target.flatten(), minlength=256)[:19].cpu()
            raw, _ = resize_image(rgb)
            feats, before = head_features(model, raw, mean, std)
            canonical = {("frozen", 0): before}
            canonical.update({key: head(feats) for key, head in heads.items()})
            render = dataset in TESTS and index in EXAMPLES
            demo = {}
            for (arm, seed), logits in canonical.items():
                prediction = native_labels(logits, target.shape)
                full = confusion_matrix(prediction, target, 19)
                edge = confusion_matrix(prediction, edge_target, 19)
                if not torch.equal(full.sum(1), expected) or not torch.equal(edge.sum(1), expected_edge):
                    raise ValueError("A method changed the scored label or boundary pixels")
                matrices[dataset, arm, seed]["full"] += full
                matrices[dataset, arm, seed]["boundary"] += edge
                records.append(dict(dataset=dataset, scene=scene, group=group, arm=arm, seed=seed,
                                    full=full.tolist(), boundary=edge.tolist()))
                if render and (arm, seed) in (("frozen", 0), ("hard", SEEDS[0]), (MAIN, SEEDS[0])):
                    demo[arm] = prediction.byte().cpu().numpy()
            if render:
                native = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].cuda().float() / 255.
                clear = (pgl_rgb(restorer, native).clamp(0., 1.)[0] * 255.).round().byte().permute(1, 2, 0).cpu().numpy()
                if clear.shape != rgb.shape:
                    raise ValueError("RGB restoration geometry differs from segmentation")
                destination = ROOT / "visuals" / f"{dataset}-{index + 1:02d}"
                outputs = save_demo(destination, rgb, clear, label, demo["frozen"], demo["hard"], demo[MAIN], scene)
                exports[destination.name] = dict(outputs=outputs, size=list(rgb.shape[1::-1]), scene=scene)
            count += 1
            if count % 10 == 0:
                print(dict(dataset=dataset, evaluated=count), flush=True)
        if count != COUNTS.get(dataset, len(val)):
            raise ValueError("A registered evaluation image was missed")
        counts[dataset] = count
    overall, per_seed = summarize(matrices)
    table = ROOT / "confusions.json"
    atomic_write(table, records)
    report = dict(miou=overall, per_seed=per_seed, counts=counts, records=len(records),
                  confusions_SHA256=digest(table), training_SHA256=digest(ROOT / "completed.json"),
                  class_names=list(NAMES), segmentation=registration["model"], seed_display=SEEDS[0],
                  restored_checkpoint_SHA256=restorer_hash, visuals=exports, finished_utc=now())
    atomic_write(ROOT / "test.json", report)
    print(overall, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "test"))
    stage = parser.parse_args().stage
    test() if stage == "test" else fit(smoke=stage == "smoke")

"""Test whether tone-stable restoration evidence improves native 19-class fog segmentation."""
from boundary_refinement.data.source_integrity import matches_registered
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch import nn

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.models.segmentation.coarse_semantics import SPLIT, blend, boundary_ids, coarse_pair, head_features, native_labels, paint, partitions
from boundary_refinement.inference.dual_view_inference import HASHES, RELEASE, REVISION, load_b1
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import DRIVING, ZURICH, load_pairs, resize_image
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.models.restoration.factorial_restoration import ROOT as RGB, Restoration, predictions, trained as rgb_heads
from boundary_refinement.data.kitti_fog import MANIFEST as KITTI, load as kitti, render
from boundary_refinement.training.parallel_utility import capture_group
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.local_utility import SEEDS
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/tonal-agreement-v66"
MODES = ("raw_adapter", "utility_only", "concat", "average", "agreement", "shifted")
CASCADE = ("mcaf", "utility", "tone", "utility_tone", "published_mct")
REFERENCE = "raw_b1"
STEPS, CROP = 420, 64
RGB_SEED = SEEDS[0]
COHORTS = dict(foggy_zurich_testv2=40, foggy_driving_fine=33, kitti_light=36, kitti_dense=36)


class SemanticHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.raw = nn.Conv2d(channels, 24, kernel_size=1)
        self.layers = nn.Sequential(nn.Conv2d(62, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                                    nn.Conv2d(32, 19, kernel_size=1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, guidance, logits):
        update = self.layers(torch.cat((self.raw(features), guidance), dim=1))
        return logits + 3. * (update / 3.).tanh()


@dataclass
class Features:
    spatial: torch.Tensor
    raw: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor


def tone_intersection(before, after):
    minimum = torch.minimum(before.abs(), after.abs())
    return torch.where(before * after > 0., before.sign() * minimum, 0.)


def guidance(item, mode):
    before, after = item.before, item.after
    zero = torch.zeros_like(before)
    if mode == "raw_adapter":
        return torch.cat((zero, zero), dim=1)
    if mode == "utility_only":
        return torch.cat((before, zero), dim=1)
    if mode == "concat":
        return torch.cat((before, after), dim=1)
    center = (before + after) * .5 if mode == "average" else tone_intersection(before, after)
    pair = torch.cat((center, after - before), dim=1)
    if mode == "shifted":
        pair = pair.roll(tuple(x // 2 for x in pair.shape[-2:]), dims=(-2, -1))
    return pair


class DualOutput:
    def __init__(self):
        self.restorer = Restoration()
        self.heads = rgb_heads()
        self.seg, self.mean, self.std = load_b1()
        self.seg.requires_grad_(False)

    def segment(self, x):
        return self.seg(pixel_values=(x - self.mean) / self.std).logits.float()

    @torch.no_grad()
    def __call__(self, raw, extra=False):
        images = predictions(self.restorer(raw), self.heads, RGB_SEED)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features, original = head_features(self.seg, raw, self.mean, self.std)
            keys = CASCADE if extra else ("utility", "utility_tone")
            arms = {arm: self.segment(images[arm]) for arm in keys}
        d0, d1 = (arms[arm] - original for arm in ("utility", "utility_tone"))
        d0, d1 = ((delta - delta.mean(dim=1, keepdim=True)) / 3. for delta in (d0, d1))
        return Features(features.float(), original, d0, d1), arms, images["utility_tone"]


def registration(channels):
    return dict(code_SHA256=digest(__file__), rgb_code_SHA256=digest(SOURCE_ROOT / "boundary_refinement/models/restoration/factorial_restoration.py"),
                rgb_training_SHA256=digest(RGB / "completed.json"), rgb_seed=RGB_SEED,
                data={p.name: digest(p) for p in (SPLIT, KITTI, ZURICH, DRIVING)},
                cityscapes19=dict(model=RELEASE, revision=REVISION, hashes=HASHES),
                seeds=list(SEEDS), steps=STEPS, crop=CROP, modes=list(MODES), feature_channels=channels,
                parameters=sum(p.numel() for p in SemanticHead(channels).parameters()),
                design="Freeze MCAF RGB and B1 semantic weights. Keep the RGB output identical across semantic arms. "
                       "Use centered class-logit differences from raw to utility, before and after the learned tonal edit. "
                       "In the agreement arm preserve only the signed interval shared by both differences; retain the "
                       "signed before-after disagreement as a separate feature. Compare a raw head, one restoration "
                       "view, ordinary concatenation, mean with disagreement, and spatially displaced agreement. "
                       "Use the same training labels, parameters, updates, residual bounds and crop sequence. "
                       "No threshold or seed chosen using the evaluation splits.")


def cpu_item(value, target):
    down = F.interpolate(target[None, None].float(), value.raw.shape[-2:], mode="nearest-exact").long()[0, 0]
    packed = Features(*(a.to(device="cpu", dtype=torch.float16) for a in (value.spatial, value.raw, value.before, value.after)))
    return packed, down.cpu().byte()


def real_frames(rows):
    for row in rows:
        rgb, mask = coarse_pair(row)
        x, _ = resize_image(rgb)
        yield x, torch.from_numpy(mask.astype(np.int64, copy=False)).cuda()


def rendered_samples(row):
    clear = row["rgb"][None].cuda().float() / 255.
    fog, _, _ = render(clear, row["distance"][None].cuda().float(), row["role"], row["id"])
    return fog[1:].split(1), row["label"].cuda().long()


@torch.no_grad()
def cache(engine, smoke):
    fit, _ = partitions()
    real = [cpu_item(engine(x)[0], labels) for x, labels in real_frames(fit[:2] if smoke else fit)]
    synthetic = []
    for row in (r for r in kitti() if r["role"] == "fit"):
        xs, labels = rendered_samples(row)
        synthetic.extend(cpu_item(engine(x)[0], labels) for x in xs)
        if len(synthetic) % 48 == 0:
            print(dict(rendered_for_fit=len(synthetic)), flush=True)
        if smoke and len(synthetic) == 4:
            break
    if (len(real), len(synthetic)) != ((2, 4) if smoke else (50, 174)):
        raise ValueError("The segmentation training split changed")
    print(dict(real_fit=len(real), rendered_fit=len(synthetic)), flush=True)
    return real, synthetic


def crop(item, rng, mode):
    value, truth = item
    h, w = truth.shape
    sy, sx = min(CROP, h), min(CROP, w)
    y = int(torch.randint(h - sy + 1, (), generator=rng))
    x = int(torch.randint(w - sx + 1, (), generator=rng))
    if not bool((truth[y:y + sy, x:x + sx] != 255).any()):
        valid = (truth != 255).nonzero()
        if not len(valid):
            raise ValueError("A registered training frame has no labelled pixels")
        cy, cx = valid[torch.randint(len(valid), (), generator=rng)].tolist()
        y, x = min(h - sy, max(0, cy - sy // 2)), min(w - sx, max(0, cx - sx // 2))
    patch = (..., slice(y, y + sy), slice(x, x + sx))
    spatial = value.spatial[patch].cuda().float()
    raw = value.raw[patch].cuda().float()
    if mode == "shifted":
        iy = (torch.arange(y, y + sy) - h // 2) % h
        ix = (torch.arange(x, x + sx) - w // 2) % w
        shifted = (..., iy[:, None], ix[None, :])
        before, after = (a[shifted].cuda().float() for a in (value.before, value.after))
        evidence = guidance(Features(spatial, raw, before, after), "agreement")
    else:
        before, after = (a[patch].cuda().float() for a in (value.before, value.after))
        evidence = guidance(Features(spatial, raw, before, after), mode)
    return spatial, evidence, raw, truth[patch][None].cuda().long()


def fit(smoke):
    configure_cuda()
    engine = DualOutput()
    channels = engine.seg.decode_head.classifier.in_channels
    recipe = registration(channels)
    lock = ROOT / "protocol.json"
    if not smoke:
        if lock.exists() and read(lock) != recipe:
            raise ValueError("The recorded semantic protocol changed")
        atomic_write(lock, recipe)
    pool = cache(engine, smoke)
    rows = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            weights = ROOT / f"{mode}-{seed}.pth"
            ledger = weights.with_suffix(".json")
            if not smoke and ledger.exists():
                row = read(ledger)
                if digest(weights) != row["weights_SHA256"] or digest(lock) != row["protocol_SHA256"]:
                    raise ValueError(f"The recorded segmentation run changed {mode}/{seed}")
                rows.append(row)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 66173)
            model = SemanticHead(channels).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=.00055, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00005)
            for step in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                losses = []
                for domain in pool:
                    for index in torch.randint(len(domain), (2,), generator=rng).tolist():
                        features, cue, y0, label = crop(domain[index], rng, mode)
                        output = model(features, cue, y0)
                        valid = label != 255
                        penalty = (output - y0).square().mean(dim=1)[valid].mean()
                        losses.append(F.cross_entropy(output, label, ignore_index=255) + .018 * penalty)
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                opt.step()
                schedule.step()
            row = dict(seed=seed, method=mode, steps=step + 1, loss=float(loss.detach()))
            if not smoke:
                torch_save(weights, {k: v.detach().cpu() for k, v in model.state_dict().items()})
                row.update(weights_SHA256=digest(weights), protocol_SHA256=digest(lock))
                atomic_write(ledger, row)
                rows.append(row)
            print(row, flush=True)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=rows, steps=sum(r["steps"] for r in rows),
                     gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                     protocol_SHA256=digest(lock), finished_utc=now()))


def trained():
    completed = read(ROOT / "completed.json")
    recipe = ROOT / "protocol.json"
    if digest(recipe) != completed["protocol_SHA256"] or not matches_registered(read(recipe)['code_SHA256'], digest(__file__)):
        raise ValueError("The tonal agreement configuration changed")
    channels = read(recipe)["feature_channels"]
    result = {}
    for row in completed["runs"]:
        method, seed = row["method"], row["seed"]
        path = ROOT / f"{method}-{seed}.pth"
        if digest(path) != row["weights_SHA256"]:
            raise ValueError(f"The recorded semantic checkpoint changed: {path.name}")
        module = SemanticHead(channels).cuda().eval().requires_grad_(False)
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        result[method, seed] = module
    if set(result) != {(m, seed) for m in MODES for seed in SEEDS}:
        raise ValueError("Semantic training runs are incomplete")
    return result


def population():
    for data in ("foggy_zurich_testv2", "foggy_driving_fine"):
        for key, group, src, labels in load_pairs(data):
            x, _ = resize_image(src)
            yield data, key, group or capture_group(key), x, torch.from_numpy(labels.astype(np.int64, copy=False)).cuda(), src
    for row in kitti():
        if row["role"] == "held":
            xs, labels = rendered_samples(row)
            for tag, x in zip(("kitti_light", "kitti_dense"), xs, strict=True):
                yield tag, row["id"], row["drive"], x, labels, to_rgb(x)


def pooled_iou(cm):
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    union = cm.sum(axis=-2) + cm.sum(axis=-1) - diagonal
    ratios = np.divide(diagonal, union, out=np.zeros_like(diagonal, dtype=float), where=union != 0)
    return 100. * ratios.sum(axis=-1) / (union != 0).sum(axis=-1).clip(min=1)


def summarize(rows):
    scopes, scenes = defaultdict(list), defaultdict(list)
    for r in rows:
        ids = (r["dataset"], r["method"], r["seed"])
        scopes[ids].append(r)
        scenes[(*ids, r["group"])].append(np.asarray(r["counts"], dtype=np.int64))
    scores = defaultdict(list)
    for (d, method, seed), items in scopes.items():
        cm = np.sum([r["counts"] for r in items], axis=0, dtype=np.int64)
        edges = np.sum([r["edge_counts"] for r in items], axis=0, dtype=np.int64)
        scores[d, method].append(dict(seed=seed, miou=100. * segmentation_metrics(torch.from_numpy(cm))["miou"],
                                     edge_miou=100. * segmentation_metrics(torch.from_numpy(edges))["miou"]))
    summary = {"/".join(key): {"miou": mean(r["miou"] for r in values),
                              "edge_miou": mean(r["edge_miou"] for r in values), "by_seed": values}
               for key, values in scores.items()}
    intervals = {}
    for data in COHORTS:
        groups = sorted({r["group"] for r in rows if r["dataset"] == data})
        rng = np.random.default_rng(66007)
        sampled = rng.multinomial(len(groups), np.full(len(groups), 1. / len(groups)), size=3000)
        estimates = {}
        for method in (*MODES, REFERENCE, "utility_tone"):
            seeds = SEEDS if method in MODES else (0,)
            gs = np.asarray([[np.sum(scenes[data, method, seed, group], axis=0, dtype=np.int64) for seed in seeds]
                             for group in groups])
            matrices = (sampled @ gs.reshape(len(groups), -1)).reshape((-1, len(seeds), 19, 19))
            estimates[method] = pooled_iou(matrices).mean(axis=1)
        for baseline in (REFERENCE, "raw_adapter", "utility_only", "concat", "average", "shifted", "utility_tone"):
            change = estimates["agreement"] - estimates[baseline]
            point = summary[f"{data}/agreement"]["miou"] - summary[f"{data}/{baseline}"]["miou"]
            intervals[f"{data}/agreement-vs-{baseline}"] = dict(diff=point, ci_95=np.quantile(change, (.025, .975)).tolist(),
                                                                  groups=len(groups))
    return summary, intervals


@torch.inference_mode()
def test():
    configure_cuda()
    engine, bank = DualOutput(), trained()
    lock = ROOT / "evaluation_protocol.json"
    atomic_write(lock, dict(training_SHA256=digest(ROOT / "completed.json"), source_SHA256=digest(__file__),
                cohorts=COHORTS, metrics="Native-grid pooled mIoU and boundary mIoU, then averaged over matched seeds. "
                "Paired capture-group bootstrap uses the pooled confusion matrices. "
                "Foggy Zurich and Driving Fine are repeat development benchmarks; KITTI pairs share six held drive groups."))
    seen, rows, panels = Counter(), [], []
    for data, key, group, x, y, src in population():
        inputs, serial, dehazed = engine(x, extra=True)
        methods = {(tag, 0): value for tag, value in {REFERENCE: inputs.raw, **serial}.items()}
        for mode in MODES:
            cue = guidance(inputs, mode)
            for seed in SEEDS:
                methods[mode, seed] = bank[mode, seed](inputs.spatial, cue, inputs.raw)
        if not all(bool(torch.isfinite(logit).all()) for logit in methods.values()):
            raise ValueError(f"Non-finite semantic predictions for {data}/{key}")
        edge = boundary_ids(y)
        masks = {name: native_labels(result, y.shape) for name, result in methods.items()}
        for (mode, seed), pred in masks.items():
            rows.append(dict(dataset=data, id=key, group=group, method=mode, seed=seed,
                             counts=confusion_matrix(pred, y, 19).tolist(), edge_counts=confusion_matrix(pred, edge, 19).tolist()))
        if seen[data] in (0, 10):
            masks_to_show = ((REFERENCE, 0), ("utility_tone", 0), ("raw_adapter", SEEDS[0]),
                             ("concat", SEEDS[0]), ("agreement", SEEDS[0]))
            renderings = [blend(src, masks[name].cpu().numpy().astype(np.uint8)) for name in masks_to_show]
            panels.append((f"{data} / {key}", [src, to_rgb(dehazed), *renderings, paint(y.byte().cpu().numpy())],
                           ("Raw image", "Restored RGB", "Raw B1", "Restore then B1", "Raw adapter",
                            "Concat adapter", "Tone agreement", "Reference")))
        seen[data] += 1
        if seen[data] == COHORTS[data]:
            print(dict(dataset=data, images=seen[data]), flush=True)
    if seen != COHORTS:
        raise ValueError(f"Unexpected labelled cohorts: {seen}")
    metrics, ci = summarize(rows)
    table, picture = ROOT / "individual.json", ROOT / "predictions.png"
    atomic_write(table, rows)
    draw(picture, panels)
    atomic_write(ROOT / "test.json", dict(counts=dict(seen), rows=len(rows), metrics=metrics, contrasts=ci,
                 table_SHA256=digest(table), picture_SHA256=digest(picture), protocol_SHA256=digest(lock), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in metrics.items()}, flush=True)


@torch.inference_mode()
def infer(path):
    configure_cuda()
    engine, bank = DualOutput(), trained()
    output = ROOT / "native-demo"
    with Image.open(path) as photo:
        source = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
    raw, _ = resize_image(source)
    native = torch.from_numpy(source.copy()).permute(2, 0, 1)[None].cuda().float() / 255.
    cue, _, _ = engine(raw)
    native_rgb = predictions(engine.restorer(native), engine.heads, RGB_SEED)["utility_tone"]
    logits = bank["agreement", SEEDS[0]](cue.spatial, guidance(cue, "agreement"), cue.raw)
    raw_mask, adapted = [native_labels(z, source.shape[:2]).byte().cpu().numpy() for z in (cue.raw, logits)]
    if native_rgb.shape[-2:] != source.shape[:2] or adapted.shape != source.shape[:2] or int(adapted.max()) >= 19:
        raise ValueError("Native dual outputs have invalid geometry or labels")
    assets = {"raw": source, "restored": to_rgb(native_rgb), "mask": adapted, "color_mask": paint(adapted),
              "raw_overlay": blend(source, raw_mask), "agreement_overlay": blend(source, adapted)}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in assets.items():
        Image.fromarray(value).save(output / f"{name}.png")
    draw(output / "comparison.png", [(path.name, [assets[k] for k in ("raw", "restored", "raw_overlay", "agreement_overlay")],
                                       ("Raw", "Restored RGB", "Raw B1 mask", "Tone agreement mask"))])
    atomic_write(output / "metadata.json", dict(input_SHA256=digest(path), size=list(adapted.shape[::-1]),
                 changed_pixels=int(np.count_nonzero(adapted != raw_mask)),
                 files={p.name: digest(p) for p in sorted(output.glob("*.png"))}, finished_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "test", "infer"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.stage == "test":
        test()
    elif args.stage == "infer":
        if args.input is None or args.input.parent.resolve() == (ROOT / "native-demo").resolve():
            parser.error("Supply an input outside the native-demo output directory")
        infer(args.input)
    else:
        fit(args.stage == "smoke")

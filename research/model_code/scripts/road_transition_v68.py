"""Fit a road-specific restoration anchor and learn the risks of semantic label transitions."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import AUTHOR_HASH, AUTHOR_RELEASE, author_loader
from scripts.coarse_semantics_v39 import blend, native_labels, paint, partitions
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_parallel_rgb_v8 import bind
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, resize_image, verified_image
from scripts.evaluate_shared_haze_field_v40 import quality, to_rgb
from scripts.factorial_restoration_v65 import (
    ColorHead, MixtureHead, adjustment, mix_loss, mix_weights, mixture_sample, tonal_context, tonal_sample, tone_loss
)
from scripts.kitti_fog_v40 import MANIFEST as KITTI, load as kitti
from scripts.overlap_quadratic_v53 import Geometry, Partition, decoded
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.region_restore_v48 import SEEDS, view_size
from scripts.stage_muses_v67 import INDEX as MUSES_INDEX, MANIFEST as MUSES
from scripts.tonal_agreement_v66 import COHORTS, population, real_frames, rendered_samples
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/road-transition-v68"
RGB_ARMS = ("direct", "utility", "base_tone", "utility_tone")
SEM_ARMS = ("generic", "transition", "shifted")
SEM_VIEWS = ("mct", "utility", "utility_tone")
RGB_STEPS, SEM_STEPS = 250, 320
MARGIN = .08


def load_source():
    bind()
    return load_author_reference()


@torch.no_grad()
def geometry(raw, author):
    baseline = author_rgb(author, raw).clamp(0., 1.)
    flip = author_rgb(author, raw.flip(-1)).flip(-1).clamp(0., 1.)
    small = resize(raw, tuple(max(16, int(n * .75)) for n in raw.shape[-2:]), antialias=True)
    coarse = author_rgb(author, small).clamp(0., 1.)
    lifted = (raw + resize(coarse - small, raw.shape[-2:])).clamp(0., 1.)
    deltas = torch.cat((flip - baseline, lifted - baseline, raw - baseline))
    p = Partition(raw, view_size(raw, 12, 4))
    gram = p.average((deltas[:, None] * deltas[None, :]).mean(2)).permute(2, 3, 0, 1)[None]
    rms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
    corr = (gram / (rms[..., :, None] * rms[..., None, :])).clamp(-1., 1.)
    correlations = torch.stack([corr[..., a, b] for a, b in ((0, 1), (0, 2), (1, 2))], dim=1)
    color = p.average(torch.cat((2. * raw - 1., 2. * baseline - 1., *(4. * d[None] for d in deltas)), dim=1))
    contrast = (p.average(raw.square()) - p.average(raw).square()).clamp_min(0.).sqrt()
    cues = torch.cat((color, 4. * rms.permute(0, 3, 1, 2), correlations, 4. * contrast), dim=1)
    return Geometry(cues, gram, rms, p, baseline, deltas)


def render_images(g, weights, seed):
    images = {"mct": g.base}
    for arm in ("direct", "utility"):
        a, _ = mix_weights(g, weights[arm, seed], arm)
        images[arm] = decoded(g, a)
    for anchor in ("mct", "utility"):
        name = "base_tone" if anchor == "mct" else "utility_tone"
        images[name] = adjustment(tonal_context(g, images[anchor]), weights[name, seed], "local").clamp(0., 1.)
    return images


class TransitionHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(63, 28, 3, padding=1), nn.GroupNorm(4, 28), nn.SiLU(),
                         nn.Conv2d(28, 28, 3, padding=1), nn.GroupNorm(4, 28), nn.SiLU(), nn.Conv2d(28, 6, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)


@dataclass
class Guidance:
    logits: torch.Tensor
    features: torch.Tensor


class Engine:
    def __init__(self):
        self.rgb, self.rgb_release = load_source()
        self.bank = load_trained("rgb")
        author, self.semantic, self.lid_source = author_loader()
        self.mean, self.std = (torch.from_numpy(k).cuda()[None, :, None, None] for k in (author.MEAN, author.STD))

    @torch.no_grad()
    def __call__(self, image):
        pictures = render_images(geometry(image, self.rgb), self.bank, SEEDS[0])
        keys = ("raw", *SEM_VIEWS)
        views = {"raw": image, **pictures}
        # The original release operates on native geometry and keeps all Cityscapes train IDs.
        logits = {k: self.semantic.predict((views[k] - self.mean) / self.std).float() for k in keys}
        side = tuple(max(1, n // 8) for n in image.shape[-2:])
        low = torch.cat([resize(logits[k], side) for k in ("raw", "utility", "utility_tone")])
        centered = (low - low.mean(dim=1, keepdim=True)).clamp(-18., 18.) / 4.
        cues = torch.cat((centered[:1], (centered[1:] - centered[:1]).flatten(0, 1)[None],
                          resize(torch.cat([4. * (views[k] - image) for k in ("utility", "utility_tone")], dim=1), side)), dim=1)
        return Guidance(low, cues), logits, pictures


def declare(stage):
    value = dict(code_SHA256=digest(__file__), common_rgb_SHA256=digest(REPO / "scripts/factorial_restoration_v65.py"),
                 seeds=list(SEEDS), torch=str(torch.__version__), data={p.name: digest(p) for p in (KITTI, DRIVING, ZURICH)})
    if stage == "rgb":
        value.update(steps=RGB_STEPS, arms=list(RGB_ARMS), design="Fit on 87 KITTI fit images x two synthetic fog types. "
                     "Start both algorithms from the published MCT with three same-sized candidate offsets (horizontal "
                     "flip, 75% resolution residual, raw). Use the same RGB objectives as V65. Fit tone on MCT and on "
                     "the estimated local utility output. No test or coarse segmentation masks used.")
    else:
        value.update(steps=SEM_STEPS, arms=list(SEM_ARMS), rgb_SHA256=digest(ROOT / "rgb-completed.json"),
                     author=dict(release=AUTHOR_RELEASE, hash=AUTHOR_HASH), probability_margin=MARGIN,
                     design="Freeze all RGB and official LiDSeg networks. Compare equally sized ordinary NLL mixture "
                     "with supervision of rescue and harm for pixels on which each restored view disagrees with raw. "
                     "For each candidate fit probabilities of neither, rescue and harm using pixel labels. Switch only "
                     "when estimated rescue minus harm beats 0.08. In the shifted control roll the cue by half the field. "
                     "Use the same 50 coarse real images and 174 synthetic views. MUSES is held out from all fitting.")
    return value


def plan(stage, smoke):
    path = ROOT / f"{stage}-protocol.json"
    value = declare(stage)
    if not smoke:
        if path.exists() and read(path) != value:
            raise ValueError(f"Provenance changed: {stage}")
        atomic_write(path, value)
    return path


def train_head(stage, arm, seed, proto, module, objective, steps, smoke):
    path = ROOT / f"{stage}-{arm}-{seed}.pth"
    log = path.with_suffix(".json")
    if not smoke and log.exists():
        rec = read(log)
        if rec["SHA256"] != digest(path) or rec["protocol_SHA256"] != digest(proto):
            raise ValueError(f"Training record changed: {path.name}")
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return module.eval().requires_grad_(False), rec
    rng = torch.Generator().manual_seed(seed + 68027)
    opt = torch.optim.AdamW(module.parameters(), lr=.0008, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.00008)
    for i in range(2 if smoke else steps):
        opt.zero_grad(set_to_none=True)
        loss = objective(module, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    rec = dict(arm=arm, seed=seed, steps=i + 1, loss=float(loss.detach()), parameters=sum(x.numel() for x in module.parameters()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in module.state_dict().items()})
        rec.update(SHA256=digest(path), protocol_SHA256=digest(proto))
        atomic_write(log, rec)
    print(rec, flush=True)
    return module.eval().requires_grad_(False), rec


def batch_loss(fn, pool, mode):
    def loss(model, random):
        selected = torch.randint(len(pool), (3,), generator=random).tolist()
        return torch.stack([fn(model, pool[j], mode) for j in selected]).mean()
    return loss


def fit_rgb(smoke):
    configure_cuda()
    reference, author = load_source()
    proto = plan("rgb", smoke)
    pairs = []
    for example in (r for r in kitti() if r["role"] == "fit"):
        clear = example["rgb"][None].cuda().float() / 255.
        for fog in rendered_samples(example)[0]:
            pairs.append((geometry(fog, reference), clear))
        if smoke and len(pairs) == 4:
            break
    if len(pairs) != (4 if smoke else 174):
        raise ValueError("Road RGB fit is incomplete")
    print(dict(rgb_fitting_views=len(pairs)), flush=True)
    pool = [mixture_sample(g, y) for g, y in pairs]
    records = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for arm in RGB_ARMS[:2]:
            set_seed(seed)
            net, record = train_head("rgb", arm, seed, proto, MixtureHead().cuda(), batch_loss(mix_loss, pool, arm), RGB_STEPS, smoke)
            records.append(record)
        for arm in RGB_ARMS[2:]:
            samples = [tonal_sample(g, g.base if arm == "base_tone" else decoded(g, mix_weights(g, net, "utility")[0]), y)
                       for g, y in pairs]
            set_seed(seed)
            _, record = train_head("rgb", arm, seed, proto, ColorHead().cuda(), batch_loss(tone_loss, samples, "local"), RGB_STEPS, smoke)
            records.append(record)
    if not smoke:
        atomic_write(ROOT / "rgb-completed.json", dict(runs=records, protocol_SHA256=digest(proto), author=author, finished_utc=now()))


def warp(cue):
    return cue.roll(tuple(n // 2 for n in cue.shape[-2:]), dims=(-2, -1))


def transition_cost(net, record, mode):
    item, labels = record
    logits = item.logits.cuda().float()
    features = item.features.cuda().float()
    labels = labels.cuda().long()
    expected = net(warp(features) if mode == "shifted" else features).unflatten(1, (2, 3))[0]
    if mode == "generic":
        move = expected[:, 1] - expected[:, 2] - .5 * expected[:, 0]
        weights = torch.cat((torch.zeros_like(move[:1]), move)).softmax(0)
        probabilities = (weights[:, None] * logits.softmax(1)).sum(0)
        return F.nll_loss(probabilities.clamp_min(1e-8).log()[None], labels[None], ignore_index=255)
    predicted = logits.argmax(1)
    real = labels != 255
    changed = (predicted[1:] != predicted[:1]) & real
    helpful = (predicted[1:] == labels) & changed
    harmful = (predicted[:1] == labels) & changed
    targets = helpful.long() + 2 * harmful.long()
    events = F.cross_entropy(expected, targets, reduction="none")
    return (events * changed).sum() / changed.sum().clamp_min(1)


@torch.no_grad()
def prep_item(engine, raw, label):
    input, _, _ = engine(raw)
    target = F.interpolate(label[None, None].float(), input.features.shape[-2:], mode="nearest-exact")[0, 0].byte().cpu()
    if not bool((target != 255).any()):
        raise ValueError("No valid pixels at the fixed semantic fitting grid")
    return Guidance(input.logits.half().cpu(), input.features.half().cpu()), target


def fit_semantic(smoke):
    configure_cuda()
    engine = Engine()
    proto = plan("semantic", smoke)
    rows, _ = partitions()
    real = [prep_item(engine, x, y) for x, y in real_frames(rows[:2] if smoke else rows)]
    virtual = []
    for row in (r for r in kitti() if r["role"] == "fit"):
        views, y = rendered_samples(row)
        virtual.extend(prep_item(engine, x, y) for x in views)
        if len(virtual) % 48 == 0:
            print(dict(road_rendered=len(virtual)), flush=True)
        if smoke and len(virtual) == 4:
            break
    print(dict(road_real=len(real), road_rendered=len(virtual)), flush=True)
    if (len(real), len(virtual)) != ((2, 4) if smoke else (50, 174)):
        raise ValueError("The source segmentation training population changed")
    records = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for arm in SEM_ARMS:
            set_seed(seed)
            def objective(model, rng):
                rows = [domain[int(torch.randint(len(domain), (), generator=rng))] for domain in (real, virtual)]
                return torch.stack([transition_cost(model, row, arm) for row in rows]).mean()

            _, rec = train_head("semantic", arm, seed, proto, TransitionHead().cuda(), objective, SEM_STEPS, smoke)
            records.append(rec)
    if not smoke:
        atomic_write(ROOT / "semantic-completed.json", dict(runs=records, protocol_SHA256=digest(proto), finished_utc=now()))


def load_trained(stage):
    data = read(ROOT / f"{stage}-completed.json")
    protocol = ROOT / f"{stage}-protocol.json"
    if digest(protocol) != data["protocol_SHA256"] or read(protocol)["code_SHA256"] != digest(__file__):
        raise ValueError(f"The registered experiment changed: {stage}")
    result = {}
    for note in data["runs"]:
        arm, seed = note["arm"], note["seed"]
        path = ROOT / f"{stage}-{arm}-{seed}.pth"
        if note["SHA256"] != digest(path):
            raise ValueError(f"Weights changed: {path.name}")
        cls = TransitionHead if stage == "semantic" else MixtureHead if arm in RGB_ARMS[:2] else ColorHead
        model = cls().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        result[arm, seed] = model
    if set(result) != {(m, seed) for m in (SEM_ARMS if stage == "semantic" else RGB_ARMS) for seed in SEEDS}:
        raise ValueError(f"Incomplete weights: {stage}")
    return result


def measure_switch(provisional, item, logits, arms, model, shape):
    z = model(warp(item.features) if arms == "shifted" else item.features).unflatten(1, (2, 3))[0]
    if arms == "generic":
        move = z[:, 1] - z[:, 2] - .5 * z[:, 0]
        modes = torch.cat((torch.zeros_like(move[:1]), move)).softmax(0)
        image_prob = torch.cat([resize(logits[k], item.features.shape[-2:]).softmax(1) for k in ("raw", "utility", "utility_tone")])
        return native_labels((modes[:, None] * image_prob).sum(0, keepdim=True), shape)
    q = z.softmax(1)
    advantage = resize((q[:, 1] - q[:, 2])[None], shape)[0]
    desired = torch.stack([provisional[key] for key in ("utility", "utility_tone")])
    allowed = desired != provisional["raw"]
    strength, where = advantage.masked_fill(~allowed, -1.).max(0)
    selection = desired.gather(0, where[None])[0]
    return torch.where(strength > MARGIN, selection, provisional["raw"])


def muses_cases():
    record = read(MUSES)
    if record["index_SHA256"] != digest(MUSES_INDEX):
        raise ValueError("External evaluation registration changed")
    for row in record["pairs"]:
        a, b = row["frame_camera"], row["gt_semantic"]
        raw = verified_image(a["file"], a["SHA256"], False)
        mask = verified_image(b["file"], b["SHA256"], True)
        fog, _ = resize_image(raw)
        yield "muses_fog_val", row["id"], row["group"], fog, torch.from_numpy(mask.astype(np.int64)).cuda(), raw


def semantic_metrics(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["dataset"], row["arm"], row["seed"]].append(np.asarray(row["confusion"], dtype=np.int64))
    cells = defaultdict(list)
    for (d, arm, _), group in buckets.items():
        score = segmentation_metrics(torch.from_numpy(np.sum(group, axis=0, dtype=np.int64)))
        cells[d, arm].append(score["miou"] * 100.)
    return {"/".join(k): dict(miou=mean(s), by_seed=s) for k, s in cells.items()}


@torch.inference_mode()
def test(external):
    configure_cuda()
    engine, bank = Engine(), load_trained("semantic")
    trials = COHORTS if not external else {"muses_fog_val": len(read(MUSES)["pairs"])}
    dataset_card = digest(MUSES) if external else None
    suffix = "muses" if external else "development"
    protocol = ROOT / f"{suffix}-evaluation-protocol.json"
    atomic_write(protocol, dict(weights_SHA256=digest(ROOT / "semantic-completed.json"), trials=trials,
                 muses_manifest_SHA256=dataset_card, confidence_margin=MARGIN,
                 score="Pooled native-grid macro IoU over 19 classes. Both semantic directions compare on the same weights."))
    rows, panels, seen = [], [], Counter()
    for data, key, group, raw, truth, src in muses_cases() if external else population():
        item, logits, pictures = engine(raw)
        original = {tag: native_labels(value, truth.shape) for tag, value in logits.items()}
        labels = {(tag, 0): pred for tag, pred in original.items()}
        serial_oracle = torch.where((original["utility"] == truth) | (original["utility_tone"] == truth), truth, original["raw"])
        labels["pixel_oracle", 0] = serial_oracle
        for (arm, seed), model in bank.items():
            labels[arm, seed] = measure_switch(original, item, logits, arm, model, truth.shape)
        for (arm, seed), mask in labels.items():
            real = truth != 255
            good, start = (mask == truth) & real, (labels["raw", 0] == truth) & real
            rows.append(dict(dataset=data, id=key, group=group, arm=arm, seed=seed,
                             rescued=int((good & ~start).sum()), harmed=int((start & ~good).sum()),
                             confusion=confusion_matrix(mask, truth, 19).tolist()))
        if seen[data] == 0:
            overlay = [blend(src, labels[arm, seed].byte().cpu().numpy()) for arm, seed in
                       (("raw", 0), ("utility_tone", 0), ("generic", SEEDS[0]), ("transition", SEEDS[0]))]
            panels.append((f"{data} / {key}", [src, to_rgb(pictures["utility_tone"]), *overlay, paint(truth.byte().cpu().numpy())],
                           ("Raw image", "Restored RGB", "LiDSeg on raw", "Cascade", "Generic", "Transition", "Reference")))
        seen[data] += 1
    if seen != trials:
        raise ValueError(f"Incomplete road transition evaluation: {seen}")
    table, picture = ROOT / f"{suffix}-individual.json", ROOT / f"{suffix}-predictions.png"
    atomic_write(table, rows)
    draw(picture, panels)
    scores = semantic_metrics(rows)
    atomic_write(ROOT / f"{suffix}-test.json", dict(metrics=scores, rows=len(rows), observations=dict(seen),
                 table_SHA256=digest(table), picture_SHA256=digest(picture), protocol_SHA256=digest(protocol), finished_utc=now()))
    print({k: round(v["miou"], 3) for k, v in scores.items()}, flush=True)


@torch.inference_mode()
def paired_rgb():
    configure_cuda()
    mct, _ = load_source()
    bank = load_trained("rgb")
    rows = []
    for entry in kitti():
        if entry["role"] != "held":
            continue
        clear = entry["rgb"][None].cuda().float() / 255.
        for kind, raw in zip(("kitti_light", "kitti_dense"), rendered_samples(entry)[0], strict=True):
            g = geometry(raw, mct)
            for seed in SEEDS:
                for key, image in render_images(g, bank, seed).items():
                    rows.append(dict(dataset=kind, id=entry["id"], group=entry["drive"], arm=key, seed=seed, **quality(image, clear)))
    cases = Counter((r["dataset"], r["arm"], r["seed"]) for r in rows)
    if len(cases) != 2 * (len(RGB_ARMS) + 1) * len(SEEDS) or set(cases.values()) != {36}:
        raise ValueError("Missing joint KITTI RGB results")
    by = defaultdict(list)
    for row in rows:
        by[row["dataset"], row["arm"]].append(row["psnr"])
    output = ROOT / "kitti-rgb-individual.json"
    atomic_write(output, rows)
    result = {"/".join(k): mean(v) for k, v in by.items()}
    atomic_write(ROOT / "kitti-rgb.json", dict(psnr=result, scored_frames=72, table_SHA256=digest(output), finished_utc=now()))
    print({k: round(v, 3) for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rgb-smoke", "rgb-fit", "semantic-smoke", "semantic-fit", "test", "muses", "rgb-test"))
    stage = parser.parse_args().stage
    if stage.startswith("rgb-") and stage != "rgb-test":
        fit_rgb(stage.endswith("smoke"))
    elif stage.startswith("semantic-"):
        fit_semantic(stage.endswith("smoke"))
    elif stage == "rgb-test":
        paired_rgb()
    else:
        test(external=stage == "muses")

"""Separate the semantic effect of dehazing from the effect of input resolution."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix
from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import AUTHOR_HASH, paired_iou_intervals
from scripts.class_protected_residual_v81 import class_examples, group_sample, soft_counts
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, resize_image
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import DATA, aggregate, paths
from scripts.restoration_affinity_v75 import boundary_band, compress_scores
from scripts.restoration_cue_v71 import match_channels
from scripts.road_transition_v68 import ROOT as ROAD, SEEDS as RGB_SEEDS, Engine, geometry, render_images
from scripts.soft_proposals_v79 import evaluation_cases
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/resolution-interaction-v84"
SEEDS = (101, 211, 307)
ARMS = ("raw_detail", "ordinary_fusion", "restored_gate", "interaction", "tone_interaction", "shift_interaction")
PRIMARY = "interaction"
STEPS = 480
CROP = 88
FINE_SIDE = 1536
SMALL_OBJECTS = (5, 6, 7, 11, 12, 17, 18)


@dataclass
class Views:
    # Position zero is the raw scene, one the restoration, and two the monotone RGB match.
    coarse: torch.Tensor
    fine: torch.Tensor

    def to(self, *args, **kwargs):
        return Views(self.coarse.to(*args, **kwargs), self.fine.to(*args, **kwargs))

    def cut(self, region):
        return Views(self.coarse[region], self.fine[region])


class DualScaleFeatures:
    def __init__(self):
        self.core = Engine()

    def logits(self, rgb):
        return self.core.semantic.predict((rgb - self.core.mean) / self.core.std).float()

    @torch.no_grad()
    def __call__(self, original):
        # The restored RGB output uses exactly the same initial renderer for every mask method.
        h, w = original.shape[:2]
        scale = min(1., FINE_SIDE / max(h, w))
        shape = (round(h * scale), round(w * scale))
        fine = F.interpolate(torch.from_numpy(original).cuda().permute(2, 0, 1)[None].float() / 255.,
                             shape, mode="bilinear", align_corners=False, antialias=True)
        coarse, _ = resize_image(original)
        g = geometry(fine, self.core.rgb)
        restored = render_images(g, self.core.bank, RGB_SEEDS[0])["utility_tone"]
        matched = match_channels(fine, restored)
        grid = tuple((x + 3) // 4 for x in shape)
        fine_scene = (fine, restored, matched)
        coarse_scene = (coarse, resize(restored, coarse.shape[-2:], antialias=True),
                        resize(matched, coarse.shape[-2:], antialias=True))
        f = torch.stack([resize(self.logits(image), grid) for image in fine_scene], dim=1)
        logits = tuple(self.logits(image) for image in coarse_scene)
        c = torch.stack([resize(z, grid) for z in logits], dim=1)
        return Views(c, f), restored, logits[0]

def centered(logits):
    return logits - logits.mean(dim=1, keepdim=True)


def response(views, arm):
    raw, raw_fine = views.coarse[:, 0], views.fine[:, 0]
    detail = centered(raw_fine - raw)
    idx = 2 if arm == "tone_interaction" else 1
    coarse_delta = centered(views.coarse[:, idx] - raw)
    fine_delta = centered(views.fine[:, idx] - raw_fine)
    if arm == "raw_detail":
        coarse_delta = torch.zeros_like(detail)
        fine_delta = torch.zeros_like(detail)
    if arm == "shift_interaction":
        shift = tuple(n // 2 for n in detail.shape[-2:])
        coarse_delta = coarse_delta.roll(shift, dims=(-2, -1))
        fine_delta = fine_delta.roll(shift, dims=(-2, -1))
    feature = torch.cat((centered(raw), detail, coarse_delta, fine_delta), dim=1).clamp(-24., 24.) / 4.
    if arm == "raw_detail":
        direction = detail
    elif arm == "restored_gate":
        direction = fine_delta
    else:
        direction = fine_delta - coarse_delta
    return .5 * (raw + raw_fine), feature, direction


class DetailHead(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(76, 40, 3, padding=1), nn.GroupNorm(5, 40), nn.SiLU(),
                         nn.Conv2d(40, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(), nn.Conv2d(32, 38, 1))
        nn.init.zeros_(self[-1].weight)
        nn.init.zeros_(self[-1].bias)

    def forward(self, feature, direction, arm):
        free, branch = super().forward(feature).chunk(2, dim=1)
        edit = 4. * (free / 4.).tanh()
        if arm == "ordinary_fusion":
            return edit + 3. * (branch / 3.).tanh()
        # The gate starts near zero while allowing a less restricted correction in each arm.
        return edit + (branch - 3.).sigmoid() * direction.clamp(-12., 12.)


def protocol():
    manifests = [DATA / f"{scope}.json" for scope in ("fit", "calibrate")]
    return dict(source_SHA256=digest(__file__), training={p.name: digest(p) for p in manifests},
                road_release_SHA256=digest(ROAD / "rgb-completed.json"), lidseg_SHA256=AUTHOR_HASH,
                steps=STEPS, seeds=list(SEEDS), arms=list(ARMS), crop=CROP, fine_side=FINE_SIDE,
                fit={"muses_train_frames": 96, "recordings": 9},
                design="Freeze the RGB renderer and LiDSeg. Use the same raw coarse/fine anchor, class-weighted CE plus patch IoU, "
                       "network shapes, sample draws and budget. Compare raw detail, unconstrained residual fusion "
                       "on all four logits, gating the restored-minus-raw fine logits, and gating the difference "
                       "of the fine and coarse restoration responses. Match color and roll the restoration cues "
                       "for additional controls. Select no strength on the evaluation cohorts. Compare original "
                       "resolution labels and record the full class confusion matrices.",
                primary_endpoints=["recording-paired mIoU vs raw_detail and ordinary_fusion",
                                   "same-feature attribution vs restored_gate, tone_interaction and shift_interaction"],
                reused_eval=["MUSES 186", "Foggy Zurich 40", "Foggy Driving fine 33/coarse 68"],
                small_class_ids=list(SMALL_OBJECTS))


@torch.no_grad()
def packets(engine, smoke):
    groups = defaultdict(list)
    counts = torch.zeros(19, dtype=torch.int64)
    for i, (_, group, _, _, truth, original) in enumerate(islice(paths("fit"), 6 if smoke else None), 1):
        views, _, _ = engine(original)
        label = F.interpolate(truth[None, None].float(), views.coarse.shape[-2:], mode="nearest-exact")[0, 0]
        label = label.byte().cpu()
        counts += torch.bincount(label[label < 19].long(), minlength=19)
        groups[group].append((views.to("cpu", dtype=torch.float16), label))
        if i % 16 == 0:
            print(dict(encoded=i), flush=True)
    if sum(map(len, groups.values())) != (6 if smoke else 96):
        raise ValueError("Unexpected resolution-interaction fit population")
    weight = ((counts.sum() / 19. + 1.) / (counts + 1.)).pow(.20).clamp(.65, 2.).cuda()
    rare = (counts > 0) & (counts < .02 * counts.sum())
    return tuple(groups.values()), weight, class_examples(groups.values(), rare)


def crop(frame, rng, point):
    views, truth = frame
    h, w = truth.shape
    size = min(CROP, h, w)
    if point is None:
        top, left = (int(torch.randint(n - size + 1, (), generator=rng)) for n in (h, w))
    else:
        dy, dx = torch.randint(max(1, size // 2), (2,), generator=rng).tolist()
        top, left = (max(0, min(n - size, p - size // 4 - d)) for n, p, d in zip((h, w), point, (dy, dx)))
    if not bool((truth[top:top + size, left:left + size] < 19).any()):
        visible = (truth < 19).nonzero()
        y, x = visible[int(torch.randint(len(visible), (), generator=rng))].tolist()
        top, left = min(max(0, y - size // 2), h - size), min(max(0, x - size // 2), w - size)
    row, col = slice(top, top + size), slice(left, left + size)
    return views.cut((..., row, col)).to("cuda", dtype=torch.float32), truth[row, col].cuda().long()


def signed_views(views):
    # The rolled null is formed on the full frame, so its spatial transformation stays identical at test time.
    shift = tuple(n // 2 for n in views.coarse.shape[-2:])
    coarse, fine = views.coarse.clone(), views.fine.clone()
    coarse[:, 1] = coarse[:, 0] + (coarse[:, 1] - coarse[:, 0]).roll(shift, dims=(-2, -1))
    fine[:, 1] = fine[:, 0] + (fine[:, 1] - fine[:, 0]).roll(shift, dims=(-2, -1))
    return Views(coarse, fine)


def batch(groups, positions, rng, arm):
    blocks = []
    for frame, point in group_sample(groups, positions, rng, False):
        views, truth = frame
        if arm == "shift_interaction":
            frame = signed_views(views), truth
        blocks.append(crop(frame, rng, point))
    side = min(min(b.shape) for _, b in blocks)
    tight = (..., slice(0, side), slice(0, side))
    levels = Views(*(torch.cat([getattr(a, key)[tight] for a, _ in blocks]) for key in ("coarse", "fine")))
    return levels, torch.stack([b[tight] for _, b in blocks])


def fit_one(arm, seed, groups, positions, weights, smoke):
    path = ROOT / f"{arm}-{seed}.pth"
    note = path.with_suffix(".json")
    registered = digest(ROOT / "protocol.json") if not smoke else None
    if not smoke and note.is_file():
        old = read(note)
        if old["weights_SHA256"] != digest(path) or old["protocol_SHA256"] != registered:
            raise ValueError(f"Mismatched finished fit {path.name}")
        return old
    set_seed(seed)
    net = DetailHead().cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS, eta_min=.00006)
    rng = torch.Generator().manual_seed(seed + 84001)
    steps = 4 if smoke else STEPS
    for _ in range(steps):
        views, truth = batch(groups, positions, rng, arm)
        response_arm = PRIMARY if arm == "shift_interaction" else arm
        anchor, context, direction = response(views, response_arm)
        edit = net(context, direction, arm)
        value = anchor + edit
        i, u, c, _ = soft_counts(value, truth)
        loss = F.cross_entropy(value, truth, weight=weights, ignore_index=255)
        loss += .8 * (1. - i[c > 0] / u[c > 0].clamp_min(1e-5)).mean() + .008 * edit.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
    item = dict(method=arm, seed=seed, steps=steps, params=sum(p.numel() for p in net.parameters()),
                loss=float(loss.detach()))
    if not smoke:
        torch_save(path, {k: v.detach().cpu() for k, v in net.state_dict().items()})
        item.update(weights_SHA256=digest(path), protocol_SHA256=registered)
        atomic_write(note, item)
    print(item, flush=True)
    return item


def fit(smoke):
    configure_cuda()
    lock = ROOT / "protocol.json"
    if not smoke:
        cfg = protocol()
        if lock.is_file() and read(lock) != cfg:
            raise ValueError("Resolution experiment protocol changed")
        atomic_write(lock, cfg)
    pool, weights, positions = packets(DualScaleFeatures(), smoke)
    runs = [fit_one(arm, s, pool, positions, weights, smoke)
            for s in (SEEDS[:1] if smoke else SEEDS) for arm in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=runs, steps=sum(x["steps"] for x in runs),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def load():
    ledger, lock = read(ROOT / "completed.json"), ROOT / "protocol.json"
    if ledger["protocol_SHA256"] != digest(lock) or read(lock) != protocol():
        raise ValueError("Resolution test requires the frozen experiment")
    available = {}
    for item in ledger["runs"]:
        key = item["method"], item["seed"]
        path = ROOT / f"{key[0]}-{key[1]}.pth"
        net = DetailHead().cuda().eval().requires_grad_(False)
        if digest(path) != item["weights_SHA256"] or item["steps"] != STEPS:
            raise ValueError(f"Unexpected state {path.name}")
        net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        if item["params"] != sum(p.numel() for p in net.parameters()):
            raise ValueError("Capacity match failed")
        available[key] = net
    if set(available) != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("Incomplete resolution experiment")
    return available


def selection(part):
    if part == "calibrate":
        for key, group, lighting, x, target, rgb in paths(part):
            yield "muses_calibrate", key, group, lighting, x, target, rgb
    else:
        yield from evaluation_cases("large", part)


def add_global_cohort(rows):
    return rows + [{**r, "dataset": "foggy_driving_101"} for r in rows if r["dataset"].startswith("foggy_driving_")]


def all_scores(rows):
    report = {}
    for name, part in (("full", rows), ("boundary", [{**r, "counts": r["edge"]} for r in rows if "edge" in r])):
        panel = add_global_cohort(part) if name == "full" else part
        contrast = [(PRIMARY, a) for a in ("author_raw", "raw_ensemble", *ARMS) if a != PRIMARY]
        report[name] = dict(metrics=aggregate(panel), intervals=paired_iou_intervals(panel, contrast, "method", "counts"))
    return report


def result_rows(views, author_raw, models, truth, dataset, key, group):
    shape = truth.shape
    edge = None if dataset == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
    all_responses = {method: response(views, method) for method in ARMS}
    base = all_responses[PRIMARY][0]
    reference = dict(author_raw=author_raw, raw_fine=views.fine[:, 0], raw_ensemble=base,
                     restored_fine=views.fine[:, 1], average_fusion=.5 * (base + views.fine[:, 1]))
    for seed in SEEDS:
        logits = reference | {method: a + models[method, seed](b, c, method)
                              for method, (a, b, c) in all_responses.items()}
        preds = {method: resize(value, shape).argmax(1)[0] for method, value in logits.items()}
        for method, pred in preds.items():
            row = dict(dataset=dataset, id=key, group=group, method=method, seed=seed,
                       counts=confusion_matrix(pred, truth, 19).tolist())
            if edge is not None:
                row["edge"] = confusion_matrix(pred, edge, 19).tolist()
            yield row, pred


@torch.inference_mode()
def score(part):
    configure_cuda()
    engine, nets = DualScaleFeatures(), load()
    rows, counts, panels = [], Counter(), {}
    for ds, key, group, lighting, _, truth, rgb in selection(part):
        views, restored, author_raw = engine(rgb)
        show = (ds, group) not in panels and sum(k[0] == ds for k in panels) < 4
        examples = {}
        pictures = ("author_raw", "raw_detail", "ordinary_fusion", "restored_gate", PRIMARY)
        for row, prediction in result_rows(views, author_raw, nets, truth, ds, key, group):
            rows.append(row)
            if show and row["seed"] == SEEDS[0] and row["method"] in pictures:
                examples[row["method"]] = blend(rgb, prediction.byte().cpu().numpy())
        if show:
            truth_panel = paint(truth.byte().cpu().numpy())
            panels[ds, group] = (f"{key} / {lighting or ds}", [rgb, to_rgb(restored),
                                 *(examples[tag] for tag in pictures), truth_panel],
                                 ("Fog", "Restored RGB", "LiDSeg", "Raw detail", "Ordinary fusion",
                                  "Restored gate", "Resolution interaction", "Reference"))
        counts[ds] += 1
        if sum(counts.values()) % 24 == 0:
            print(dict(part=part, evaluated=sum(counts.values())), flush=True)
    expected = {"calibrate": 24, "held": 186, "external": 141}
    if sum(counts.values()) != expected[part]:
        raise ValueError(f"Incomplete resolution evaluation {counts}")
    file, preview = ROOT / f"{part}-individual.json.gz", ROOT / f"{part}-predictions.png"
    compress_scores(file, rows)
    draw(preview, list(panels.values()))
    report = all_scores(rows)
    atomic_write(ROOT / f"{part}-test.json", dict(counts=dict(counts), **report, rows_SHA256=digest(file),
                 picture_SHA256=digest(preview), weights_SHA256=digest(ROOT / "completed.json"), utc=now(),
                 external_manifests_SHA256={k.name: digest(k) for k in (DRIVING, ZURICH)}))
    print({name: round(result["miou"], 4) for name, result in report["full"]["metrics"].items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "calibrate", "held", "external"))
    args = parser.parse_args()
    if args.stage in ("fit", "smoke"):
        fit(args.stage == "smoke")
    else:
        score(args.stage)

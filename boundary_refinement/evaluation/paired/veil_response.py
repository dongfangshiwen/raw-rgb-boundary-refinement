"""Use photo-intervention response as a cue for two frozen restorers."""
from boundary_refinement.paths import SOURCE_ROOT, record_path
import argparse
from collections import defaultdict
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import pgl_rgb
from boundary_refinement.models.segmentation.coarse_semantics import rgb_checkpoint
from boundary_refinement.training.crop_context import originals
from boundary_refinement.evaluation.photographic.evaluate_cross_scale_rgb import fresh_revide
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_paired_basis import validation_source
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.photographic.evaluate_selective_adaptation import samples as original_rw
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import quality, to_rgb
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.training.haze_response import linear_rgb, srgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_haze_data import ROOT as NH_ROOT, load_data
from boundary_refinement.training.source_replay import PROTOCOL as RW_FIT
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD
from boundary_refinement.data.staging.stage_revide_remaining import MANIFEST, ROOT, pairs
from boundary_refinement.training.structure_transfer import SEEDS, training_scenes
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ARMS = ("resolution", "exposure", "veil")
MAIN = "veil"
STEPS = 480
AMOUNTS = (.25, .5, .75, 1.)
FLOOR = -.25
PREVIEW = 256
NH_RESERVED = tuple(f"{i:02d}" for i in range(5, 46, 5))
CHECK = ROOT / "protocol.json"


def scaled(x, long):
    shape = tuple(max(32, round(v * long / max(x.shape[-2:]))) for v in x.shape[-2:])
    return F.interpolate(x, shape, mode="bilinear", align_corners=False, antialias=True)


def interventions(x):
    light = linear_rgb(x)
    luminance = light.mean(1).flatten(1)
    selected = luminance.topk(max(8, round(.02 * luminance.shape[1])), dim=1).indices
    air = light.flatten(2).gather(2, selected[:, None].expand(-1, 3, -1)).mean(-1)[:, :, None, None]
    veiled = light.lerp(air, .18)
    magnitude = (light - veiled).square().mean((1, 2, 3), keepdim=True).sqrt()
    exposure = (magnitude / light.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)).clamp_max(.75)
    return srgb(veiled).clamp(0, 1), srgb(light * (1. - exposure)).clamp(0, 1)


def moments(x):
    contrast = x.flatten(2).std(2, unbiased=False)
    grad = .5 * (x.diff(dim=-1).abs().mean((2, 3)) + x.diff(dim=-2).abs().mean((2, 3)))
    return torch.cat((x.mean((2, 3)), contrast, grad), dim=1)


def respond(x, references, altered):
    denom = (x.square().mean().sqrt() + .01)
    changes = [value - before for value, before in zip(altered, references, strict=True)]
    return torch.cat((moments(changes[0] / denom), moments(changes[1] / denom),
                      moments((changes[0] - changes[1]) / denom)), dim=1)


def get_experts():
    networks, source = experts()
    anchor, baseline_hash = rgb_checkpoint()
    return (partial(pgl_rgb, anchor), networks[0]), dict(author=source, anchor_SHA256=baseline_hash)


@torch.inference_mode()
def describe(x, models, modes=ARMS):
    previews = scaled(x, min(PREVIEW, max(x.shape[-2:])))
    standard = tuple(restorer(previews).clamp(0., 1.) for restorer in models)
    basic = torch.cat((moments(previews), moments(standard[0]), moments(standard[1]),
                       moments(standard[1] - standard[0])), dim=1)
    fog, exposure = interventions(previews)
    results = {}
    inputs = dict(resolution=scaled(previews, 192), exposure=exposure, veil=fog)
    for label in modes:
        image = inputs[label]
        altered = tuple(restorer(image).clamp(0., 1.) for restorer in models)
        if image.shape[-2:] != previews.shape[-2:]:
            altered = tuple(F.interpolate(y, previews.shape[-2:], mode="bilinear", align_corners=False) for y in altered)
            magnitude = F.interpolate(image, previews.shape[-2:], mode="bilinear", align_corners=False) - previews
        else:
            magnitude = image - previews
        results[label] = torch.cat((basic, respond(magnitude, standard, altered)), dim=1)
    return results


def native(x, models):
    p, m = (restorer(x).clamp(0., 1.) for restorer in models)
    return p, m


def error_terms(p, m, y):
    d, e = m - p, p - y
    return torch.stack((e.square().mean(), (2. * e * d).mean(), d.square().mean()))


def expected(terms, weights):
    a, b, c = terms.unbind(1)
    return (a + b * weights + c * weights.square()).clamp_min(1e-10)


class Trust(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(channels, 24), nn.SiLU(), nn.Linear(24, 1))

    def forward(self, values):
        return self.fc(values)[:, 0].sigmoid()


def nh(role):
    for row in load_data("training")["training"]:
        key = row["id"].rsplit("/", 1)[-1].removeprefix("nh-")
        digits = "".join(ch for ch in key if ch.isdecimal())
        if (digits in NH_RESERVED) != (role == "calibration"):
            continue
        l, t, r, b = row["roi"]
        yield "nh_cal" if role == "calibration" else "nh", row["id"], row["id"], *(row[k][:, t:b, l:r]
                                                                                   for k in ("hazy", "clean"))


def training_pairs():
    for r in training_scenes():
        for j in (1, 3, 5, 7):
            yield "lm", r["name"] + "/" + str(j), r["name"], r["frames"][j], r["clear"]
    for i, item in enumerate(originals()):
        yield "rw", str(i), str(i), *item
    yield from nh("fit")


@torch.inference_mode()
def features(records, models):
    all_cues, observations = defaultdict(list), []
    for d, scene, group, fog, clear in records:
        x, y = as_float(fog), as_float(clear)
        p, m = native(x, models)
        attributes = describe(x, models)
        observations.append(dict(dataset=d, scene=scene, group=group, shape=list(x.shape[-2:]),
                                 mse=error_terms(p, m, y).cpu()))
        for mode, value in attributes.items():
            all_cues[mode].append(value.cpu())
        if len(observations) % 60 == 0:
            print(dict(described=len(observations)), flush=True)
    if not observations:
        raise ValueError("No registered real haze training observations")
    return dict(data=observations, **{k: torch.cat(v) for k, v in all_cues.items()})


def fit():
    configure_cuda()
    ROOT.mkdir(parents=True, exist_ok=True)
    model, source = get_experts()
    registration = dict(
        aim="Test if the change in two restorers' predictions under a physically aligned artificial veil predicts their true usefulness across previously unseen real capture codes.",
        methods=list(ARMS), main=MAIN, seed=list(SEEDS), steps=STEPS, previews=[256, 192], sources=source,
        inputs={record_path(p): digest(p) for p in (TRAIN_CARD, RW_FIT,
             NH_ROOT / "data_manifest.json", SOURCE_ROOT / "boundary_refinement/evaluation/paired/veil_response.py")},
        condition="All responses use the same two full RGB predictors and two standard and two altered thumbnail calls. Resolution changes sizing, exposure changes intensity along the zero-background ray with matched RMS in linear RGB whenever possible, and veil changes transmission while holding an estimated upper-tail airlight fixed. Equal trainable capacity and room-size-independent global mixture.",
        supervision="Optimize the exact relative MSE quadratic between PGL and MCT without changing published backbone weights. Sample equal counts from 32 LMHaze fit scenes, 86 RW2AH records, and 36 NH-HAZE fit pairs. Reserve every fifth NH-HAZE ID 05 to 45. All arms use identical examples.",
        criteria="Lock one common amplitude per arm under existing RW calibration -0.25 dB source budget, maximizing equal relative mean gain on the eight LM calibration scenes and nine reserved NH images. Compare each perturbed cue with static interpolation, opponent perturbations, and individual authors. Locked first evaluation of the remaining REVIDE captures. Primary thresholds +0.10 dB vs both matched preview responses on additional videos and no source loss larger than 0.25 dB.",
        nh_reserved=list(NH_RESERVED), alpha_choices=[0., *AMOUNTS], source_floor=FLOOR,
    )
    if CHECK.exists() and read(CHECK) != registration:
        raise ValueError("The response supervision protocol changed")
    if not CHECK.exists():
        atomic_write(CHECK, registration)
    prep = ROOT / "training_features.pth"
    stamp = ROOT / "training_features.json"
    if stamp.exists():
        note = read(stamp)
        if note["SHA256"] != digest(prep) or note["protocol_SHA256"] != digest(CHECK):
            raise ValueError("Real haze preview descriptors changed")
        data = torch.load(prep, map_location="cpu", weights_only=True)
    else:
        data = features(training_pairs(), model)
        torch_save(prep, data)
        atomic_write(stamp, dict(SHA256=digest(prep), protocol_SHA256=digest(CHECK)))
    labels = torch.stack([v["mse"] for v in data["data"]]).cuda()
    by_bank = {domain: torch.tensor([i for i, row in enumerate(data["data"]) if row["dataset"] == domain])
               for domain in ("lm", "rw", "nh")}
    if {k: len(x) for k, x in by_bank.items()} != dict(lm=128, rw=86, nh=36):
        raise ValueError("Unexpected compared training populations")
    ledger = []
    for seed in SEEDS:
        for arm in ARMS:
            weights = ROOT / f"{arm}-{seed}.pth"
            receipt = weights.with_suffix(".json")
            if receipt.exists():
                saved = read(receipt)
                if saved["checkpoint_SHA256"] != digest(weights) or saved["protocol_SHA256"] != digest(CHECK):
                    raise ValueError("Response training checkpoint changed")
                ledger.append(saved)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 45013)
            x = data[arm].cuda()
            center = x.mean(0)
            scale = x.std(0, unbiased=False).clamp_min(.015)
            inputs = ((x - center) / scale).clamp(-8., 8.)
            net = Trust(inputs.shape[1]).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.0016, weight_decay=.035)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.0001)
            for step in range(STEPS):
                ids = torch.cat([v[torch.randint(len(v), (16,), generator=rng)] for v in by_bank.values()]).cuda()
                opt.zero_grad(set_to_none=True)
                weight = net(inputs[ids])
                terms = labels[ids]
                losses = (expected(terms, weight) / terms[:, 0].clamp_min(1e-10)).log()
                harm = weight[16:32].square().mean()
                loss = losses.mean() + .08 * harm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2., error_if_nonfinite=True)
                opt.step()
                schedule.step()
            torch_save(weights, dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                                     center=center.cpu(), scale=scale.cpu()))
            row = dict(arm=arm, seed=seed, updates=step + 1, trainable=sum(p.numel() for p in net.parameters()),
                       loss=loss.item(), file=str(weights), checkpoint_SHA256=digest(weights),
                       protocol_SHA256=digest(CHECK), finished_utc=now())
            atomic_write(receipt, row)
            ledger.append(row)
            print({k: row[k] for k in ("arm", "seed", "updates", "trainable", "loss")}, flush=True)
    atomic_write(ROOT / "completed.json", dict(runs=ledger, updates=sum(r["updates"] for r in ledger),
                         protocol_SHA256=digest(CHECK), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                         finished_utc=now()))


def load_heads():
    state = read(ROOT / "completed.json")
    if state["protocol_SHA256"] != digest(CHECK) or {
        (r["arm"], r["seed"]) for r in state["runs"]
    } != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("The response decoder study is incomplete")
    heads = {}
    for row in state["runs"]:
        if digest(row["file"]) != row["checkpoint_SHA256"]:
            raise ValueError("Response gate weights changed")
        record = torch.load(row["file"], map_location="cpu", weights_only=True)
        model = Trust(len(record["center"])).cuda()
        model.load_state_dict(record["state"], strict=True)
        heads[row["arm"], row["seed"]] = model.eval().requires_grad_(False), record["center"].cuda(), record["scale"].cuda()
    return heads


def choose(attributes, fields):
    return {key: model(((attributes[key[0]] - center) / scale).clamp(-8., 8.))
            for key, (model, center, scale) in fields.items()}


def calibration_pairs():
    yield from validation_source()
    yield from data_split("calibrate")
    reserve = list(nh("calibration"))
    if len(reserve) != len(NH_RESERVED):
        raise ValueError("The held NH calibration fold is incomplete")
    yield from reserve


@torch.inference_mode()
def calibrate():
    configure_cuda()
    dst = ROOT / "calibration.json"
    signature = digest(ROOT / "completed.json")
    if dst.exists():
        old = read(dst)
        if old["fit_SHA256"] != signature:
            raise ValueError("Calibration precedes the trained gate state")
        return old
    nets = load_heads()
    restorer, _ = get_experts()
    observations = features(calibration_pairs(), restorer)
    columns = {a: observations[a].cuda() for a in ARMS}
    w = choose(columns, nets)
    terms = torch.stack([r["mse"] for r in observations["data"]]).cuda()
    groups = {d: torch.tensor([i for i, r in enumerate(observations["data"]) if r["dataset"] == d], device="cuda")
              for d in ("rw_reserve", "calibrate", "nh_cal")}
    reference = -10. * terms[:, 0].clamp_min(1e-10).log10()
    weights = {a: torch.stack([v for (b, _), v in w.items() if b == a]) for a in ARMS}
    weights["static"] = torch.ones_like(terms[:, :1].T)
    curves = {}
    scales = {}
    for method, gain in weights.items():
        options = {0.: 0.}
        for amount in AMOUNTS:
            predict = torch.stack([-10. * expected(terms, amount * z).log10() for z in gain]).mean(0)
            diff = predict - reference
            effect = {key: diff[indices].mean().item() for key, indices in groups.items()}
            curves[f"{method}/{amount}"] = effect
            if effect["rw_reserve"] >= FLOOR:
                options[amount] = .5 * (effect["calibrate"] + effect["nh_cal"])
        scales[method] = max(options, key=lambda a: (options[a], -a))
    atomic_write(dst, dict(scales=scales, fit_SHA256=signature, curves=curves,
                     counts={k: len(v) for k, v in groups.items()}, finished_utc=now()))
    print(dict(amounts=scales), flush=True)
    return read(dst)


def test_pairs():
    yield from data_split("lmhaze")
    yield from original_rw("fresh_rw")
    yield from fresh_revide()
    yield from pairs()


def summarize(rows):
    list_of_values = defaultdict(list)
    by_image = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k, method = r["dataset"], r["method"]
        list_of_values[k, method].append(r["psnr"])
        by_image[k, r["group"], r["scene"]][method].append(r["psnr"])
    means = {"/".join(k): float(np.mean(value)) for k, value in list_of_values.items()}
    paired = defaultdict(lambda: defaultdict(list))
    for (dataset, group, _), values in by_image.items():
        for control in ("resolution", "exposure", "static", "anchor"):
            paired[dataset, control][group].append(np.mean(values[MAIN]) - np.mean(values[control]))
    intervals = {"/".join(k): bootstrap({g: np.mean(v) for g, v in groups.items()}) for k, groups in paired.items()}
    return means, intervals


@torch.inference_mode()
def test():
    configure_cuda()
    settings = ROOT / "calibration.json"
    if not settings.exists():
        raise ValueError("Freeze all amplitudes before scoring held RGB")
    calibration = calibrate()
    external = read(MANIFEST)
    nets = load_heads()
    restorer, _ = get_experts()
    rows, presentations, counts = [], [], defaultdict(int)
    for ds, scene, group, fog, clear in test_pairs():
        x, y = as_float(fog), as_float(clear)
        p, m = native(x, restorer)
        values = choose(describe(x, restorer), nets)
        terms = error_terms(p, m, y)
        oracle = (-terms[1] / (2. * terms[2]).clamp_min(1e-12)).clamp(0., 1.)
        alphas = {("anchor", 0): 0., ("author_mct", 0): 1., ("equal", 0): .5,
                  ("static", 0): calibration["scales"]["static"], ("pair_oracle", 0): oracle.item()}
        alphas.update({k: calibration["scales"][k[0]] * v.item() for k, v in values.items()})
        for (name, seed), a in alphas.items():
            img = p.lerp(m, a)
            rows.append(dict(dataset=ds, scene=scene, group=group, method=name, seed=seed, alpha=a, **quality(img, y)))
        counts[ds] += 1
        if counts[ds] == 1 and ds in ("lmhaze", "revide_final"):
            result = p.lerp(m, alphas[MAIN, SEEDS[0]])
            presentations.append((ds + " / " + scene, tuple(to_rgb(v) for v in (x, p, m, result, y))))
        if counts[ds] % 40 == 0:
            print(dict(split=ds, tested=counts[ds]), flush=True)
    if dict(counts) != dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=external["count"]):
        raise ValueError("Incomplete matched inference")
    summary, comparisons = summarize(rows)
    raw = ROOT / "individual.json"
    atomic_write(raw, rows)
    path = ROOT / "predictions.png"
    grid = Image.new("RGB", (5 * 274, len(presentations) * 232), "#f3f6f9")
    pen = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=14)
    caption = ("Hazy input", "PGL adaptation anchor", "Author MCT", "Veil response mixture", "Clear reference")
    for i, (scene, views) in enumerate(presentations):
        pen.text((8, i * 232 + 6), scene, font=font, fill="#263747")
        for j, (title, img) in enumerate(zip(caption, views, strict=True)):
            pen.text((j * 274 + 8, i * 232 + 27), title, font=font, fill="#263747")
            im = Image.fromarray(img)
            im.thumbnail((260, 176), Image.Resampling.LANCZOS)
            grid.paste(im, (j * 274 + (274 - im.width) // 2, i * 232 + 45 + (176 - im.height) // 2))
    grid.save(path)
    atomic_write(ROOT / "test.json", dict(psnr=summary, paired=comparisons, counts=dict(counts),
        row_count=len(rows), individual_SHA256=digest(raw), last_revide_SHA256=digest(MANIFEST),
        calibration_SHA256=digest(settings), panel_SHA256=digest(path), finished_utc=now()))
    print({k: round(v, 5) for k, v in summary.items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fit", "calibrate", "test"))
    actions = {"fit": fit, "calibrate": calibrate, "test": test}
    actions[cli.parse_args().stage]()

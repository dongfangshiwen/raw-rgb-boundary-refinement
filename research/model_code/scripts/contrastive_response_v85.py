"""Train a semantic residual using a matched color view as a negative control."""
import argparse
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from dehaze_seg.metrics import confusion_matrix
from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import paired_iou_intervals
from scripts.class_protected_residual_v81 import soft_counts
from scripts.coarse_semantics_v39 import blend, paint
from scripts.cross_scale_rgb_v44 import resize
from scripts.edge_projection_v47 import draw
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_fog_correction_v70 import aggregate
from scripts.resolution_interaction_v84 import (
    ROOT as V84, SEEDS, DetailHead, DualScaleFeatures, add_global_cohort, batch, centered,
    load as reference_weights, packets, response, selection, signed_views,
)
from scripts.restoration_affinity_v75 import compress_scores
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/contrastive-response-v85"
STEPS = 480
PRIMARY = "difference_tone"
ARMS = ("raw_extra", "plain_restored", "average_views", "difference_raw", PRIMARY, "difference_shift")
REFERENCES = ("author_raw", "raw_anchor", "v84_fusion", "v84_interaction")


def registration():
    return dict(source_SHA256=digest(__file__), parent=digest(V84 / "completed.json"),
                parent_source=digest(REPO / "scripts/resolution_interaction_v84.py"), fit_population=96,
                steps=STEPS, seeds=list(SEEDS), arms=list(ARMS), fixed_restoration=True,
                design="Train matching residual heads around the frozen V84 raw detail anchor. Forward the same learned head "
                       "with two registered inputs where applicable. Test the difference between responses to restoration and "
                       "its monotone color control, a raw reference instead, an ordinary average, restoration alone, raw alone "
                       "and spatially shifted restoration. Match all supervised sampling and optimization. RGB stays identical.",
                primary="Three-seed full mIoU vs ordinary average, difference from raw, shifted restoration, and same-capacity "
                        "raw continuation. Check previously evaluated real fog and separately held, partial positive annotations.")


def cues(views, key):
    mode = {"raw": "raw_detail", "tone": "tone_interaction", "restored": "ordinary_fusion"}[key]
    return response(views, mode)[1]


def raw_anchor(raw_model, views):
    start, cue, delta = response(views, "raw_detail")
    return start + raw_model(cue, delta, "raw_detail")


def responses(net, views, arm):
    source = "raw" if arm == "raw_extra" else "restored"
    other = {"average_views": "tone", "difference_raw": "raw", PRIMARY: "tone", "difference_shift": "tone"}.get(arm)
    cue = cues(views, source)
    if other is None:
        return centered(net(cue, None, "ordinary_fusion"))
    control = cues(views, other)
    first, second = net(torch.cat((cue, control)), None, "ordinary_fusion").chunk(2)
    sign = 1. if arm == "average_views" else -1.
    return .5 * centered(first + sign * second)


def train(arm, seed, raw, pool, positions, weight, smoke):
    file = ROOT / f"{arm}-{seed}.pth"
    note = file.with_suffix(".json")
    lock = digest(ROOT / "protocol.json") if not smoke else None
    if not smoke and note.is_file():
        result = read(note)
        if result["weights_SHA256"] != digest(file) or result["protocol_SHA256"] != lock:
            raise ValueError(f"Unexpected completed checkpoint {note.name}")
        return result
    set_seed(seed)
    net = DetailHead().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.00055, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.000055)
    gen = torch.Generator().manual_seed(85007 + seed)
    steps = 4 if smoke else STEPS
    for _ in range(steps):
        sample, label = batch(pool, positions, gen, "shift_interaction" if arm == "difference_shift" else arm)
        base = raw_anchor(raw, sample)
        delta = responses(net, sample, arm)
        logits = base + delta
        hit, union, count, _ = soft_counts(logits, label)
        loss = F.cross_entropy(logits, label, weight=weight, ignore_index=255)
        loss += .8 * (1. - hit[count > 0] / union[count > 0].clamp_min(1e-5)).mean() + .015 * delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    result = dict(method=arm, seed=seed, steps=steps, params=sum(p.numel() for p in net.parameters()),
                  loss=float(loss.detach()))
    if not smoke:
        torch_save(file, {k: v.detach().cpu() for k, v in net.state_dict().items()})
        result.update(weights_SHA256=digest(file), protocol_SHA256=lock)
        atomic_write(note, result)
    print(result, flush=True)
    return result


def fit(smoke):
    configure_cuda()
    reference = reference_weights()
    lock = ROOT / "protocol.json"
    cfg = registration()
    if not smoke:
        if lock.is_file() and read(lock) != cfg:
            raise ValueError("The contrastive response scope changed")
        atomic_write(lock, cfg)
    pool, weight, positions = packets(DualScaleFeatures(), smoke)
    seeds = SEEDS[:1] if smoke else SEEDS
    runs = [train(arm, seed, reference["raw_detail", seed], pool, positions, weight, smoke) for seed in seeds for arm in ARMS]
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=runs, steps=sum(x["steps"] for x in runs),
                     protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def load():
    lock, complete = ROOT / "protocol.json", read(ROOT / "completed.json")
    if read(lock) != registration() or digest(lock) != complete["protocol_SHA256"]:
        raise ValueError("The reference response experiment changed")
    models = {}
    for row in complete["runs"]:
        key = row["method"], row["seed"]
        file = ROOT / f"{key[0]}-{key[1]}.pth"
        net = DetailHead().cuda().eval().requires_grad_(False)
        if digest(file) != row["weights_SHA256"] or row["steps"] != STEPS or row["params"] != sum(p.numel() for p in net.parameters()):
            raise ValueError(f"Reference response checkpoint differs from registration: {file.name}")
        net.load_state_dict(torch.load(file, map_location="cpu", weights_only=True), strict=True)
        models[key] = net
    if set(models) != {(m, s) for m in ARMS for s in SEEDS}:
        raise ValueError("Missing response control")
    return models


def predictions(raw, views, bank, reference, seed, shape, shifted):
    start = raw_anchor(reference["raw_detail", seed], views)
    logits = {"author_raw": raw, "raw_anchor": start}
    for tag, mode in (("v84_fusion", "ordinary_fusion"), ("v84_interaction", "interaction")):
        z, x, h = response(views, mode)
        logits[tag] = z + reference[mode, seed](x, h, mode)
    for mode in ARMS:
        logits[mode] = start + responses(bank[mode, seed], shifted if mode == "difference_shift" else views, mode)
    return {k: resize(z, shape).argmax(1)[0] for k, z in logits.items()}


@torch.inference_mode()
def score(part):
    configure_cuda()
    reference, bank, engine = reference_weights(), load(), DualScaleFeatures()
    rows, frames, panels = [], Counter(), {}
    for dataset, key, group, lighting, _, label, rgb in selection(part):
        pack, restored, raw = engine(rgb)
        shifted = signed_views(pack)
        present = (dataset, group) not in panels and sum(x[0] == dataset for x in panels) < 4
        for seed in SEEDS:
            output = predictions(raw, pack, bank, reference, seed, label.shape, shifted)
            rows.extend(dict(dataset=dataset, id=key, group=group, method=method, seed=seed,
                             counts=confusion_matrix(value, label, 19).tolist()) for method, value in output.items())
            if present and seed == SEEDS[0]:
                tags = ("raw_anchor", "v84_fusion", "average_views", "difference_raw", PRIMARY)
                masks = [blend(rgb, output[tag].byte().cpu().numpy()) for tag in tags]
                panels[dataset, group] = (f"{key} / {lighting or dataset}", [rgb, to_rgb(restored), *masks,
                                          paint(label.byte().cpu().numpy())],
                                          ("Fog", "Restored RGB", "Raw anchor", "V84 fusion", "Average views",
                                           "Raw subtraction", "Tone subtraction", "Reference"))
        frames[dataset] += 1
        if sum(frames.values()) % 32 == 0:
            print(dict(part=part, evaluated=sum(frames.values())), flush=True)
    v84 = read(V84 / f"{part}-test.json")
    if dict(frames) != v84["counts"]:
        raise ValueError("Reference response and V84 evaluated different cohorts")
    data = add_global_cohort(rows)
    comparison = [(PRIMARY, m) for m in (*REFERENCES, *ARMS) if m != PRIMARY]
    comparison += [("average_views", "raw_extra"), ("plain_restored", "raw_extra")]
    results = aggregate(data)
    out, preview = ROOT / f"{part}-individual.json.gz", ROOT / f"{part}-predictions.png"
    compress_scores(out, rows)
    draw(preview, list(panels.values()))
    atomic_write(ROOT / f"{part}-test.json", dict(counts=dict(frames), metrics=results,
                 contrasts=paired_iou_intervals(data, comparison, "method", "counts"),
                 rows_SHA256=digest(out), picture_SHA256=digest(preview), prior_scores_SHA256=digest(V84 / f"{part}-test.json"),
                 weights_SHA256=digest(ROOT / "completed.json"), utc=now()))
    print({name: round(score["miou"], 4) for name, score in results.items()}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("smoke", "fit", "calibrate", "held", "external"))
    args = cli.parse_args()
    if args.stage in ("fit", "smoke"):
        fit(args.stage == "smoke")
    else:
        score(args.stage)

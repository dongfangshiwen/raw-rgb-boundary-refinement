"""Select a matched number of update coordinates using real haze pairs."""
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from dehaze_seg.utils import set_seed, write_csv
from scripts.author_pgl_v25 import load_pgl
from scripts.crop_context_v32 import originals
from scripts.evaluate_paired_basis_v34 import load_blend, run_scores, test_input, validation_source
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_structure_transfer_v32 import data_split
from scripts.haze_response_v35 import averages, source_loss
from scripts.paired_basis_v34 import ROOT as V34, SOURCE_FLOOR, TARGET_MARGIN, state_copy, stream
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.selective_adaptation_v33 import layout, verify_changes
from scripts.source_replay_v32 import PROTOCOL as REPLAY
from scripts.stage_lmhaze_v31 import CARD as LMH, TRAIN_CARD
from scripts.stage_postfusion_holdout_v30 import RW_CARD
from scripts.structure_transfer_v32 import PAIRS, SEEDS, STEPS, objective, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/gradient-support-v36"
ARMS = ("target_gradient", "source_fisher", "shuffled_density", "paired_density")
MAIN = "paired_density"
ANCHOR = "local_matched"
PROBES = 24
PROBE_SEED = 36043
ALPHAS = (.25, .5, .75, 1.)


def gradients(output, target, parameters):
    per_view = [objective(torch.cat((x[None], x[None])), y[None], None, "point")[0]
                for x, y in zip(output, target, strict=True)]
    arrays = [torch.cat([p.reshape(-1) for p in torch.autograd.grad(
        v, parameters, retain_graph=True)]) for v in per_view]
    return torch.stack(arrays)


def probe(author, params, target, source):
    buffers = torch.zeros((4, sum(p.numel() for p in params)), dtype=torch.float32, device="cuda")
    samples = stream(target, source, PROBE_SEED)
    for _ in range(PROBES):
        x, y, z = next(samples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            light, dense, replay = author(x).float().split(PAIRS)
        a = gradients(light, y, params)
        b = gradients(dense, y, params)
        source_terms = [source_loss(o[None] - t[None]) for o, t in zip(replay, z, strict=True)]
        s = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            term, params, retain_graph=i < len(source_terms) - 1)])
                        for i, term in enumerate(source_terms)])
        buffers[0] += .5 * (a + b).mean(0)
        buffers[1] += s.square().mean(0)
        buffers[2] += (.5 * (b - a.roll(1, 0))).square().mean(0)
        buffers[3] += (.5 * (b - a)).square().mean(0)
    return buffers.cpu().div_(PROBES)


def rank(moment):
    target, source, false_pair, true_pair = moment
    signal = target.square()
    floor = (.1 * source.mean()).clamp_min(1e-16)
    # Set a common scale for the instability terms. Both controls use the
    # same ratio, regardless of how much a shuffled difference spreads.
    factor = source.mean() / false_pair.mean().clamp_min(1e-16)
    baseline = source + floor
    return (signal, signal / baseline, signal / (baseline + factor * false_pair),
            signal / (baseline + factor * true_pair))


def supports(moment, names, params, budget):
    scored = rank(moment)
    sizes = [p.numel() for p in params]
    results = {}
    for arm, score in zip(ARMS, scored, strict=True):
        ids = score.argsort(descending=True, stable=True)[:budget]
        if not score[ids].isfinite().all() or score[ids[-1]] <= 0:
            raise ValueError("Insufficient finite, nonzero gradient coordinates")
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask[ids] = True
        split = [(name, active.nonzero().flatten()) for name, active in zip(names, mask.split(sizes), strict=True)]
        results[arm] = {name: index for name, index in split if len(index)}
    return results


class Coordinates(nn.Module):
    def __init__(self, model, selected):
        super().__init__()
        self.names = tuple(selected)
        initial = [model.get_parameter(name).detach() for name in self.names]
        self.shapes = [p.shape for p in initial]
        self.sizes = [p.numel() for p in initial]
        self.register_buffer("start", torch.cat([p.flatten() for p in initial]))
        sizes = torch.tensor(self.sizes, dtype=torch.long)
        offsets = torch.cat((sizes.new_zeros(1), sizes[:-1].cumsum(0)))
        self.register_buffer("indices", torch.cat([selected[n] + i for n, i in zip(self.names, offsets, strict=True)])
                             .to(self.start.device))
        self.active = nn.Parameter(self.start[self.indices].clone())

    def tensors(self):
        flat = self.start.index_copy(0, self.indices, self.active)
        return {name: v.view(shape) for name, v, shape in zip(
            self.names, flat.split(self.sizes), self.shapes, strict=True
        )}

    def forward(self, model, image):
        return functional_call(model, self.tensors(), (image,))


def fit(smoke=False):
    configure_cuda()
    model, author = load_pgl()
    params_by_name = dict(model.named_parameters())
    initial = state_copy(model)
    budget = sum(item["budget"] for item in layout(model))
    target, source = training_scenes(), originals()
    total_updates = 2 * STEPS
    code = (REPO / "scripts/paired_basis_v34.py", REPO / "scripts/haze_response_v35.py",
            REPO / "scripts/evaluate_paired_basis_v34.py")
    plan = dict(
        hypothesis="A source-risk-normalized update mask should transfer better when it prioritizes shared responses to the clear target at two registered haze levels and penalizes disagreement. Distinguish the pairing signal from target-only and source-Fisher selection.",
        author=author, arms=list(ARMS), capacity=budget, seeds=list(SEEDS), updates=total_updates,
        probes=PROBES, probe_seed=PROBE_SEED, code_SHA256=digest(__file__),
        controls=dict(local_exact_completed=digest(V34 / "completed.json"),
                      shared_code={str(p.relative_to(REPO)): digest(p) for p in code}),
        data={str(p.relative_to(REPO)): digest(p) for p in (REPLAY, TRAIN_CARD, LMH, RW_CARD)},
        selection="Probe author model on training data. Target-only sorts squared mean per-coordinate reconstruction gradients. Source Fisher normalizes by uncentered source second moment plus 10% of its overall mean. Paired also penalizes the gradient difference across light/dense images of the exact scene. Shuffled pairs use the same gradients using other scenes in the same batch. Take 49,056 highest indices, then train on the same sequence and replay as V34 with fixed BN. Every arm uses the same author and final update.",
        evaluation="Calibrate scales using existing eight LMHaze scenes and RW reserve with a -0.25 dB source floor. Score the 13 LMHaze test scenes and previous 46 RW captures with fixed coefficients. Report repeated-benchmark evidence, compare within-seed and scene-cluster differences.",
        alpha_grid=[0., *ALPHAS], source_floor_db=SOURCE_FLOOR, target_margin_db=TARGET_MARGIN,
    )
    path = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if path.exists() and read(path) != plan:
            raise ValueError("Gradient support protocol changed")
        if not path.exists():
            atomic_write(path, plan)
    probes = ROOT / "probe.pth"
    signature = ROOT / "probe.json"
    if signature.exists() and not smoke:
        metadata = read(signature)
        if metadata["SHA256"] != digest(probes) or metadata["protocol_SHA256"] != digest(path):
            raise ValueError("Gradient probe provenance differs")
        table = torch.load(probes, map_location="cpu", weights_only=True)
    else:
        table = probe(model, tuple(params_by_name.values()), target, source)
        if not smoke:
            torch_save(probes, table)
            atomic_write(signature, dict(SHA256=digest(probes), protocol_SHA256=digest(path), finished_utc=now()))
    masks = supports(table, list(params_by_name), list(params_by_name.values()), budget)
    if not smoke:
        mask_path = ROOT / "masks.pth"
        if mask_path.exists():
            previous = torch.load(mask_path, map_location="cpu", weights_only=True)
            if any(v.keys() != previous[m].keys() or any(not torch.equal(a, previous[m][k])
                       for k, a in v.items()) for m, v in masks.items()):
                raise ValueError("Registered active coordinates changed")
        else:
            torch_save(mask_path, masks)
    print(dict(support_counts={m: len(v) for m, v in masks.items()},
               pair_vs_src_overlap=sum(len(set(ids.tolist()) & set(masks["source_fisher"].get(k, torch.empty(0)).tolist()))
                                      for k, ids in masks[MAIN].items())), flush=True)
    model.eval().requires_grad_(False)
    rows = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in ARMS:
            trained = ROOT / f"{mode}-{seed}.pth"
            receipt = trained.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["checkpoint_SHA256"] != digest(trained) or row["protocol_SHA256"] != digest(path):
                    raise ValueError("Gradient support checkpoint differs")
                rows.append(row)
                continue
            set_seed(seed)
            selected = masks[mode]
            coordinates = Coordinates(model, selected)
            trainable = list(coordinates.parameters())
            if coordinates.active.numel() != budget:
                raise ValueError("All masks must contain the same number of trained coordinates")
            optimizer = torch.optim.AdamW(trainable, lr=.00012, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, eta_min=.00002)
            started = time.perf_counter()
            for step, (x, y, z) in enumerate(stream(target, source, seed), 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    clean, replay = coordinates(model, x).float().split((2 * PAIRS, PAIRS))
                loss = objective(clean, y, None, "point")[0] + source_loss(replay - z)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                if smoke and step == 2:
                    break
            if not smoke and step != total_updates:
                raise ValueError("Did not meet the matched training budget")
            with torch.no_grad():
                weights = dict(initial)
                weights.update({name: v.detach().cpu().clone() for name, v in coordinates.tensors().items()})
            changed = verify_changes(initial, weights, set(params_by_name), selected, "local_matched")
            stats = dict(arm=mode, seed=seed, updates=step, trainable=budget, changed=changed,
                         seconds=round(time.perf_counter() - started, 2), loss=loss.item())
            print(stats, flush=True)
            if not smoke:
                torch_save(trained, weights)
                row = dict(stats, file=str(trained), checkpoint_SHA256=digest(trained),
                           protocol_SHA256=digest(path), finished_utc=now())
                atomic_write(receipt, row)
                rows.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(
            protocol_SHA256=digest(path), updates=sum(r["updates"] for r in rows), runs=rows,
            mask_SHA256=digest(ROOT / "masks.pth"), GPU=torch.cuda.get_device_name(),
            torch=str(torch.__version__), finished_utc=now(),
        ))


def networks():
    result = read(ROOT / "completed.json")
    if result["protocol_SHA256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Gradient support protocol differs")
    if {(r["arm"], r["seed"]) for r in result["runs"]} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Unmatched experiments")
    controls = read(V34 / "completed.json")
    check = read(ROOT / "protocol.json")
    if check["controls"]["local_exact_completed"] != digest(V34 / "completed.json"):
        raise ValueError("Local coordinate control differs")
    models = result["runs"] + [r for r in controls["runs"] if r["arm"] == ANCHOR]
    if any(digest(r["file"]) != r["checkpoint_SHA256"] for r in models):
        raise ValueError("An evaluated checkpoint changed")
    return models


@torch.inference_mode()
def calibrate():
    configure_cuda()
    # A fresh registration makes both precision and all scale candidates explicit.
    # Verify calibration precision against the same PGL models in ordinary FP32.
    models = networks()
    path = ROOT / "calibration.json"
    if path.exists():
        found = read(path)
        if (found["csv_SHA256"] != digest(ROOT / "calibration.csv")
                or found["fit_SHA256"] != digest(ROOT / "completed.json")):
            raise ValueError("Calibration scores changed")
        return found
    pgl = load_pgl()[0].to(memory_format=torch.channels_last)
    source, target = list(validation_source()), list(data_split("calibrate"))
    both = [*source[:2], *target[:2]]
    anchor = ("pgl", 0, 0.)
    base = {anchor: pgl, (MAIN, SEEDS[0], 1.): load_blend(pgl, next(
        r for r in models if r["arm"] == MAIN and r["seed"] == SEEDS[0]), 1.)}
    fp32, _ = run_scores(both, base)
    torch.backends.cudnn.allow_tf32 = True
    fast, _ = run_scores(both, base)
    max_diff = max(abs(a["psnr"] - b["psnr"]) for a, b in zip(fp32, fast, strict=True))
    use_tf32 = max_diff <= .01
    torch.backends.cudnn.allow_tf32 = use_tf32
    fits = {anchor: pgl}
    for r in models:
        for a in ALPHAS:
            fits[r["arm"], r["seed"], a] = load_blend(pgl, r, a)
    sr, _ = run_scores(source, fits)
    s = averages(sr)
    floor = s["rw_reserve", "pgl", 0.] + SOURCE_FLOOR
    permitted = {(m, a) for m in (*ARMS, ANCHOR) for a in ALPHAS if s["rw_reserve", m, a] >= floor}
    used = {k: v for k, v in fits.items() if k == anchor or (k[0], k[2]) in permitted}
    tr, _ = run_scores(target, used)
    t = averages(tr)
    choice = {}
    for mode in (*ARMS, ANCHOR):
        options = {0.: t["calibrate", "pgl", 0.]}
        options.update({a: t["calibrate", m, a] for m, a in permitted if m == mode})
        choice[mode] = max(options, key=lambda a: (options[a], -a))
    table = ROOT / "calibration.csv"
    write_csv(table, sr + tr)
    card = dict(scales=choice, max_precision_diff=max_diff, cudnn_tf32=use_tf32,
                fit_SHA256=digest(ROOT / "completed.json"),
                csv_SHA256=digest(table), source_floor=floor, src={"/".join(map(str, k)): v for k, v in s.items()},
                target={"/".join(map(str, k)): v for k, v in t.items()}, finished_utc=now())
    atomic_write(path, card)
    print(card["scales"], flush=True)
    return card


@torch.inference_mode()
def test():
    card = read(ROOT / "calibration.json")
    if (card["csv_SHA256"] != digest(ROOT / "calibration.csv")
            or card["fit_SHA256"] != digest(ROOT / "completed.json")):
        raise ValueError("Calibration results are inconsistent")
    configure_cuda()
    torch.backends.cudnn.allow_tf32 = card["cudnn_tf32"]
    pgl = load_pgl()[0].to(memory_format=torch.channels_last)
    nets = {("pgl", 0, 0.): pgl}
    for r in networks():
        a = card["scales"][r["arm"]]
        nets[r["arm"], r["seed"], a] = load_blend(pgl, r, a) if a else pgl
    rows, _ = run_scores(test_input(), nets)
    table = ROOT / "test.csv"
    write_csv(table, rows)
    scores = {f"{d}/{m}": v for (d, m, _), v in averages(rows).items()}
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["dataset"], r["group"], r["scene"]][r["method"]].append(r["psnr"])
    intervals = {}
    for control in (*ARMS[:-1], ANCHOR, "pgl"):
        by_scene = defaultdict(list)
        for (d, g, _), results in groups.items():
            if d == "lmhaze":
                by_scene[g].append(np.mean(results[MAIN]) - np.mean(results[control]))
        intervals[control] = bootstrap({k: np.mean(v) for k, v in by_scene.items()})
    src = scores[f"rw_new/{MAIN}"] - scores["rw_new/pgl"]
    margin = {m: scores[f"lmhaze/{MAIN}"] - scores[f"lmhaze/{m}"] for m in (*ARMS[:-1], ANCHOR)}
    decision = dict(source_delta=src, source_pass=src >= SOURCE_FLOOR, target_margin=margin,
                    target_pass={m: v >= TARGET_MARGIN and intervals[m]["ci95"][0] > 0 for m, v in margin.items()})
    output = dict(psnr=scores, intervals=intervals, evidence=decision, rows=len(rows),
                  csv_SHA256=digest(table), fit_SHA256=digest(ROOT / "completed.json"),
                  calibration_SHA256=digest(ROOT / "calibration.json"), finished_utc=now())
    atomic_write(ROOT / "test.json", output)
    print(dict(psnr=scores, decision=decision), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("smoke", "fit", "calibrate", "test"))
    stage = cli.parse_args().stage
    if stage in ("smoke", "fit"):
        fit(smoke=stage == "smoke")
    elif stage == "calibrate":
        calibrate()
    else:
        test()

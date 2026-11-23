"""Control point-distillation strength and test source camera transfer."""
import argparse
from collections import defaultdict
import csv
import time

import numpy as np
import torch

from dehaze_seg.utils import set_seed, write_csv
from scripts.author_pgl_v25 import load_pgl
from scripts.crop_context_v32 import originals
from scripts.evaluate_paired_basis_v34 import load_blend, run_scores, test_input
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.gradient_support_v36 import ROOT as V36, networks as v36_networks
from scripts.haze_response_v35 import MAIN, ROOT as V35, completed as v35_completed, source_loss, teacher_cost, attenuated
from scripts.paired_basis_v34 import ROOT as V34, state_copy, stream
from scripts.prepare_rw2ah_v25 import ROOT as RW2AH
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.selective_adaptation_v33 import adapt, layout, materialize, verify_changes
from scripts.source_replay_v32 import PROTOCOL as REPLAY
from scripts.stage_postfusion_holdout_v30 import RW_CARD, camera_code
from scripts.structure_transfer_v32 import PAIRS, SEEDS, STEPS, objective, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/point-strength-v37"
CONTROL = "matched_point"
STOCK = ("point", MAIN)


def fit():
    configure_cuda()
    v35_completed()
    ROOT.mkdir(parents=True, exist_ok=True)
    teacher, author = load_pgl()
    model = load_pgl()[0]
    initial = state_copy(model)
    names = set(dict(model.named_parameters()))
    stages = layout(model)
    budget = sum(s["budget"] for s in stages)
    target, source = training_scenes(), originals()
    updates = 2 * STEPS
    path = ROOT / "protocol.json"
    protocol = dict(
        hypothesis="Compare paired haze response distillation to exactly the same weight on the point-distillation term without the response term.",
        parent_SHA256=digest(V35 / "completed.json"), calibration_SHA256=digest(V35 / "calibration.json"),
        code_SHA256=digest(__file__), author=author, count=budget, seeds=list(SEEDS), updates=updates,
        deployment="Use the same raw full update for all three objective controls. The V35 validation choices were also 1.0 for both references. This follow-up checks the original strict comparison at a matched point weight.",
        extra_source="Use the original 35 RW2AH test images from prefixes 2, 9 and 10. No replay or calibration camera has those prefixes. These images were scored earlier in the project and are a separate camera cohort for the current fits.",
    )
    if path.exists() and read(path) != protocol:
        raise ValueError("The point weight protocol changed")
    if not path.exists():
        atomic_write(path, protocol)
    rows = []
    for seed in SEEDS:
        output = ROOT / f"{CONTROL}-{seed}.pth"
        receipt = output.with_suffix(".json")
        if receipt.exists():
            previous = read(receipt)
            if previous["checkpoint_SHA256"] != digest(output) or previous["protocol_SHA256"] != digest(path):
                raise ValueError("A point-only checkpoint changed")
            rows.append(previous)
            continue
        set_seed(seed)
        rng = torch.Generator().manual_seed(seed + 35023)
        model.load_state_dict(initial, strict=True)
        selected = adapt(model, "local_lora", stages)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable) != budget:
            raise ValueError("Point-only budget differs from V35")
        optimizer = torch.optim.AdamW(trainable, lr=.00012, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=.00002)
        start = time.perf_counter()
        for step, (haze, dest, src) in enumerate(stream(target, source, seed), 1):
            a = (.68 + .22 * torch.rand((PAIRS, 1, 1, 1), generator=rng)).to(src)
            weak = attenuated(haze[2 * PAIRS:], src, a)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(torch.cat((haze, weak))).float()
                with torch.no_grad():
                    frozen = teacher(torch.cat((haze[2 * PAIRS:], weak))).float()
            y, src_y = out.split(2 * PAIRS)
            loss = objective(y, dest, None, "point")[0] + source_loss(src_y - torch.cat((src, src)))
            loss = loss + .5 * teacher_cost(src_y - frozen, a, "point")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
        if step != updates:
            raise ValueError("Point-only update schedule differs from V35")
        weights = materialize(model, selected)
        changed = verify_changes(initial, weights, names, selected, "local_lora")
        torch_save(output, weights)
        row = dict(arm=CONTROL, seed=seed, protocol_SHA256=digest(path), checkpoint_SHA256=digest(output),
                   file=str(output), updates=step, trainable=budget, changed=changed, loss=loss.item(),
                   seconds=round(time.perf_counter() - start, 2), finished_utc=now())
        atomic_write(receipt, row)
        print({k: row[k] for k in ("arm", "seed", "updates", "loss", "seconds")}, flush=True)
        rows.append(row)
    atomic_write(ROOT / "completed.json", dict(runs=rows, updates=sum(r["updates"] for r in rows),
                                              protocol_SHA256=digest(path), finished_utc=now()))


def state():
    result = read(ROOT / "completed.json")
    if result["protocol_SHA256"] != digest(ROOT / "protocol.json"):
        raise ValueError("Point-only protocol changed")
    if {r["seed"] for r in result["runs"]} != set(SEEDS):
        raise ValueError("Point-only results have a missing seed")
    for row in result["runs"]:
        if row["checkpoint_SHA256"] != digest(row["file"]):
            raise ValueError("Point-only checkpoint changed")
    return result["runs"]


@torch.inference_mode()
def test():
    configure_cuda()
    author = load_pgl()[0]
    model = {(CONTROL, r["seed"], 1.): load_blend(author, r, 1.) for r in state()}
    scores, _ = run_scores(test_input(), model)
    path = ROOT / "test.csv"
    write_csv(path, scores)
    prev = read(V35 / "test.json")
    prior_path = V35 / "test.csv"
    if prev["csv_SHA256"] != digest(prior_path) or any(prev["scales"][m] != 1. for m in STOCK):
        raise ValueError("Previous exact scorer or scales differ")
    values = defaultdict(lambda: defaultdict(list))
    with prior_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("pgl", *STOCK):
                values[r["dataset"], r["group"], r["scene"]][r["method"]].append(float(r["psnr"]))
    for r in scores:
        values[r["dataset"], r["group"], r["scene"]][CONTROL].append(r["psnr"])
    diffs = defaultdict(lambda: defaultdict(list))
    summaries = defaultdict(list)
    for (dataset, group, _), pairs in values.items():
        for method, psnr in pairs.items():
            summaries[dataset, method].append(np.mean(psnr))
        diffs[dataset]["/" + group].append(np.mean(pairs[MAIN]) - np.mean(pairs[CONTROL]))
    metrics = {"/".join(k): float(np.mean(v)) for k, v in summaries.items()}
    result = dict(rows=len(scores), scores=metrics, fit_SHA256=digest(ROOT / "completed.json"),
                  current_csv_SHA256=digest(path), v35_csv_SHA256=digest(prior_path),
                  pair_minus_same_point={k: bootstrap({g: np.mean(v) for g, v in items.items()}) for k, items in diffs.items()},
                  finished_utc=now())
    atomic_write(ROOT / "test.json", result)
    print(result, flush=True)


def extra_source():
    original = read(RW2AH / "data_manifest.json")["test"]
    if original["SHA256"] != digest(original["file"]):
        raise ValueError("Original benchmark pairs differ")
    used = set(camera_code(x) for x in read(REPLAY)["rw_fit_ids"] + read(RW_CARD)["ids"])
    selected = torch.load(original["file"], map_location="cpu", weights_only=True)
    groups = {camera_code(r["id"]) for r in selected}
    if groups & used or len(selected) != 35 or groups != {"2", "9", "10"}:
        raise ValueError("Expected all three source test cameras to be unseen in adaptation and selection")
    for row in selected:
        scene = row["id"]
        yield "rw_camera_holdout", scene, camera_code(scene), row["hazy"], row["clean"]


@torch.inference_mode()
def cameras():
    configure_cuda()
    author = load_pgl()[0]
    models = {("pgl", 0, 0.): author}
    ledgers = (
        (V34, "chosen", ("paired", "local_matched"), read(V34 / "completed.json")["runs"]),
        (V35, "chosen", STOCK, v35_completed()["runs"]),
        (V36, "scales", ("source_fisher", "paired_density"), v36_networks()),
    )
    for folder, key, arms, rows in ledgers:
        scale = read(folder / "calibration.json")[key]
        for row in rows:
            if row["arm"] not in arms:
                continue
            if row["checkpoint_SHA256"] != digest(row["file"]):
                raise ValueError("A source camera comparison checkpoint differs")
            arm = row["arm"]
            a = scale[arm]
            models[folder.name[-3:] + "_" + arm, row["seed"], a] = load_blend(author, row, a) if a else author
    for row in state():
        models[CONTROL, row["seed"], 1.] = load_blend(author, row, 1.)
    results, _ = run_scores(extra_source(), models)
    table = ROOT / "source_cameras.csv"
    write_csv(table, results)
    overall, by_camera = defaultdict(list), defaultdict(list)
    for r in results:
        overall[r["method"]].append(r["psnr"])
        by_camera[r["method"], r["group"]].append(r["psnr"])
    scores = dict(rows=len(results), heads=sorted({r["group"] for r in results}),
                  overall={k: float(np.mean(v)) for k, v in overall.items()},
                  by_camera={"/".join(k): float(np.mean(v)) for k, v in by_camera.items()},
                  input_SHA256=digest(RW2AH / "data_manifest.json"), csv_SHA256=digest(table), finished_utc=now())
    atomic_write(ROOT / "source_cameras.json", scores)
    print(scores["overall"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fit", "test", "cameras"))
    actions = {"fit": fit, "test": test, "cameras": cameras}
    actions[cli.parse_args().stage]()

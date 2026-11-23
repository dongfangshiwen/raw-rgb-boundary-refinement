"""Train matched decoder adapters with and without registered haze pairs."""
from boundary_refinement.paths import SOURCE_ROOT, record_path
import argparse
from contextlib import ExitStack
import time

import torch
from torch import nn
from torch.nn.utils import parametrize

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import load_pgl
from boundary_refinement.training.crop_context import batches, originals
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.selective_adaptation import (
    MASK_SEED, ROOT as V33, LocalLoRA, adapt, layout, materialize, verify_changes,
)
from boundary_refinement.training.source_replay import PROTOCOL as REPLAY, band_parts
from boundary_refinement.data.staging.stage_lmhaze import CARD as LMH, TRAIN_CARD
from boundary_refinement.data.staging.stage_postfusion_holdout import RW_CARD
from boundary_refinement.training.structure_transfer import PAIRS, SEEDS, SIDE, STEPS, objective, training_scenes
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/paired-basis-v34"
MAIN = "paired"
ARMS = ("local_lora", "local_matched", "source_metric", "marginal", "shuffled", MAIN)
MOMENTS = ("source", "marginal", "shuffled", "paired")
PASSES = 2
PARALLEL_CROP_STEPS = 32
BASIS_SEED = 34057
RIDGE = .10
ALPHAS = (.5, .75, 1.)
SOURCE_FLOOR = -.25
TARGET_MARGIN = .10


def stream(target, source, seed):
    for repeat in range(PASSES):
        for fog, _, clean, reference, _ in batches(target, source, seed + 10007 * repeat):
            yield fog, clean, reference


def positions(x):
    return x.float()[..., ::4, ::4].permute(0, 2, 3, 1)


def second_moment(x):
    flat = x.reshape(-1, x.shape[-1])
    return (flat.T @ flat) / len(flat)


@torch.no_grad()
def collect(model, stages, target, source):
    moments = {}
    for stage in stages:
        for name in stage["lora_ranks"]:
            shape = (len(MOMENTS), stage["width"], stage["width"])
            moments[name] = torch.zeros(shape, device="cuda", dtype=torch.float32)

    def record(statistics, inputs):
        light, dense, replay = positions(inputs[0]).split(PAIRS)
        controls = (replay, torch.cat((light, dense)), dense - light.roll(1, 0), dense - light)
        for output, values in zip(statistics, controls, strict=True):
            output.add_(second_moment(values))

    with ExitStack() as stack:
        for name, statistics in moments.items():
            handle = model.get_submodule(name).register_forward_pre_hook(
                lambda _layer, inputs, c=statistics: record(c, inputs)
            )
            stack.callback(handle.remove)
        iterator = stream(target, source, BASIS_SEED)
        for _ in range(PARALLEL_CROP_STEPS):
            fog, _, _ = next(iterator)
            model(fog)
    return {name: (matrix / PARALLEL_CROP_STEPS).cpu().double() for name, matrix in moments.items()}


def eigenspaces(statistics):
    source, *target = statistics
    eigenvalues, eigenvectors = torch.linalg.eigh(source)
    scale = eigenvalues.mean().clamp_min(1e-12)
    inverse = (eigenvalues.clamp_min(0) / scale + RIDGE).rsqrt()
    # Keep average squared gain at one so the metric changes directions.
    inverse /= inverse.square().mean().sqrt()
    metric = (eigenvectors * inverse) @ eigenvectors.T
    bases = {}
    for mode, moment in zip(MOMENTS[1:], target, strict=True):
        _, vectors = torch.linalg.eigh(metric @ moment @ metric)
        bases[mode] = vectors.flip(1).T.float()
    return metric.float(), bases


class MetricLoRA(LocalLoRA):
    def __init__(self, weight, rank, metric, basis):
        super().__init__(weight, rank)
        self.register_buffer("metric", metric.to(weight))
        with torch.no_grad():
            lengths = self.down.norm(dim=1, keepdim=True)
            if basis is None:
                basis = torch.linalg.qr(self.down.T).Q.T
            else:
                basis = basis[:rank].to(weight)
            mapped_norm = (basis @ self.metric).norm(dim=1, keepdim=True)
            self.down.copy_(basis * (lengths / mapped_norm.clamp_min(1e-12)))

    def forward(self, original):
        with torch.autocast("cuda", enabled=False):
            return original + ((self.up @ self.down) @ self.metric).view_as(original)


def attach(model, mode, stages, eigens):
    if mode in ("local_lora", "local_matched"):
        return adapt(model, mode, stages)
    model.eval().requires_grad_(False)
    selected = {}
    for stage in stages:
        for name, rank in stage["lora_ranks"].items():
            module = model.get_submodule(name)
            metric, bases = eigens[name]
            adapter = MetricLoRA(module.weight, rank, metric, bases.get(mode))
            parametrize.register_parametrization(module, "weight", adapter)
            selected[name + ".weight"] = None
    return selected


def state_copy(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def training_plan(author, stages):
    files = (TRAIN_CARD, LMH, REPLAY, RW_CARD, V33 / "rw_data.json", V33 / "protocol.json",
             SOURCE_ROOT / "boundary_refinement/training/crop_context.py", SOURCE_ROOT / "boundary_refinement/training/selective_adaptation.py")
    return dict(
        hypothesis="Within-scene real-haze feature differences improve source-retaining transfer more than marginal or scene-shuffled target features at fixed per-stage LoRA capacity.",
        capacity=sum(s["budget"] for s in stages), stages=stages, author=author,
        arms=list(ARMS), main=MAIN, seeds=list(SEEDS), square_crop=SIDE, updates=STEPS * PASSES,
        basis_steps=PARALLEL_CROP_STEPS, basis_seed=BASIS_SEED, sparse_coordinate_seed=MASK_SEED,
        metric_ridge=RIDGE, code_SHA256=digest(__file__),
        inputs={record_path(p): digest(p) for p in files},
        optimizer="AdamW, peak 0.00012, cosine end 0.00002, common replay, fixed BN, BF16 CUDA",
        basis="The frozen author 1x1 inputs give uncentered moments on training batches only. The source inverse regularized covariance root normalizes all metric variants. Initialize both-trainable LoRA down factors from matched-norm directions. Marginal uses target features directly. Shuffled differences roll the other scene within each batch. Paired differences subtract aligned light from dense features. The random metric control uses QR directions.",
        evaluation="Select one scale per arm across seeds, on the eight existing LMHaze calibration scenes and the existing RW reserve, at the registered source budget. Score the previously used 13 LMHaze test scenes and the 46 V33 RW captures after the choice is fixed. Previous experiments used these data, so the tests are repeated benchmarks.",
        compare_to=["local_lora", "local_matched", "source_metric", "marginal", "shuffled"],
        alpha_grid=[0., *ALPHAS], source_floor_db=SOURCE_FLOOR, target_margin_db=TARGET_MARGIN,
    )


def get_basis(model, stages, target, source, signature):
    path = ROOT / "moments.pth"
    card = ROOT / "moments.json"
    if card.exists():
        record = read(card)
        if record["protocol_SHA256"] != signature or record["SHA256"] != digest(path):
            raise ValueError("Training moments changed")
        return torch.load(path, map_location="cpu", weights_only=True)
    tables = collect(model, stages, target, source)
    if any(not value.isfinite().all() for value in tables.values()):
        raise ValueError("Nonfinite features in the training moments")
    torch_save(path, tables)
    atomic_write(card, dict(protocol_SHA256=signature, SHA256=digest(path), finished_utc=now()))
    return tables


def fit(smoke=False):
    configure_cuda()
    model, author = load_pgl()
    stages = layout(model)
    initial = state_copy(model)
    param_names = set(dict(model.named_parameters()))
    plan = training_plan(author, stages)
    protocol = ROOT / "protocol.json"
    target, source = training_scenes(), originals()
    if smoke:
        moment = collect(model, stages, target, source)
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        if protocol.exists() and read(protocol) != plan:
            raise ValueError("The paired-basis protocol changed after registration")
        if not protocol.exists():
            atomic_write(protocol, plan)
        moment = get_basis(model, stages, target, source, digest(protocol))
    eigens = {name: eigenspaces(x) for name, x in moment.items()}
    selected_seeds = SEEDS[:1] if smoke else SEEDS
    ledger = []
    for seed in selected_seeds:
        for mode in ARMS:
            path = ROOT / f"{mode}-{seed}.pth"
            receipt = path.with_suffix(".json")
            if not smoke and receipt.exists():
                row = read(receipt)
                if row["protocol_SHA256"] != digest(protocol) or row["checkpoint_SHA256"] != digest(path):
                    raise ValueError("A matched adapter checkpoint changed")
                ledger.append(row)
                continue
            set_seed(seed)
            model.load_state_dict(initial, strict=True)
            selected = attach(model, mode, stages, eigens)
            params = [p for p in model.parameters() if p.requires_grad]
            trainable = sum(p.numel() for p in params)
            if trainable != plan["capacity"]:
                raise ValueError(f"{mode} uses {trainable} trainable scalars")
            optimizer = torch.optim.AdamW(params, lr=.00012, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, plan["updates"], eta_min=.00002)
            started = time.perf_counter()
            for step, (fog, target_rgb, source_rgb) in enumerate(stream(target, source, seed), 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    restored = model(fog).float()
                output, replay = restored.split((2 * PAIRS, PAIRS))
                target_loss, _ = objective(output, target_rgb, None, "point")
                low, high = band_parts(replay - source_rgb)
                loss = target_loss + (low.square() + 1e-6).sqrt().mean() + .2 * high.abs().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                schedule.step()
                if smoke and step == 2:
                    break
            if not smoke and step != plan["updates"]:
                raise ValueError("The common update budget changed")
            with torch.no_grad():
                before = model(fog[:1])
                state = materialize(model, selected)
                torch.testing.assert_close(model(fog[:1]), before, rtol=0., atol=1e-6)
            changed = verify_changes(initial, state, param_names, selected, mode)
            stats = dict(arm=mode, seed=seed, updates=step, trainable=trainable, changed=changed,
                         loss=loss.item(), seconds=round(time.perf_counter() - started, 2))
            print(stats, flush=True)
            if not smoke:
                torch_save(path, state)
                row = dict(stats, file=str(path), checkpoint_SHA256=digest(path),
                           protocol_SHA256=digest(protocol), finished_utc=now())
                atomic_write(receipt, row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(
            runs=ledger, updates=sum(row["updates"] for row in ledger), protocol_SHA256=digest(protocol),
            gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now(),
        ))


def completed():
    protocol = ROOT / "protocol.json"
    check = read(ROOT / "completed.json")
    if check["protocol_SHA256"] != digest(protocol):
        raise ValueError("The paired-basis training plan changed")
    if {(r["arm"], r["seed"]) for r in check["runs"]} != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("The matched seed matrix is incomplete")
    for row in check["runs"]:
        if row["checkpoint_SHA256"] != digest(row["file"]):
            raise ValueError("A completed checkpoint changed")
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    fit(parser.parse_args().smoke)

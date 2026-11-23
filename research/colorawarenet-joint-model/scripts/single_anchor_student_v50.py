"""Distill low-frequency RGB corrections into one backbone with spatial controls."""
import argparse
from collections import Counter, defaultdict
from statistics import mean, median
from time import perf_counter

import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.author_pgl_v25 import pgl_rgb
from scripts.coarse_semantics_v39 import rgb_checkpoint
from scripts.conditional_fusion_v29 import BEDDE, city_manifest, city_pairs
from scripts.cross_scale_rgb_v44 import resize, same_box
from scripts.edge_projection_v47 import draw
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_cross_scale_rgb_v44 import psnr, visible_edges
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_parallel_rgb_v8 import bind
from scripts.evaluate_region_restore_v48 import calibrate_data, exact_stats, render_rgb, test_data
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.region_restore_v48 import FLOOR_DB, GRID, SCALES, SEEDS, ROOT as V48, LowField, load_heads, view_size
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save
from scripts.veil_response_v45 import training_pairs


ROOT = REPO / "results/single-anchor-student-v50"
MODES = ("plain", "uniform", "reliable", "shifted")
STEPS = 512


def pack(x, base, second=None):
    shape = view_size(x, GRID, 8)
    signal, native = [F.adaptive_avg_pool2d(t, shape) for t in (x, base)]
    context = same_box(native, 5) if second is None else F.adaptive_avg_pool2d(second, shape)
    var = (F.adaptive_avg_pool2d(x.square(), shape) - signal.square()).clamp_min(0.)
    features = torch.cat((2. * signal - 1., 2. * native - 1., 2. * context - 1., 4. * var.sqrt()), dim=1)
    return features, native, context, None, None


def weighting(native, teacher, truth):
    native_error, target_error = ((v - truth).square().mean(1, keepdim=True) for v in (native, teacher))
    improvement = (native_error - target_error) / (.005 + native_error + target_error)
    weight = .05 + (4. * improvement).sigmoid()
    return weight / weight.mean((-1, -2), keepdim=True)


def heads_for_teacher():
    direct_alpha = read(V48 / "calibration.json")["amplitudes"]["direct"]
    models = load_heads(("direct",))
    bind()
    expert, card = load_author_reference()
    return direct_alpha, models, expert, card


@torch.no_grad()
def train_data(anchor, teachers, smoke):
    amount, models, mct, _ = teachers
    examples = defaultdict(list)
    benefits = defaultdict(list)
    for ds, _, _, hazy, clean in training_pairs():
        if smoke and len(examples[ds]) >= 3:
            continue
        images = torch.stack((hazy, clean)).cuda().float() / 255.
        x, y = resize(images, view_size(images, 768), antialias=True).split(1)
        base, second = pgl_rgb(anchor, x).clamp(0, 1), author_rgb(mct, x).clamp(0, 1)
        aid = pack(x, base, second)
        alone = pack(x, base)
        clean_grid = F.adaptive_avg_pool2d(y, alone[1].shape[-2:])
        deltas = [net(aid, "direct") for net in models.values()]
        teacher = (alone[1] + amount * torch.stack(deltas).mean(0)).clamp(0, 1)
        w = weighting(alone[1], teacher, clean_grid)
        examples[ds].append((alone, teacher, clean_grid, w))
        benefits[ds].append(float(((teacher - clean_grid).square().mean(1) < (alone[1] - clean_grid).square().mean(1)).float().mean()))
    counts = {k: len(v) for k, v in examples.items()}
    if counts != (dict(lm=3, rw=3, nh=3) if smoke else dict(lm=128, rw=86, nh=36)):
        raise ValueError(f"The frozen teacher received an unexpected population {counts}")
    print(dict(training_counts=counts, teacher_helpful_region_fraction={k: mean(v) for k, v in benefits.items()}), flush=True)
    return tuple(examples[k] for k in ("lm", "rw", "nh"))


def objective(model, sample, mode):
    context, teacher, truth, weight = sample
    native = context[1]
    prediction = (native + model(context, "direct")).clamp(0, 1)
    denominator = (native - truth).square().mean().clamp_min(.0025)
    supervise = ((prediction - truth).square().mean() / denominator + .25).log()
    if mode == "plain":
        return supervise
    trust = weight if mode == "reliable" else 1.
    if mode == "shifted":
        trust = weight.roll(tuple(n // 2 for n in weight.shape[-2:]), dims=(-2, -1))
    return supervise + .5 * ((prediction - teacher).square() * trust).mean() / denominator


def fit(smoke):
    configure_cuda()
    anchor, rgb_hash = rgb_checkpoint()
    teachers = heads_for_teacher()
    plan = dict(steps=STEPS, seeds=list(SEEDS), arms=list(MODES), trainable=sum(p.numel() for p in LowField().parameters()),
        code_SHA256=digest(__file__), parent_fit_SHA256=digest(V48 / "completed.json"),
        parent_calibration_SHA256=digest(V48 / "calibration.json"), native_pgl_SHA256=rgb_hash,
        teacher="Mean of all three V48 direct-field predictions at the fixed amplitude. Use V48's post-development direct field because it was the best LMHaze trained arm. These demonstrations remain development evidence.",
        controls="Plain supervised student, uniform distillation, aligned reliability and spatially rolled reliability. Weight means are identical per image. Select a single student amplitude on the same calibration sets. Every arm has the same samples, optimizer, schedule and single PGL backbone.",
        evaluation="Compare to baseline, the frozen complete V48 direct teacher, and other equal-capacity students. Use the fixed V48 evaluation pairs plus full BeDDE with its provided valid region masks. No BeDDE or REVIDE fits/calibration occur in V50.",
        bedde_SHA256=digest(BEDDE), useful_gain_db=.10, source_floor_db=FLOOR_DB,
    )
    lock = ROOT / "protocol.json"
    if not smoke:
        ROOT.mkdir(parents=True, exist_ok=True)
        if lock.exists() and read(lock) != plan:
            raise ValueError("The single-backbone training registry changed")
        if not lock.exists():
            atomic_write(lock, plan)
    banks = train_data(anchor, teachers, smoke)
    receipts = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            file = ROOT / f"{mode}-{seed}.pth"
            record = file.with_suffix(".json")
            if not smoke and record.exists():
                old = read(record)
                if old["checkpoint_SHA256"] != digest(file) or old["plan_SHA256"] != digest(lock):
                    raise ValueError("An existing student weight changed")
                receipts.append(old)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed + 50227)
            net = LowField().cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=.0017, weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.0001)
            started = perf_counter()
            for step in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                a, b, c = (p[int(torch.randint(len(p), (), generator=rng))] for p in banks)
                total = objective(net, a, mode) + .75 * (objective(net, b, mode) + objective(net, c, mode))
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2., error_if_nonfinite=True)
                opt.step()
                schedule.step()
            item = dict(mode=mode, seed=seed, updates=step + 1, loss=total.item(), seconds=round(perf_counter() - started, 2))
            print(item, flush=True)
            if not smoke:
                torch_save(file, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                item.update(checkpoint_SHA256=digest(file), plan_SHA256=digest(lock))
                atomic_write(record, item)
                receipts.append(item)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=receipts, updates=sum(q["updates"] for q in receipts),
            protocol_SHA256=digest(lock), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def load_students():
    meta = read(ROOT / "completed.json")
    protocol = ROOT / "protocol.json"
    if meta["protocol_SHA256"] != digest(protocol) or read(protocol)["code_SHA256"] != digest(__file__):
        raise ValueError("The student protocol changed")
    if {(i["mode"], i["seed"]) for i in meta["runs"]} != {(m, s) for m in MODES for s in SEEDS}:
        raise ValueError("Matched student training is incomplete")
    result = {}
    for i in meta["runs"]:
        path = ROOT / f'{i["mode"]}-{i["seed"]}.pth'
        if digest(path) != i["checkpoint_SHA256"]:
            raise ValueError("The student checkpoint changed")
        model = LowField().cuda()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        result[i["mode"], i["seed"]] = model.eval().requires_grad_(False)
    return result


@torch.inference_mode()
def calibrate(anchor, models):
    result = ROOT / "calibration.json"
    sig = digest(ROOT / "completed.json")
    if result.exists():
        card = read(result)
        if card["fit_SHA256"] != sig:
            raise ValueError("The fixed student calibration changed")
        return card
    values, counts = defaultdict(list), Counter()
    for domain, stream in calibrate_data():
        for _, _, _, x, y in stream:
            raw, truth = as_float(x), as_float(y)
            base = pgl_rgb(anchor, raw).clamp(0, 1)
            prompt = pack(raw, base)
            fixed = psnr(base, truth)
            for (mode, _), model in models.items():
                delta = resize(model(prompt, "direct"), base.shape[-2:])
                for a in SCALES:
                    difference = psnr((base + a * delta).clamp(0, 1), truth) - fixed if a else 0.
                    values[domain, mode, a].append(difference)
            counts[domain] += 1
    if counts != dict(lm=64, rw=47, nh=9):
        raise ValueError("The student received a different calibration dataset")
    table = {k: mean(v) for k, v in values.items()}
    eligible = {m: [a for a in SCALES if table["rw", m, a] >= FLOOR_DB] for m in MODES}
    selected = {m: max(eligible[m], key=lambda a: (mean(table[d, m, a] for d in ("lm", "nh")), -a)) for m in MODES}
    card = dict(scales=selected, counts=dict(counts), fit_SHA256=sig,
        means={"/".join(map(str, k)): val for k, val in table.items()}, finished_utc=now())
    atomic_write(result, card)
    print(dict(student_calibrated_scale=selected), flush=True)
    return card


def paired_test_data():
    for row in test_data():
        yield (*row, None)
    yield from city_pairs(city_manifest())


def area_psnr(value, truth, roi):
    if roi is None:
        return psnr(value, truth)
    err = ((value - truth).square() * roi).sum() / (roi.sum() * 3)
    return float(-10. * err.clamp_min(1e-12).log10())


@torch.inference_mode()
def test():
    configure_cuda()
    anchor, _ = rgb_checkpoint()
    models = load_students()
    calibration = calibrate(anchor, models)
    strength, teachers, external, card = heads_for_teacher()
    counts, values, paired, every, panels = Counter(), defaultdict(list), defaultdict(lambda: defaultdict(list)), [], []
    for ds, name, group, hazy, clear, roi in paired_test_data():
        raw, truth = as_float(hazy), as_float(clear)
        native = pgl_rgb(anchor, raw).clamp(0, 1)
        student_input = pack(raw, native)
        challenger = author_rgb(external, raw).clamp(0, 1)
        teacher_input = pack(raw, native, challenger)
        options = {("native", 0): native, ("mct", 0): challenger}
        options.update({("teacher", seed): render_rgb(native, net(teacher_input, "direct"), strength)
                        for (_, seed), net in teachers.items()})
        options.update({(m, seed): render_rgb(native, net(student_input, "direct"), calibration["scales"][m])
                        for (m, seed), net in models.items()})
        support = visible_edges(raw, truth) if roi is None else None
        region = roi.to(raw.device) if roi is not None else None
        within = defaultdict(list)
        for (method, seed), prediction in options.items():
            metrics = (exact_stats(prediction, truth, support, student_input[1].shape[-2:]) if region is None
                       else dict(psnr=area_psnr(prediction, truth, region)))
            every.append(dict(dataset=ds, id=name, group=group, method=method, seed=seed, **metrics))
            within[method].append(metrics["psnr"])
        for m, value in within.items():
            values[ds, m].append(mean(value))
            paired[ds, m][group].append(mean(within["reliable"]) - mean(value))
        counts[ds] += 1
        if counts[ds] == 1:
            sample = (raw, native, options["teacher", SEEDS[0]], options["plain", SEEDS[0]],
                      options["reliable", SEEDS[0]], truth)
            panels.append((f"{ds} / {name}", [to_rgb(t) for t in sample],
                 ("Hazy input", "Native PGL", "Two-backbone teacher", "Plain one-backbone head", "Region-weighted student", "Reference")))
        if counts[ds] % 40 == 0:
            print(dict(dataset=ds, total=counts[ds]), flush=True)
    expected = dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=16, nhhaze2=5, bedde=208)
    if counts != expected or len(every) != sum(expected.values()) * 17:
        raise ValueError(f"Unexpected measured population {counts}")
    summary = {"/".join(k): mean(v) for k, v in values.items()}
    intervals = {"/".join(k): bootstrap({g: mean(v) for g, v in vals.items()}) for k, vals in paired.items()}
    atomic_write(ROOT / "individual.json", every)
    draw(ROOT / "predictions.png", panels)
    atomic_write(ROOT / "test.json", dict(counts=dict(counts), rows=len(every), mean_psnr=summary, scales=calibration["scales"],
        reliable_vs=intervals, mct=card, calibration_SHA256=digest(ROOT / "calibration.json"),
        table_SHA256=digest(ROOT / "individual.json"), image_SHA256=digest(ROOT / "predictions.png"), finished_utc=now()))
    for k, v in summary.items():
        print(k, round(v, 4), flush=True)


@torch.inference_mode()
def benchmark():
    configure_cuda()
    anchor, _ = rgb_checkpoint()
    students = load_students()
    strength, teachers, challenger, _ = heads_for_teacher()
    rgb, _, _, fog, _ = next(iter(test_data()))
    x = as_float(fog)
    x = resize(x, (1080, 1920), antialias=True)
    for mode, number in (("plain", 1), ("reliable", 1), ("teacher", 2)):
        student = mode in MODES
        net = (students[mode, SEEDS[0]] if student else teachers["direct", SEEDS[0]])
        scale = read(ROOT / "calibration.json")["scales"][mode] if student else strength
        def forward():
            b = pgl_rgb(anchor, x).clamp(0, 1)
            m = None if student else author_rgb(challenger, x).clamp(0, 1)
            return render_rgb(b, net(pack(x, b, m), "direct"), scale)
        for _ in range(4):
            forward()
        duration = []
        for _ in range(15):
            torch.cuda.synchronize()
            start = perf_counter()
            result = forward()
            torch.cuda.synchronize()
            duration.append((perf_counter() - start) * 1000)
            if result.shape[-2:] != (1080, 1920):
                raise ValueError("Native latency probe returned the wrong image size")
        print(dict(method=mode, backbone_passes=number, milliseconds=round(median(duration), 2)), flush=True)
        atomic_write(ROOT / f"latency-{mode}.json", dict(method=mode, backbone_passes=number, milliseconds=duration,
            median_ms=median(duration), dimensions=[1920, 1080], dataset=rgb, gpu=torch.cuda.get_device_name(), finished_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "test", "latency", "data-check"))
    args = parser.parse_args()
    if args.stage in ("smoke", "fit"):
        fit(args.stage == "smoke")
    elif args.stage == "latency":
        benchmark()
    elif args.stage == "data-check":
        counts = Counter(v[0] for v in paired_test_data())
        if counts != dict(lmhaze=130, rw_new=46, revide_external=32, revide_final=16, nhhaze2=5, bedde=208):
            raise ValueError(f"Unexpected testing population {counts}")
        print(counts, flush=True)
    else:
        test()

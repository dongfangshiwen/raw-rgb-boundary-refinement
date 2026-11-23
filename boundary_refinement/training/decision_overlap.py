"""Train the candidate-weight solver on the exact reconstructed RGB risk."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import PurePosixPath
from statistics import mean
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_overlap_quadratic import POPULATION, samples
from boundary_refinement.evaluation.photographic.evaluate_region_restore import calibrate_data
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.exact_overlap import OFFSETS, matrix_blocks
from boundary_refinement.data.cohorts.fresh_real import CARD as FRESH, RR, registration, image
from boundary_refinement.training.overlap_quadratic import ROOT as PREVIOUS, LocalHead, SEEDS, decoded, direct, geometry, label, quadratic, references, simplex
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor
from boundary_refinement.data.real_haze_data import fingerprint
from boundary_refinement.models.restoration.region_restore import view_size
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save
from boundary_refinement.evaluation.paired.veil_response import TRAIN_CARD, training_pairs


ROOT = REPO / "results/decision-overlap-v55"
MANIFEST = ROOT / "training_extension.json"
MODES = ("direct", "diagonal", "full_upper", "full_image")
STEPS, ITERATIONS = 320, 12
SCALES = (0., .25, .5, .75, 1.)


def pair_index(archive, roles):
    result = defaultdict(dict)
    for name in archive.namelist():
        n = PurePosixPath(name)
        if len(n.parts) == 4 and n.parts[1] in roles and n.suffix == ".png":
            level, role, kind, _ = n.parts
            key = "haze" if kind == "hazy" else "clean"
            result[role, level, n.stem][key] = name
    if any(set(v) != {"haze", "clean"} for v in result.values()):
        raise ValueError("The published RRSHID fit or validation pairs mismatch")
    return result


def extension():
    rr, _ = registration()
    if MANIFEST.exists():
        prior = read(MANIFEST)
        if prior["source_SHA256"] != rr["SHA256"] or prior["fixed_evaluation_SHA256"] != digest(FRESH):
            raise ValueError("The extra development pairs changed")
        return prior
    held = [r for r in read(FRESH)["records"] if r["dataset"] == "rrshid"]
    forbidden = {r["id"].split("/")[-1] for r in held}
    planned = []
    with ZipFile(RR / "RRSHID.zip") as archive:
        records = pair_index(archive, {"train", "val"})
        signatures = [(r["id"], fingerprint(image(archive, r["clean"]))) for r in held]
        for role, count in (("val", 24), ("train", 80)):
            for level in ("thin_fog", "moderate_fog", "thick_fog"):
                keys = [n for r, d, n in records if (r, d) == (role, level)]
                keys.sort(key=lambda k: sha256(f"decision-v55/{role}/{level}/{k}".encode()).digest())
                accepted = 0
                for number in keys:
                    if number in forbidden:
                        continue
                    pair = records[role, level, number]
                    haze, clean = (image(archive, pair[k]) for k in ("haze", "clean"))
                    if haze.size != clean.size or haze.size != (256, 256):
                        raise ValueError("Publication geometry disagrees with the development grid")
                    p, n = fingerprint(clean)
                    if any(np.count_nonzero(p != q) <= 8 and float(n @ m) >= .96 for _, (q, m) in signatures):
                        continue
                    key = f"{level}/{number}"
                    planned.append(dict(role=role, level=level, id=key, **pair))
                    forbidden.add(number)
                    signatures.append((key, (p, n)))
                    accepted += 1
                    if accepted == count:
                        break
                if accepted != count:
                    raise ValueError("Too few release crops for the prespecified split")
    card = dict(source_SHA256=rr["SHA256"], fixed_evaluation_SHA256=digest(FRESH), pairs=planned,
        counts=dict(Counter(r["role"] for r in planned)),
        rule="Use official training and validation only. Draw 80 training and 24 validation pairs per fog condition by a fixed filename hash. Exclude collision with any official test numeric ID and clear-reference near duplicates before fitting. The evaluation split has previously been seen by V53 and V54, so V55 is calibrated development evidence.",
        finished_utc=now())
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write(MANIFEST, card)
    print(dict(rrshid_extension=card["counts"]), flush=True)
    return card


def rr_records(role):
    card = extension()
    with ZipFile(RR / "RRSHID.zip") as z:
        for r in card["pairs"]:
            if r["role"] == role:
                yield ("rr", r["id"], r["id"], *(image_tensor(image(z, r[k])) for k in ("haze", "clean")))


def unroll(gram, z, diagonal):
    d = gram.diagonal(dim1=-1, dim2=-2)
    g = torch.diag_embed(d) if diagonal else gram
    jitter = (.02*d.mean(-1, keepdim=True)).clamp_min(1e-6)
    g = g + torch.diag_embed(jitter.expand_as(d))
    rate = g.abs().sum(-1).amax(-1, keepdim=True).clamp_min(1e-6).reciprocal()
    a = torch.zeros_like(z)
    for _ in range(ITERATIONS):
        a = simplex(a - rate * ((g @ a.unsqueeze(-1)).squeeze(-1) - z))
    return a


def action(net, cues, gram, rms, mode):
    code = net(cues)
    if mode == "direct":
        return direct(code), None
    center = .15 * code
    return unroll(gram, center * rms, mode == "diagonal"), center


def exact(terms, a, truth):
    neighborhood = torch.stack([a.roll((-i, -j), dims=(1, 2)) for i, j in OFFSETS])
    second_term = (terms @ neighborhood.unsqueeze(-1)).squeeze(-1).sum(0)
    return ((second_term - 2.*truth) * a).sum(-1)


@torch.no_grad()
def make_bank(locked, smoke):
    cache = defaultdict(list)
    inputs = list(training_pairs()) + list(rr_records("train"))
    counts = Counter()
    for domain, _, _, x, y in inputs:
        if smoke and counts[domain] >= 3:
            continue
        pair = torch.cat((as_float(x), as_float(y)))
        pair = resize(pair, view_size(pair, 576), antialias=True)
        variants = (pair,) if domain == "rr" else (pair, pair.pow(1.8))
        for item in variants:
            fog, target = item.split(1)
            g = geometry(fog, locked)
            ideal, error = label(g, target)
            matrices = matrix_blocks(g)
            terms = torch.stack([matrices[offset] for offset in OFFSETS])
            weight = g.partition.mass / g.partition.mass.sum()
            cache[domain].append((g.features, g.gram, g.scale, terms, ideal, error, weight))
        counts[domain] += 1
    if counts != (dict(lm=3, rw=3, nh=3, rr=3) if smoke else dict(lm=128, rw=86, nh=36, rr=240)):
        raise ValueError("The exact risk training pool differs")
    print(dict(fit_counts=dict(counts), views={k: len(v) for k, v in cache.items()}), flush=True)
    return tuple(cache.values())


def objective(net, sample, mode):
    x, g, local, blocks, y, base, mass = sample
    a, center = action(net, x, g, local, mode)
    value = quadratic(g, y, a) if mode == "full_upper" else exact(blocks, a, y)
    normalizer = (base * mass).sum().clamp_min(.0025)
    total = (value * mass).sum() / normalizer
    if center is not None:
        target = (y / local).clamp(-.35,.35)
        total = total + .10 * F.smooth_l1_loss(center, target, beta=.05)
    return total


def fit(smoke=False):
    configure_cuda()
    extra = extension()
    lock = ROOT / "protocol.json"
    protocol = dict(code_SHA256=digest(__file__), data_SHA256=digest(MANIFEST), modes=list(MODES), seeds=list(SEEDS),
        parameters=sum(p.numel() for p in LocalHead().parameters()), steps=STEPS, projections=ITERATIONS,
        rgb_proposals_SHA256=digest(PREVIOUS / "protocol.json"), source_SHA256=digest(TRAIN_CARD),
        rule="Train through the constrained solver to optimize exact image-space error. Full and diagonal see the same pairs and model, with an identical exact supervision. The full_upper ablation changes the objective to the separable upper bound. Direct learns softmax weights with exact supervision. Both old and RRSHID official training pools are fixed.",
        evaluation="Commit calibration on the 120 old validation and 72 official RRSHID validation pairs. Re-evaluate the registered collection only. RRSHID Test has no weights/calibration exposure, but has been used for development diagnostics by other runs.")
    if not smoke:
        if lock.exists() and read(lock) != protocol:
            raise ValueError("The fixed decision-training protocol differs")
        if not lock.exists():
            atomic_write(lock, protocol)
    bank = make_bank(references(), smoke)
    ledger = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            filename = ROOT / f"{mode}-{seed}.pth"
            run_log = filename.with_suffix(".json")
            if not smoke and run_log.exists():
                known = read(run_log)
                if known["weights_SHA256"] != digest(filename) or known["protocol_SHA256"] != digest(lock):
                    raise ValueError("A previously fitted decision checkpoint changed")
                ledger.append(known)
                continue
            set_seed(seed)
            rng = torch.Generator().manual_seed(seed+55231)
            net = LocalHead().cuda()
            with torch.no_grad():
                net.layers[-1].bias.fill_(.10)
            opt = torch.optim.AdamW(net.parameters(), lr=.0011, weight_decay=.01)
            lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00009)
            for step in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                losses = [objective(net, p[int(torch.randint(len(p), (), generator=rng))], mode) for p in bank]
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
                opt.step()
                lr.step()
            row = dict(mode=mode, seed=seed, steps=step+1, loss=loss.item())
            print(row, flush=True)
            if not smoke:
                torch_save(filename, {k: v.cpu().clone() for k,v in net.state_dict().items()})
                row.update(weights_SHA256=digest(filename), protocol_SHA256=digest(lock))
                atomic_write(filename.with_suffix(".json"), row)
                ledger.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=ledger, updates=sum(r["steps"] for r in ledger), protocol_SHA256=digest(lock),
            extra_counts=extra["counts"], torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), finished_utc=now()))


def models():
    ledger = read(ROOT / "completed.json")
    lock = ROOT / "protocol.json"
    if ledger["protocol_SHA256"] != digest(lock) or not matches_registered(read(lock)['code_SHA256'], digest(__file__)):
        raise ValueError("A fixed decision-training source changed")
    bank = {}
    for r in ledger["runs"]:
        key = (r["mode"], r["seed"])
        path = ROOT / f"{key[0]}-{key[1]}.pth"
        if digest(path) != r["weights_SHA256"]:
            raise ValueError("A frozen model changed")
        head = LocalHead().cuda()
        head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[key] = head.eval().requires_grad_(False)
    if set(bank) != {(m,s) for m in MODES for s in SEEDS}:
        raise ValueError("Missing same-capacity ablation weights")
    return bank


def coefficients(g, bank):
    result = {k: action(net, g.features, g.gram, g.scale, k[0])[0] for k, net in bank.items()}
    result.update({("frozen_diagonal", s): action(bank["full_image",s], g.features, g.gram, g.scale, "diagonal")[0] for s in SEEDS})
    return result


def calibrate(locked, bank):
    path = ROOT / "calibration.json"
    stamp = digest(ROOT / "completed.json")
    if path.exists():
        known = read(path)
        if known["training_SHA256"] != stamp:
            raise ValueError("The calibrated solver and weights do not match")
        return known
    table, count = defaultdict(list), Counter()
    for d, rows in [*calibrate_data(), ("rr", rr_records("val"))]:
        for _, _, _, a, b in rows:
            x, y = as_float(a), as_float(b)
            g = geometry(x, locked)
            score = area_psnr(g.base, y, None)
            for (m, _), w in coefficients(g, bank).items():
                for s in SCALES:
                    table[d,m,s].append(area_psnr(decoded(g,w,s),y,None) - score if s else 0.)
            count[d] += 1
    if count != dict(lm=64, rw=47, nh=9, rr=72):
        raise ValueError(f"Unexpected calibration counts {count}")
    quality = {k: mean(v) for k,v in table.items()}
    selected = {m: max((s for s in SCALES if quality["rw",m,s] >= -.25),
        key=lambda s: (mean(quality[d,m,s] for d in ("lm","nh","rr")),-s)) for m in MODES}
    selected["frozen_diagonal"] = selected["full_image"]
    record = dict(used=selected, counts=dict(count), training_SHA256=stamp,
        values={"/".join(map(str,k)): v for k,v in quality.items()}, finished_utc=now())
    atomic_write(path, record)
    print(dict(calibration=selected), flush=True)
    return record


@torch.inference_mode()
def test():
    configure_cuda()
    locked = references()
    heads = models()
    card = calibrate(locked, heads)
    path = ROOT / "evaluation_protocol.json"
    plan = dict(code_SHA256=digest(__file__), data_SHA256=digest(FRESH), source=POPULATION, scales=card["used"],
        compare="Full-image training against a direct gate of equal capacity, a re-trained diagonal solver, a frozen diagonal, and upper-bound supervision. Reuse V53 evaluation; no weights or calibrations use the official test pairs.")
    if path.exists() and read(path) != plan:
        raise ValueError("The fixed follow-up test differs")
    if not path.exists():
        atomic_write(path, plan)
    counts, rows, panels, metrics, grouped = Counter(), [], [], defaultdict(list), defaultdict(lambda: defaultdict(list))
    for d, k, cluster, a, b, roi, condition in samples():
        x,y = as_float(a),as_float(b)
        ctx = geometry(x,locked)
        inside = roi.cuda() if roi is not None else None
        stored = {("anchor",0): ctx.base, ("mct",0): ctx.base + ctx.deltas[1:2],
            ("mix",0): ctx.base + .50*ctx.deltas[1:2]}
        stored.update({(m,s): decoded(ctx,w,card["used"][m]) for (m,s),w in coefficients(ctx,heads).items()})
        by_mode = defaultdict(list)
        for (m,s), out in stored.items():
            value=area_psnr(out,y,inside)
            by_mode[m].append(value)
            rows.append(dict(dataset=d, id=k, group=cluster, condition=condition, method=m, seed=s, psnr=value))
        for m, value in by_mode.items():
            score = mean(value)
            metrics[d,m].append(score)
            grouped[d,m][cluster].append(mean(by_mode["full_image"])-score)
        if counts[d] == 0 and d in ("rrshid","nt_haze","lmhaze","realhyper_csnc"):
            values=(x,ctx.base,stored["mct",0],stored["direct",SEEDS[0]],stored["full_image",SEEDS[0]],y)
            panels.append((d+" / "+k,[to_rgb(v) for v in values],("Hazy input","Native anchor","Published MCT","Direct local weights","Image-trained solver","Clear reference")))
        counts[d] += 1
        if counts[d] % 160 == 0:
            print(dict(dataset=d,evaluated=counts[d]),flush=True)
    if counts != POPULATION:
        raise ValueError(f"The actual test population differs {counts}")
    values={"/".join(k): mean(v) for k,v in metrics.items()}
    intervals={"/".join(k): bootstrap({c:mean(v) for c,v in h.items()}) for k,h in grouped.items()}
    atomic_write(ROOT/"individual.json",rows)
    draw(ROOT/"predictions.png",panels)
    atomic_write(ROOT/"test.json",dict(counts=dict(counts),rows=len(rows), psnr=values, paired=intervals, calibration=card["used"],
        data_SHA256=digest(MANIFEST), protocol_SHA256=digest(path), table_SHA256=digest(ROOT/"individual.json"),
        panel_SHA256=digest(ROOT/"predictions.png"), finished_utc=now()))
    for d in counts:
        print(d,{m:round(values[f"{d}/{m}"],4) for m in ("anchor", "mct", "mix", *MODES,"frozen_diagonal")},flush=True)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("data","smoke","fit","test"))
    args=cli.parse_args()
    if args.stage == "data":
        extension()
    elif args.stage == "test":
        test()
    else:
        fit(args.stage=="smoke")

"""Train the author's window SAPA heads and transfer the donor estimator to fixed JBU."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
from hashlib import sha256
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.utils import set_seed
from boundary_refinement.evaluation.semantic import evaluate_semantic_subpixel as v94
from boundary_refinement.evaluation.semantic import evaluate_transport as v92
from boundary_refinement.models.segmentation import matched_recall as roc
from boundary_refinement.models.operators import donor_colors as method
from boundary_refinement.data.staging import stage_vkitti_fog as virtual
from boundary_refinement.models.operators import subpixel_transport as parent
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band, compress_scores
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/published-affiliation-v95"
SOURCE = REPO.parent / "benchmark_data/sapa-author-v95"
FILES = {"sapa_base.py": "2876707f1186072b961987aae9a827475915a8418a8d86816db1ff93bd289a14",
         "sapa/sapa/forward_check.py": "651caff21fa232e5a4b21dd07f40fab5481d16f99e16e44e688690786d586a58"}
ARMS = ("sapa_rgb", "sapa_context")
SEEDS, STEPS = parent.SEEDS, parent.STEPS
METHODS = ("bilinear", "jbu", "dysample", "dual_free", "dual_core", *ARMS, "jbu_core", "jbu_roll")
PAIRS = (("dual_core", "sapa_rgb"), ("dual_core", "sapa_context"), ("jbu_core", "jbu"),
         ("jbu_core", "jbu_roll"), ("dual_core", "jbu_core"))


def protocol():
    return dict(code_SHA256=digest(__file__), parent_SHA256=digest(method.ROOT / "completed.json"),
                author="https://github.com/tiny-smart/sapa", commit="a5acd520895c69886ce7175010d50a71d50b5a0e",
                source_files=FILES, arms=list(ARMS), seeds=list(SEEDS), steps=STEPS,
                model="Official SAPA class and the two exact authors' PyTorch reference functions from forward_check.py "
                "running on CUDA. Window 5, embedding 32, factor 4, norm=False to accept plain RGB and 19-channel logits. "
                "The second arm additionally receives projected bilinear logits in the high-resolution query. "
                "Both see raw and restored RGB, the same frozen V84 logits, crops, loss and step count as V94. "
                "AdamW 0.002, cosine to 0.0002. This is a decoder-interface comparison, not the paper's complete backbone.",
                zero_fit="In JBU replace only the donor cell average with V94 semantic colors or the same weights rolled. "
                "Keep the spatial range, color bandwidth, 3x3 kernel and all pixels of the original JBU. "
                "Evaluate by recording-paired boundary and full mIoU on the V94 cohorts as a follow-up.")


@lru_cache(maxsize=1)
def author_class():
    # Load only the named author definitions. Their reference file otherwise imports optional compiled extensions.
    ns = dict(torch=torch, nn=nn, F=F, __name__="sapa_author_v95")
    for filename, names in (("sapa/sapa/forward_check.py", {"qk_pytorch", "av_pytorch"}), ("sapa_base.py", {"SAPA"})):
        file = SOURCE / filename
        if digest(file) != FILES[filename]:
            raise ValueError(f"Unexpected published SAPA source: {filename}")
        tree = ast.parse(file.read_text(encoding="utf8"))
        selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in selected} != names:
            raise ValueError("Missing author definitions")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), "exec"), ns)
        if "qk_pytorch" in ns:
            ns.update(qk=ns["qk_pytorch"], av=ns["av_pytorch"])
    return ns["SAPA"]


def model(arm):
    return author_class()(6 if arm == "sapa_rgb" else 25, dim_x=19, up_factor=4,
                          up_kernel_size=5, embedding_dim=32, norm=False).cuda()


def run(arm, net, z, guides, strip=16):
    hi = torch.cat((guides[0], guides[1]), 0)[None].float() / 127.5 - 1.
    low = z / 4.
    if arm == "sapa_context":
        hi = torch.cat((hi, F.interpolate(low, scale_factor=4., mode="bilinear", align_corners=False)), 1)
    h = low.shape[-2]
    fields = []
    for start in range(0, h, strip):
        end = min(start + strip, h)
        y0, y1 = max(0, start - 2), min(h, end + 2)
        pred = 4. * net(hi[..., y0 * 4:y1 * 4, :].contiguous(), low[..., y0:y1, :].contiguous())
        fields.append(pred[..., (start - y0) * 4:(end - y0) * 4, :])
    return torch.cat(fields, dim=-2)


def train_one(arm, seed, pool):
    lock = digest(ROOT / "protocol.json")
    path = ROOT / f"{arm}-{seed}.pth"
    receipt = path.with_suffix(".json")
    if receipt.is_file():
        row = read(receipt)
        if digest(path) != row["weights_SHA256"] or row["protocol_SHA256"] != lock:
            raise ValueError("Existing author fit changed")
        return row
    set_seed(seed)
    net = model(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.0002)
    rng = np.random.default_rng(92001 + seed)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        total = 0.
        for focus in (False, True):
            z, guides, truth = parent.train_crop(pool, rng, SEEDS.index(seed), focus, step)
            result = run(arm, net, z, guides, strip=parent.SIDE)
            cut = (..., slice(8, -8), slice(8, -8))
            target = truth[cut]
            w = (target < 19).float() * (1. + 3. * boundary_band(truth)[cut] + 2. *
                                          ((target == 6) | (target == 7) | (target == 14)))
            ce = F.cross_entropy(result[cut], target[None], ignore_index=255, reduction="none")[0]
            total = total + .5 * (ce * w).sum() / w.sum().clamp_min(1.)
        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        schedule.step()
    torch_save(path, {k: v.cpu() for k, v in net.state_dict().items()})
    row = dict(method=arm, seed=seed, steps=STEPS, params=sum(p.numel() for p in net.parameters()), loss=float(total.detach()),
               weights_SHA256=digest(path), protocol_SHA256=lock, sampler_end=sha256(str(rng.bit_generator.state).encode()).hexdigest())
    atomic_write(receipt, row)
    return row


def fit():
    configure_cuda()
    lock = ROOT / "protocol.json"
    cfg = protocol()
    if lock.exists() and read(lock) != cfg:
        raise ValueError("Published method protocol changed")
    atomic_write(lock, cfg)
    frames = parent.train_frames(parent.DualScaleFeatures(), parent.anchors(), False)
    runs = []
    for s in SEEDS:
        for m in ARMS:
            runs.append(train_one(m, s, frames))
            print(runs[-1], flush=True)
    atomic_write(ROOT / "completed.json", dict(runs=runs, total_updates=len(runs) * STEPS, protocol_SHA256=digest(lock),
                 torch=str(torch.__version__), gpu=torch.cuda.get_device_name(), utc=now()))


def nets():
    lock, card = ROOT / "protocol.json", read(ROOT / "completed.json")
    if not matches_registered(read(lock), protocol()) or card["protocol_SHA256"] != digest(lock):
        raise ValueError("Published method registration mismatch")
    result = {}
    for row in card["runs"]:
        a, s = row["method"], row["seed"]
        p = ROOT / f"{a}-{s}.pth"
        if digest(p) != row["weights_SHA256"] or row["steps"] != STEPS:
            raise ValueError("Bad SAPA checkpoint")
        net = model(a).eval().requires_grad_(False)
        net.load_state_dict(torch.load(p, weights_only=True, map_location="cpu"))
        result[a, s] = net
    if set(result) != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("Missing author checkpoint")
    return result


def joint_bilateral(guides, logits, weight):
    h, w = logits.shape[-2:]
    pixels = parent.pack(guides[0], h, w).float() / 255.
    donors = (pixels * weight[..., None]).sum(1)
    z = logits[0].permute(1, 2, 0).reshape(-1, 19)
    base, offsets, d2 = parent.geometry(logits.device)
    parts = []
    for block in torch.arange(h * w, device=logits.device).split(4096):
        neighbors = ((block[:, None] // w + offsets[:, 0]).clamp(0, h - 1) * w +
                     (block[:, None] % w + offsets[:, 1]).clamp(0, w - 1))
        color = (pixels[block, :, None] - donors[neighbors][:, None]).square().mean(-1)
        mix = (-.5 * (d2 / .65 ** 2 + color / .05 ** 2)).softmax(-1)
        parts.append((mix - base) @ z[neighbors])
    return parent.unpack(torch.cat(parts), h, w)


def summary(rows, key, classes):
    grouped = defaultdict(dict)
    for r in rows:
        if key not in r:
            continue
        ds, g = r["dataset"], roc.capture_id(r["dataset"], r["group"])
        for ds in ((ds, "foggy_driving_101") if ds.startswith("foggy_driving_") and key == "counts" else (ds,)):
            if g not in grouped[ds]:
                grouped[ds][g] = np.zeros((len(METHODS), 3, 19, 19), dtype=np.int64)
            grouped[ds][g][METHODS.index(r["method"]), SEEDS.index(r["seed"])] += r[key]
    out = {}
    for ds, bank in grouped.items():
        n = len(bank)
        draws = np.ones((1, n), np.int64)
        if n > 1:
            draws = np.vstack((draws, np.random.default_rng(92033).multinomial(n, np.ones(n) / n, 2000)))
        data = np.stack(list(bank.values()))
        per = {}
        for j, arm in enumerate(METHODS):
            vals = []
            for d in np.array_split(draws, max(1, len(draws) // 100)):
                cm = (d @ data[:, j].reshape(n, -1)).reshape(-1, 3, 19, 19)
                tp = cm.diagonal(axis1=-2, axis2=-1)
                denom = cm.sum(-2) + cm.sum(-1) - tp
                iou = np.divide(100. * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
                vals.append(iou[..., list(classes)].mean(-1))
            per[arm] = np.concatenate(vals)
        effects = {}
        for a, b in PAIRS:
            delta = (per[a] - per[b]).mean(1)
            effects[f"{a}-vs-{b}"] = dict(diff=float(delta[0]), ci95=np.quantile(delta[1:], [.025, .975]).tolist() if n > 1 else None)
        out[ds] = dict(scores={a: float(v[0].mean()) for a, v in per.items()}, seeds={a: v[0].tolist() for a, v in per.items()},
                       groups=list(bank), effects=effects)
    return out


@torch.inference_mode()
def evaluate(part):
    configure_cuda()
    engine, raws, fitted = parent.DualScaleFeatures(), parent.anchors(), nets()
    source = method.ROOT / f"{part}-individual.json.gz"
    parent_card = read(method.ROOT / f"{part}-test.json")
    if digest(source) != parent_card["rows_SHA256"]:
        raise ValueError("Parent evaluations changed")
    with gzip.open(source, "rt", encoding="utf8") as stream:
        reference = [r for r in json.load(stream) if r["method"] in METHODS]
    by_frame = {(r["dataset"], r["id"], r["seed"]): r for r in reference if r["method"] == "bilinear"}
    iterator = virtual.cases() if part == "virtual" else v92.cases(part)[0]
    rows, visited = [], Counter()
    for ds, key, group, _, rgb, truth in iterator:
        views, restored, _ = engine(rgb)
        guides = parent.render_guides(rgb, restored, views.fine.shape[-2:])
        band = None if ds == "foggy_driving_coarse" else truth.masked_fill(~boundary_band(truth), 255)
        for seed in SEEDS:
            z = raw_anchor(raws["raw_detail", seed], views)
            native = resize(z, truth.shape)
            if confusion_matrix(native.argmax(1)[0], truth, 19).tolist() != by_frame[ds, key, seed]["counts"]:
                raise ValueError("Frozen baseline does not reproduce the registered cohort")
            for arm in (*ARMS, "jbu_core", "jbu_roll"):
                if arm in ARMS:
                    delta = run(arm, fitted[arm, seed], z, guides) - F.interpolate(z, scale_factor=4., mode="bilinear", align_corners=False)
                else:
                    weight = method.color_weights(z, method.PRIMARY if arm == "jbu_core" else "roll_core")
                    delta = joint_bilateral(guides, z, weight)
                pred = (native + resize(delta, truth.shape)).argmax(1)[0]
                row = dict(dataset=ds, id=key, group=group, method=arm, seed=seed, counts=confusion_matrix(pred, truth, 19).tolist())
                if band is not None:
                    row["edge"] = confusion_matrix(pred, band, 19).tolist()
                rows.append(row)
        visited[ds] += 1
        if sum(visited.values()) % 32 == 0:
            print(dict(part=part, frames=sum(visited.values())), flush=True)
    if dict(visited) != parent_card["population"]:
        raise ValueError("Score all frames")
    file = ROOT / f"{part}-additional.json.gz"
    compress_scores(file, rows)
    classes = virtual.CLASSES if part == "virtual" else range(19)
    complete = rows + reference
    tables = {k: summary(complete, key, classes) for k, key in (("full", "counts"), ("edge", "edge"))}
    atomic_write(ROOT / f"{part}-test.json", dict(**tables, count=dict(visited), ledger_SHA256=digest(file),
                 reference_SHA256=digest(source), frozen_SHA256=digest(ROOT / "completed.json"), utc=now()))
    print(part, {ds: t["scores"] for ds, t in tables["edge"].items()}, flush=True)


def check():
    configure_cuda()
    torch.manual_seed(951)
    z = torch.randn(1, 19, 7, 8, device="cuda")
    g = torch.randint(256, (4, 3, 28, 32), dtype=torch.uint8, device="cuda")
    for m in ARMS:
        net = model(m)
        torch.testing.assert_close(run(m, net, z, g, strip=3), run(m, net, z, g, strip=7), atol=1e-6, rtol=1e-6)
    uniform = torch.full((56, 16), 1. / 16, device="cuda")
    torch.testing.assert_close(joint_bilateral(g, z, uniform), v92.fixed_bilateral(g, z), atol=2e-6, rtol=1e-6)
    print("Untiled author forward agreement and uniform-donor JBU equivalence passed.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("steps", nargs="+", choices=("check", "fit", "external", "rain", "snow", "virtual"))
    for part in cli.parse_args().steps:
        if part == "check":
            check()
        elif part == "fit":
            fit()
        else:
            evaluate(part)

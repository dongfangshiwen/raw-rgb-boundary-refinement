"""Repeat the window intervention on the RGB branch that actually feeds the segmentation decoder."""
from collections import Counter, defaultdict
from dataclasses import replace
from hashlib import sha256

import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration import bounded_tone as rgb
from boundary_refinement.training import road_heads as road
from boundary_refinement.models.restoration.complementary_rgb import ColorHead, adjustment
from boundary_refinement.training.exact_overlap import OFFSETS, neighbor_axis
from boundary_refinement.models.restoration.factorial_restoration import tone_loss, tonal_context
from boundary_refinement.metrics.color import srgb_to_lab
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import DATASETS, RealPairs, V22, V23, image_tensor, long_edge
from boundary_refinement.models.restoration.tonal_window import unwindowed
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/road-tone-control-v98"
ARMS = ("luma", "affine", "rolled")
SEEDS, STEPS = road.SEEDS, road.RGB_STEPS


def registration():
    cfg = dict(script=digest(__file__), parent=digest(road.ROOT / "rgb-completed.json"),
               color_metric=digest(rgb.__file__), kitti=digest(road.KITTI),
               real=[digest(p) for p in (V22 / "held_out_images.json", V22 / "data_manifest.json", V23 / "i_haze_manifest.json")],
               seeds=list(SEEDS), arms=list(ARMS), steps=STEPS,
               design="Same frozen author MCT and seed-matched V68 utility. Identical 174 rendered KITTI train views, "
                      "3-image batches, objective, AdamW, LR and 250 updates as the shipping local correction. "
                      "Compare an exact-window replay with constant and spatially shifted windows. Retain unmodified "
                      "deployed heads as a numerical replay check. Two 36-image held driving conditions, then all "
                      "registered NH-holdout, Dense, O-, I-HAZE pairs downscaled isotropically to long edge 576 "
                      "as photographic transfer. Prespecified outcomes PSNR, DE00, Lab chromatic-plane error, and "
                      "anchor-defined luminance bins. No weight selection on these reused evaluations.")
    file = ROOT / "protocol.json"
    if file.exists() and read(file) != cfg:
        raise ValueError("Road intervention changed")
    if not file.exists():
        atomic_write(file, cfg)
    return digest(file)


def geometry(state, arm):
    if arm == "luma":
        return state
    fields = unwindowed(state.anchor)
    if arm == "rolled":
        lum = rgb.luminance(state.anchor)
        fields = fields * rgb.window(state.anchor, lum, "rolled")
    rms = state.partition.average(fields.square().mean(1)).permute(1, 2, 0)[None].clamp_min(1e-8).sqrt()
    # Leave network cues identical, including the true-window RMS cues. Change the intervention only in the renderer.
    return replace(state, fields=torch.cat((state.fields[:3], fields)), raw_rms=rms, projected_rms=rms)


def training_sample(q, target):
    p, fields = q.partition, q.fields[3:]
    moment = torch.einsum("achw,bchw->abhw", fields, fields) / fields.shape[1]
    r = {i: neighbor_axis(p.rows, i) for i in (-1, 0, 1)}
    c = {i: neighbor_axis(p.cols, i) for i in (-1, 0, 1)}
    terms = torch.stack([((r[i].T @ moment @ c[j]) / p.mass).permute(2, 3, 0, 1)[None] for i, j in OFFSETS])
    residual = target - q.anchor
    z = p.average((fields * residual).mean(1)).permute(1, 2, 0)[None]
    return (q.features, q.transform, q.raw_rms, q.projected_rms, q.mass, F.pad(terms, (3, 0, 3, 0)),
            F.pad(z, (3, 0)), residual.square().mean().clamp_min(.0025))


def state(g, bank, seed):
    coeff = road.mix_weights(g, bank["utility", seed], "utility")[0]
    return tonal_context(g, road.decoded(g, coeff))


@torch.no_grad()
def train_inputs(rows, author):
    items = []
    for row in rows:
        if row["role"] == "fit":
            clear = row["rgb"][None].cuda().float() / 255.
            items.extend((road.geometry(fog, author), clear) for fog in road.rendered_samples(row)[0])
    if len(items) != 174:
        raise ValueError("Road input mismatch")
    return items


def fit(rows, author, bank, lock):
    pool = train_inputs(rows, author)
    nets, receipts = {}, []
    for seed in SEEDS:
        with torch.no_grad():
            states = [(state(g, bank, seed), y) for g,y in pool]
        for arm in ARMS:
            set_seed(seed)
            net = ColorHead().cuda()
            path = ROOT / f"{arm}-{seed}.pth"
            note = path.with_suffix(".json")
            if note.exists():
                r = read(note)
                if r["weights"] != digest(path) or r["protocol"] != lock:
                    raise ValueError("Road checkpoint changed")
                net.load_state_dict(torch.load(path, weights_only=True))
            else:
                with torch.no_grad():
                    samples = [training_sample(geometry(q, arm), y) for q,y in states]
                objective = road.batch_loss(tone_loss, samples, "local")
                rng = torch.Generator().manual_seed(seed + 68027)
                opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
                lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00008)
                for _ in range(STEPS):
                    opt.zero_grad(set_to_none=True)
                    loss = objective(net, rng)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.5, error_if_nonfinite=True)
                    opt.step()
                    lr.step()
                torch_save(path, {k: v.detach().cpu() for k, v in net.state_dict().items()})
                r = dict(method=arm, seed=seed, steps=STEPS, protocol=lock, weights=digest(path),
                         sampler=sha256(rng.get_state().numpy().tobytes()).hexdigest(), loss=float(loss.detach()))
                atomic_write(note, r)
            nets[arm, seed] = net.eval().requires_grad_(False)
            receipts.append(r)
            print(arm, seed, flush=True)
    if any(len({r["sampler"] for r in receipts if r["seed"] == s}) != 1 for s in SEEDS):
        raise ValueError("Road draws differ")
    atomic_write(ROOT / "completed.json", dict(runs=receipts, steps=len(receipts)*STEPS, torch=str(torch.__version__),
                 gpu=torch.cuda.get_device_name(), protocol=lock))
    return nets


def cases(rows):
    for r in rows:
        if r["role"] == "held":
            y = r["rgb"][None].cuda().float() / 255.
            for name, x in zip(("kitti_light", "kitti_dense"), road.rendered_samples(r)[0], strict=True):
                yield name, r["id"], r["drive"], x, y
    with RealPairs(DATASETS) as pairs:
        for name in DATASETS:
            for key in pairs.ids(name):
                images = pairs.read(name, key)
                x, y = (image_tensor(long_edge(im, min(max(im.size), 576)))[None].cuda().float() / 255. for im in images)
                yield name, key, key, x, y


@torch.inference_mode()
def evaluate(rows, author, bank, nets, lock):
    ledger, counts = [], Counter()
    replay = defaultdict(list)
    for name, key, group, fog, clear in cases(rows):
        g = road.geometry(fog, author)
        phase = [b % 4 for b in sha256(key.encode()).digest()[:2]]
        ref_lab = srgb_to_lab(clear[..., phase[0]::4, phase[1]::4])
        for seed in SEEDS:
            q = state(g, bank, seed)
            lum = rgb.luminance(q.anchor)[..., phase[0]::4, phase[1]::4].flatten()
            bins = torch.bucketize(lum, lum.new_tensor(rgb.BINS[1:-1]), right=True)
            old = adjustment(q, bank["utility_tone", seed], "local")
            for arm in ("anchor", "deployed", *ARMS):
                pred = q.anchor if arm == "anchor" else old if arm == "deployed" else adjustment(
                    geometry(q, arm), nets[arm, seed], "local")
                if arm == "luma":
                    replay[name].append(float((pred.clamp(0,1)-old.clamp(0,1)).abs().mean()))
                ledger.append(dict(ds=name, key=key, group=group, arm=arm, seed=seed,
                                   **rgb.stats(pred, clear, phase, bins, ref_lab)))
        counts[name] += 1
        if sum(counts.values()) % 40 == 0:
            print("scored", dict(counts), flush=True)
    if counts["kitti_dense"] != 36 or counts["kitti_light"] != 36 or not set(DATASETS) <= counts.keys():
        raise ValueError("Road benchmark incomplete")
    path = ROOT / "individual.json"
    atomic_write(path, ledger)
    atomic_write(ROOT / "test.json", dict(protocol=lock, ledger=digest(path), sets=rgb.summarize(ledger),
                 counts=dict(counts), maximum_per_frame_mean_replay_difference={k:max(v) for k,v in replay.items()}, utc=now()))
    print("road complete", dict(counts), flush=True)


if __name__ == "__main__":
    configure_cuda()
    lock = registration()
    author, _ = road.load_source()
    bank, rows = road.load_trained("rgb"), road.kitti()
    nets = fit(rows, author, bank, lock)
    evaluate(rows, author, bank, nets, lock)

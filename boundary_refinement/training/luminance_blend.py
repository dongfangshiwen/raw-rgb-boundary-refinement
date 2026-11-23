"""Fit a bounded brightness-conditioned mixture of two frozen, matched road RGB heads."""
from boundary_refinement.data.source_integrity import matches_registered
from collections import Counter, defaultdict
from hashlib import sha256

import torch

from boundary_refinement.models.restoration import bounded_tone as color
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.models.restoration import road_tone_control as matched
from boundary_refinement.training import road_heads as road
from boundary_refinement.models.restoration.complementary_rgb import ColorHead, adjustment
from boundary_refinement.metrics.color import srgb_to_lab
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import RealPairs, image_tensor, long_edge
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda
from boundary_refinement.evaluation.paired.veil_response import NH_RESERVED


ROOT = REPO / "results/calibrated-tone-v99"
GATES = ("constant", "curve", "roll_curve")
SEEDS = road.SEEDS


def plan(*, frozen=False):
    lock = ROOT / "protocol.json"
    cfg = dict(code=digest(__file__), learned_rgb=digest(matched.ROOT / "completed.json"), new_scenes=digest(held.SCENES),
               seeds=list(SEEDS), gates=list(GATES), knots=[0., .25, .5, .75, 1.], ridge_fraction=.002,
               fit="Two source means weighted equally: 38 light/dense views of 19 KITTI calibration images "
                   "from four recordings and 36 NH training images (IDs 01..45 except multiples of five). "
                   "All RGB heads frozen. Fit the exact bounded quadratic over paired RGB errors on CUDA. "
                   "Compare optimal constant vs five hat-function coefficients vs identical shifted luminance.",
               evaluation="Previously inspected V98 road/photographic cohorts are exploratory. Additional same-scene "
                          "factorial stress test on all 33 MUSES clear frames using neutral linear-light airlight .85 "
                          "and uniform transmission .70 and .40. Ground truth RGB and masks unchanged, no actual fog. "
                          "One deployed RGB seed and all three frozen segmentation seeds in the factorial. Compare "
                          "fixed semantic weights vs plain mean, ΔΔ (luma vs affine), and curve vs constant/rolled. "
                          "Use complete enabled cohorts regardless of outcomes; no further tuning after these scores.")
    if frozen:
        if not matches_registered(read(lock), cfg):
            raise ValueError("Frozen model protocol changed")
        return digest(lock)
    if lock.exists() and read(lock) != cfg:
        raise ValueError("Calibration protocol changed")
    if not lock.exists():
        atomic_write(lock, cfg)
    return digest(lock)


def heads():
    records = read(matched.ROOT / "completed.json")["runs"]
    bank = {}
    for r in records:
        path = matched.ROOT / f"{r['method']}-{r['seed']}.pth"
        if digest(path) != r["weights"]:
            raise ValueError("An RGB head changed")
        net = ColorHead().cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path, weights_only=True))
        bank[r["method"], r["seed"]] = net
    return bank


def endpoints(g, road_heads, controlled_heads, seed):
    state = matched.state(g, road_heads, seed)
    a, w = (adjustment(matched.geometry(state, arm), controlled_heads[arm, seed], "local").clamp(0., 1.)
            for arm in ("affine", "luma"))
    return state.anchor, a, w


def hats(lum, mode):
    if mode == "constant":
        return torch.ones_like(lum)
    if mode == "roll_curve":
        lum = lum.roll(tuple(n//2 for n in lum.shape[-2:]), dims=(-2, -1))
    levels = torch.arange(5, device=lum.device, dtype=lum.dtype)[None, :, None, None]
    return (1. - (4.*lum-levels).abs()).clamp_min(0.)


def mixture(anchor, affine, windowed, gate, coeff):
    value = (hats(color.luminance(anchor), gate) * coeff[None, :, None, None]).sum(1, keepdim=True)
    return affine + value * (windowed-affine)


def training_cases(kitti):
    for r in kitti:
        if r["role"] == "calibrate":
            target = r["rgb"][None].cuda().float() / 255.
            for fog in road.rendered_samples(r)[0]:
                yield "kitti_cal", r["id"], fog, target
    with RealPairs(("nh_train",)) as pairs:
        for key in pairs.ids("nh_train"):
            if key not in NH_RESERVED:
                sample = pairs.read("nh_train", key)
                x, y = (image_tensor(long_edge(im, min(576, max(im.size))))[None].cuda().float()/255. for im in sample)
                yield "nh_train", key, x, y


def normal_equations(anchor, affine, windowed, truth, gate):
    b = hats(color.luminance(anchor), gate)[0].flatten(1).double()
    d = (windowed-affine).double()
    norm = (truth-affine).double().square().mean().clamp_min(.0025)
    energy = d.square().mean(1).flatten()
    align = (d*(truth-affine)).mean(1).flatten()
    return (b*energy) @ b.T / (norm*b.shape[1]), b @ align / (norm*b.shape[1]), energy.mean()/norm


def solve(gram, rhs, energy):
    k = len(rhs)
    ridge = .002 * energy.clamp_min(1e-12) / k
    gram = gram + ridge*torch.eye(k, device=rhs.device, dtype=rhs.dtype)
    rhs = rhs + ridge*.5
    scale = torch.linalg.eigvalsh(gram).max().clamp_min(1e-18)
    x = torch.full_like(rhs, .5)
    for it in range(10000):
        y = (x-(gram@x-rhs)/scale).clamp(0.,1.)
        residual = (x-y).abs().max()
        x = y
        if it % 20 == 0 and residual.item() <= 1e-10:
            break
    if residual.item() > 1e-8:
        raise ValueError("Calibration QP did not converge")
    return x.float(), it+1, residual.item()


@torch.inference_mode()
def fit(kitti, author, bank, controlled, lock):
    counts, terms = Counter(), defaultdict(list)
    for ds, key, x, y in training_cases(kitti):
        g = road.geometry(x, author)
        for s in SEEDS:
            item = endpoints(g, bank, controlled, s)
            for mode in GATES:
                terms[ds, s, mode].append(normal_equations(*item, y, mode))
        counts[ds] += 1
    if counts != dict(kitti_cal=38, nh_train=36):
        raise ValueError("Unexpected calibration cohort")
    params, rows = {}, []
    for s in SEEDS:
        for m in GATES:
            sums = [[torch.stack(v).mean(0) for v in zip(*terms[ds, s, m])] for ds in counts]
            gram, rhs, energy = (torch.stack(v).mean(0) for v in zip(*sums))
            coeff, steps, residual = solve(gram, rhs, energy)
            params[m, s] = coeff
            rows.append(dict(mode=m, seed=s, coefficients=coeff.tolist(), iterations=steps,
                             scaled_projected_residual=residual))
    atomic_write(ROOT / "coefficients.json", dict(runs=rows, counts=counts, protocol=lock, device=torch.cuda.get_device_name(),
                 torch=str(torch.__version__)))
    return params


def load():
    info = read(ROOT / "coefficients.json")
    if info["protocol"] != plan(frozen=True):
        raise ValueError("Coefficient protocol mismatch")
    return {(r["mode"], r["seed"]): torch.tensor(r["coefficients"], device="cuda") for r in info["runs"]}


def summary(rows):
    from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
    from statistics import mean
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        by[r["ds"]][r["group"]][r["arm"]].append(r)
    result = {}
    for ds, groups in by.items():
        scores, effects = {}, {}
        for met in ("psnr", "deltaE", "ab_error"):
            bank = {group: {a:mean(r[met] for r in records) for a, records in g.items()} for group,g in groups.items()}
            arms = next(iter(bank.values()))
            scores[met] = {a:mean(v[a] for v in bank.values()) for a in arms}
            effects[met] = {a+"-"+b:bootstrap({g:v[a]-v[b] for g,v in sorted(bank.items())})
                            for a,b in (("curve","constant"), ("curve","roll_curve"), ("curve","affine"), ("curve","luma"),
                                        ("luma","affine"), ("constant","affine"))}
        result[ds] = dict(groups=len(groups), scores=scores, effects=effects)
    return result


@torch.inference_mode()
def evaluate(kitti, author, bank, controlled, params, lock):
    rows = []
    for ds, key, group, x, y in matched.cases(kitti):
        g = road.geometry(x, author)
        phase = [b % 4 for b in sha256(key.encode()).digest()[:2]]
        target = srgb_to_lab(y[..., phase[0]::4, phase[1]::4])
        for seed in SEEDS:
            state, a, w = endpoints(g, bank, controlled, seed)
            bins = color.luminance(state)[..., phase[0]::4, phase[1]::4].flatten()
            bins = torch.bucketize(bins, bins.new_tensor(color.BINS[1:-1]), right=True)
            images = dict(affine=a, luma=w, **{m:mixture(state,a,w,m,params[m,seed]) for m in GATES})
            for arm, image in images.items():
                rows.append(dict(ds=ds, key=key, group=group, arm=arm, seed=seed, **color.stats(image,y,phase,bins,target)))
    old = read(matched.ROOT / "test.json")["counts"]
    if len(rows) != sum(old.values())*len(SEEDS)*(2+len(GATES)):
        raise ValueError("Missing RGB comparisons")
    path = ROOT / "individual.json"
    atomic_write(path, rows)
    atomic_write(ROOT / "test.json", dict(sets=summary(rows), protocol=lock, ledger=digest(path), counts=old, utc=now()))


if __name__ == "__main__":
    configure_cuda()
    lock = plan()
    author, _ = road.load_source()
    bank, controlled, kitti = road.load_trained("rgb"), heads(), road.kitti()
    params = fit(kitti, author, bank, controlled, lock)
    evaluate(kitti, author, bank, controlled, params, lock)

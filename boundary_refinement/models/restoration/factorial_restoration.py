"""Test learned local utility and tonal correction on the same MCAF backbone."""
import argparse
from collections import Counter, defaultdict
from itertools import chain
from statistics import mean

import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.complementary_rgb import ColorGeometry, ColorHead, adjustment, objective as tone_loss, tonal_fields
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.training.decision_overlap import MANIFEST, exact, rr_records
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_author_mct import load_author_reference
from boundary_refinement.evaluation.photographic.evaluate_complementary_rgb import HELD, NEW, additional_samples, fresh_pairs
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_parallel_rgb import bind
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.exact_overlap import OFFSETS, matrix_blocks, neighbor_axis
from boundary_refinement.baselines.mcaf_attention import memory_efficient
from boundary_refinement.baselines.mcaf_reference import ROOT as MCAF, trained as mcaf_weights
from boundary_refinement.training.overlap_quadratic import Geometry, Partition, SEEDS, decoded, label
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.region_restore import view_size
from boundary_refinement.models.restoration.local_utility import MixtureHead, residual_actions
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.models.restoration.strong_anchor_color import EXPECTED, official, split
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/factorial-restoration-v65"
MIX = ("direct", "utility")
TONE = ("base", *MIX)
STEPS = 320
CONTRASTS = (("utility", "mcaf"), ("utility", "direct"), ("tone", "mcaf"),
             ("utility_tone", "utility"), ("utility_tone", "tone"), ("utility_tone", "direct_tone"))


def register():
    return dict(code_SHA256=digest(__file__), mcaf_SHA256=digest(MCAF / "completed.json"),
                data={p.name: digest(p) for p in (MANIFEST, HELD, NEW)}, cases=split(),
                seeds=list(SEEDS), steps_per_head=STEPS, mix=list(MIX), tone=list(TONE), prediction_strength=1.,
                design="Frozen MCAF seed 32003, its flip prediction, independent MCAF seed 32017, and author MCT. "
                       "The strong native prediction is the fixed center of both utility and same-capacity direct gates. "
                       "Fit each tonal head from the corresponding frozen output on the same 48 pairs. Use the prior full "
                       "amplitude without tuning. Report held RRSHID groups and a paired difference of differences. "
                       "For inputs larger than 256, each MCAF candidate transports only its low-resolution residual.")


class Restoration:
    def __init__(self):
        bind()
        bank = mcaf_weights()
        self.mcaf = [memory_efficient(bank[seed]) for seed in SEEDS[:2]]
        self.mct, self.release = load_author_reference()

    @staticmethod
    def mcaf_rgb(net, image):
        small = resize(image, view_size(image, 256, 8), antialias=True) if max(image.shape[-2:]) > 256 else image
        height, width = small.shape[-2:]
        aligned = F.pad(small, (0, (-width) % 4, 0, (-height) % 4), mode="replicate")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = net(aligned)[0][..., :height, :width].float().clamp(0., 1.)
        if small is not image:
            result = (image + resize(result - small, image.shape[-2:])).clamp(0., 1.)
        return result

    @torch.no_grad()
    def __call__(self, image):
        base = self.mcaf_rgb(self.mcaf[0], image)
        flipped = self.mcaf_rgb(self.mcaf[0], image.flip(-1)).flip(-1)
        alternate = self.mcaf_rgb(self.mcaf[1], image)
        mct = author_rgb(self.mct, image).clamp(0., 1.)
        deltas = torch.cat((flipped - base, alternate - base, mct - base))
        part = Partition(image, view_size(image, 12, 4))
        gram = part.average((deltas[:, None] * deltas[None, :]).mean(2)).permute(2, 3, 0, 1)[None]
        rms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
        correlation = (gram / (rms[..., :, None] * rms[..., None, :])).clamp(-1., 1.)
        edges = torch.stack([correlation[..., i, j] for i, j in ((0, 1), (0, 2), (1, 2))], dim=1)
        colors = part.average(torch.cat((2. * image - 1., 2. * base - 1., *(4. * d[None] for d in deltas)), dim=1))
        contrast = (part.average(image.square()) - part.average(image).square()).clamp_min(0.).sqrt()
        features = torch.cat((colors, 4. * rms.permute(0, 3, 1, 2), edges, 4. * contrast), dim=1)
        return Geometry(features, gram, rms, part, base, deltas)


def mix_weights(g, head, mode, smooth=False):
    center = g.features.new_zeros((1, 1, 1, 3))
    return residual_actions(head, g.features, g.gram, center,
                            "local_utility" if mode == "utility" else "residual_direct", smooth)


def mix_loss(net, sample, mode):
    g, blocks, rhs, baseline, mass = sample
    weights, signal = mix_weights(g, net, mode, smooth=True)
    error = (exact(blocks, weights, rhs) * mass).sum() / baseline
    if signal is None:
        return error
    estimate, offsets, rms = signal
    observed = (offsets @ rhs.unsqueeze(-1)).squeeze(-1) / rms
    fit = F.smooth_l1_loss(estimate, observed, beta=.06, reduction="none")
    return error + .30 * (fit.mean(-1) * mass).sum() / baseline.sqrt()


def tonal_context(g, base):
    part = g.partition
    fields = tonal_fields(base)
    rms = part.average(fields.square().mean(1)).permute(1, 2, 0)[None].clamp_min(1e-8).sqrt()
    features = torch.cat((g.features, 2. * part.average(base) - 1., 3. * rms.permute(0, 3, 1, 2)), dim=1)
    # The local tonal arm uses only the final four bases. No candidate projection is applied.
    zero_fields = torch.zeros_like(g.deltas)
    return ColorGeometry(base, torch.cat((zero_fields, fields)), part, features, None, rms, rms,
                         part.mass / part.mass.sum())


@torch.no_grad()
def tonal_sample(g, base, target):
    q = tonal_context(g, base)
    p = q.partition
    fields = q.fields[3:]
    moment = torch.einsum("achw,bchw->abhw", fields, fields) / fields.shape[1]
    r = {i: neighbor_axis(p.rows, i) for i in (-1, 0, 1)}
    c = {i: neighbor_axis(p.cols, i) for i in (-1, 0, 1)}
    terms = torch.stack([((r[i].T @ moment @ c[j]) / p.mass).permute(2, 3, 0, 1)[None] for i, j in OFFSETS])
    terms = F.pad(terms, (3, 0, 3, 0))
    residual = target - base
    z = p.average((fields * residual).mean(1)).permute(1, 2, 0)[None]
    return q.features, q.transform, q.raw_rms, q.projected_rms, q.mass, terms, F.pad(z, (3, 0)), residual.square().mean().clamp_min(.0025)


def mixture_sample(g, target):
    rhs, start = label(g, target)
    matrices = matrix_blocks(g)
    mass = g.partition.mass / g.partition.mass.sum()
    return g, torch.stack([matrices[step] for step in OFFSETS]), rhs, (start * mass).sum().clamp_min(.0025), mass


def load_or_fit(name, seed, stage, factory, pool, loss, mode, smoke):
    path = ROOT / f"{name}-{seed}.pth"
    status = path.with_suffix(".json")
    protocol = ROOT / "protocol.json"
    set_seed(seed)
    model = factory().cuda()
    if not smoke and status.exists():
        row = read(status)
        if digest(path) != row["weights_SHA256"] or digest(protocol) != row["protocol_SHA256"]:
            raise ValueError(f"Changed run: {name}/{seed}")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return model.eval().requires_grad_(False), row
    rng = torch.Generator().manual_seed(seed + 65001)
    opt = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.00007)
    for step in range(2 if smoke else STEPS):
        ids = torch.randint(len(pool), (3,), generator=rng).tolist()
        opt.zero_grad(set_to_none=True)
        value = torch.stack([loss(model, pool[i], mode) for i in ids]).mean()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5, error_if_nonfinite=True)
        opt.step()
        schedule.step()
    row = dict(arm=name, seed=seed, stage=stage, steps=step + 1, loss=float(value.detach()),
               parameters=sum(parameter.numel() for parameter in model.parameters()))
    if not smoke:
        torch_save(path, {key: value.detach().cpu() for key, value in model.state_dict().items()})
        row.update(weights_SHA256=digest(path), protocol_SHA256=digest(protocol))
        atomic_write(status, row)
    print(row, flush=True)
    return model.eval().requires_grad_(False), row


def fit(smoke):
    configure_cuda()
    restorer = Restoration()
    recipe = register()
    recipe["published_mct"] = restorer.release
    lock = ROOT / "protocol.json"
    if not smoke:
        if lock.exists() and read(lock) != recipe:
            raise ValueError("The recorded unified restoration protocol changed")
        atomic_write(lock, recipe)
    fit_ids = set(recipe["cases"]["fit"])
    originals = [(fog, clear) for _, key, _, fog, clear in rr_records("val") if key in fit_ids]
    if len(originals) != 48:
        raise ValueError("The MCAF fit set changed")
    pairs = [(restorer(as_float(fog)), as_float(clear)) for fog, clear in (originals[:4] if smoke else originals)]
    pool = [mixture_sample(g, truth) for g, truth in pairs]
    print(dict(fitting=len(pool)), flush=True)
    rows = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        heads = {}
        for mode in MIX:
            heads[mode], row = load_or_fit(mode, seed, "mixture", MixtureHead, pool, mix_loss, mode, smoke)
            rows.append(row)
        for mode in TONE:
            samples = [tonal_sample(g, g.base if mode == "base" else decoded(g, mix_weights(g, heads[mode], mode)[0]), truth)
                       for g, truth in pairs]
            _, row = load_or_fit(f"{mode}_tone", seed, "tone", ColorHead, samples, tone_loss, "local", smoke)
            rows.append(row)
    if not smoke:
        atomic_write(ROOT / "completed.json", dict(runs=rows, updates=sum(r["steps"] for r in rows),
                     protocol_SHA256=digest(lock), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), finished_utc=now()))


def trained():
    completed = read(ROOT / "completed.json")
    if digest(ROOT / "protocol.json") != completed["protocol_SHA256"]:
        raise ValueError("The factorial model protocol changed")
    bank = {}
    for row in completed["runs"]:
        path = ROOT / f'{row["arm"]}-{row["seed"]}.pth'
        if digest(path) != row["weights_SHA256"]:
            raise ValueError(f"Changed trained head {path.name}")
        model = (MixtureHead if row["stage"] == "mixture" else ColorHead)().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        bank[row["arm"], row["seed"]] = model
    if set(bank) != {(arm, seed) for arm in (*MIX, *(f"{mode}_tone" for mode in TONE)) for seed in SEEDS}:
        raise ValueError("Expected every factorial head and seed")
    return bank


def predictions(g, heads, seed):
    images = {"mcaf": g.base, "published_mct": g.base + g.deltas[2:3],
              "mcaf_mean": g.base + g.deltas[:2].sum(dim=0, keepdim=True) / 3.}
    for mode in MIX:
        images[mode] = decoded(g, mix_weights(g, heads[mode, seed], mode)[0])
    for mode in TONE:
        q = tonal_context(g, images["mcaf" if mode == "base" else mode])
        images["tone" if mode == "base" else f"{mode}_tone"] = adjustment(q, heads[f"{mode}_tone", seed], "local").clamp(0., 1.)
    return images


@torch.inference_mode()
def test():
    configure_cuda()
    restorer, bank = Restoration(), trained()
    note = dict(fit_SHA256=digest(ROOT / "completed.json"), code_SHA256=digest(__file__), cases=EXPECTED,
                comparisons=[list(x) for x in CONTRASTS], design="PSNR per photo and seed, paired bootstrap by close-image group. "
                "Interaction is the difference in tonal gain with and without the utility head. All cohorts were evaluated previously.")
    lock = ROOT / "evaluation_protocol.json"
    atomic_write(lock, note)
    seen, rows, panels = Counter(), [], []
    means, groups = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for d, key, group, fog, clear, _, condition in chain(official(), additional_samples(read(HELD)), fresh_pairs(read(NEW))):
        x, y = as_float(fog), as_float(clear)
        g = restorer(x)
        for seed in SEEDS:
            images = predictions(g, bank, seed)
            scores = {arm: area_psnr(value, y, None) for arm, value in images.items()}
            for arm, value in scores.items():
                means[d, arm].append(value)
                rows.append(dict(dataset=d, id=key, group=group, condition=condition, arm=arm, seed=seed, psnr=value))
            for a, b in CONTRASTS:
                groups[d, f"{a}-vs-{b}"][group].append(scores[a] - scores[b])
            groups[d, "interaction"][group].append(scores["utility_tone"] - scores["utility"] - scores["tone"] + scores["mcaf"])
            if seed == SEEDS[0] and (not seen[d] or (d == "rrshid" and seen[d] == 240)):
                names = ("mcaf", "utility", "tone", "utility_tone")
                panels.append((f"{d} / {key}", [to_rgb(p) for p in (x, *(images[a] for a in names), y)],
                               ("Fog", "Frozen MCAF", "Local utility", "Local tone", "Utility + tone", "Reference")))
        seen[d] += 1
        if seen[d] == EXPECTED[d]:
            print(d, {m: round(mean(v), 4) for (k, m), v in means.items() if k == d}, flush=True)
    if seen != EXPECTED:
        raise ValueError(f"Incomplete evaluation cohorts: {seen}")
    table, picture = ROOT / "individual.json", ROOT / "predictions.png"
    atomic_write(table, rows)
    draw(picture, panels)
    atomic_write(ROOT / "test.json", dict(counts=dict(seen), rows=len(rows), psnr={"/".join(k): mean(v) for k, v in means.items()},
                 contrasts={"/".join(k): bootstrap({g: mean(v) for g, v in data.items()}) for k, data in groups.items()},
                 table_SHA256=digest(table), pictures_SHA256=digest(picture), protocol_SHA256=digest(lock), finished_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "fit", "test"))
    stage = parser.parse_args().stage
    if stage == "test":
        test()
    else:
        fit(smoke=stage == "smoke")

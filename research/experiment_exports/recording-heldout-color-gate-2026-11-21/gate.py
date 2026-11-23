"""Choose between released labels and a fixed RGB correction at label disagreements."""
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path.insert(0, str(PROJECT / "colorawarenet-joint-model"))
from scripts import eval_range_v101 as ev
from scripts.fit_native_guide_v102 import author_loader, boundary_band

SCALES = (512, 1024)
MODES = {"global": [], "semantic": [1, 2], "aligned": [1, 2, 3], "shifted": [1, 2, 4]}
ARMS = ["official", "fixed", *MODES]


def read(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write(file, value):
    tmp = file.with_suffix(".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(file)


def save_array(file, **values):
    tmp = file.with_suffix(".part")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(file)


def digest(path):
    return ev.digest(path)


class Source:
    def __init__(self):
        author, self.model, self.signature = author_loader()
        self.model.eval().requires_grad_(False)
        self.mean, self.std = (torch.tensor(x, device="cuda")[None, :, None, None] for x in (author.MEAN, author.STD))

    def __call__(self, image, native_shape):
        raw = self.model.predict((image - self.mean) / self.std).float()
        z = ev.resize(raw, tuple(n // 4 for n in image.shape[-2:]))
        return z, ev.resize(raw, native_shape)


def source_rows():
    root = PROJECT / "benchmark_data/muses-v70"
    groups, hashes, data = {}, set(), {}
    for split, frames, records in (("fit", 96, 9), ("calibrate", 24, 4), ("held", 58, 5)):
        rows = read(root / f"{split}.json")["pairs"]
        for row in rows:
            h = row["frame_camera"]["SHA256"]
            if h in hashes:
                raise ValueError("An RGB occurs twice")
            hashes.add(h)
        groups[split] = {x["group"] for x in rows}
        assert (len(rows), len(groups[split])) == (frames, records)
        data[split] = sorted(rows, key=lambda r: (r["group"], r["id"]))
    assert len(set.union(*groups.values())) == 18
    return data


def open_muses(r):
    from PIL import Image
    output = []
    for key in ("frame_camera", "gt_semantic"):
        asset = r[key]
        assert digest(asset["file"]) == asset["SHA256"]
        with Image.open(asset["file"]) as img:
            output.append(np.array(img.convert("RGB") if key == "frame_camera" else img))
    rgb, lab = output
    assert lab.shape == rgb.shape[:2] and np.all((lab < 19) | (lab == 255))
    return rgb, torch.from_numpy(lab).cuda().long()


def image_tensor(rgb, side):
    h, w = rgb.shape[:2]
    shape = tuple(round(n*side/max(h, w)) for n in (h, w))
    assert all(x % 32 == 0 for x in shape)
    t = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float() / 255
    return ev.resize(t, shape, antialias=True).contiguous()


def donor_cues(z, guide, ids, old, new, shape):
    height, width = shape
    h, w = z.shape[-2:]
    row, col = ids // width, ids % width
    yy, xx = (row + .5) * (h/height) - .5, (col + .5) * (w/width) - .5
    y0, x0 = yy.floor().long(), xx.floor().long()
    a, b = yy - y0, xx - x0
    priors = torch.stack(((1-a)*(1-b), (1-a)*b, a*(1-b), a*b), 1)
    top, bottom = y0.clamp(0, h-1)*w, (y0+1).clamp(0, h-1)*w
    left, right = x0.clamp(0, w-1), (x0+1).clamp(0, w-1)
    donors = torch.stack((top+left, top+right, bottom+left, bottom+right), 1)
    classes = z.argmax(1).flatten()[donors]
    old_cells = (classes == old[:, None]) & (priors > 0)
    new_cells = (classes == new[:, None]) & (priors > 0)
    vote = (priors * (new_cells.float() - old_cells.float())).sum(1)
    both = old_cells.any(1) & new_cells.any(1)
    hy, wx = guide.shape[-2:]
    guide_id = (((row + .5) * (hy/height)).long().clamp_max(hy-1) * wx
                + ((col + .5) * (wx/width)).long().clamp_max(wx-1))
    def difference(image):
        pixel = image[0].permute(1, 2, 0).reshape(-1, 3)[guide_id]
        centers = F.avg_pool2d(image, 4, 4)[0].permute(1, 2, 0).reshape(-1, 3)[donors]
        distance = (pixel[:, None] - centers).square().mean(-1)
        d_old = distance.masked_fill(~old_cells, 2.).amin(1)
        d_new = distance.masked_fill(~new_cells, 2.).amin(1)
        return torch.where(both, (d_old-d_new)/(d_old+d_new+.02**2), 0)
    return vote, difference(guide), difference(torch.roll(guide, (hy//2, wx//2), (-2, -1)))


@torch.inference_mode()
def packet(model, kernel, rgb, y, edge, side, sample_key=None):
    image = image_tensor(rgb, side)
    z, official = model(image, y.shape)
    assert image.shape[-2:] == tuple(4*n for n in z.shape[-2:])
    guide = image.clamp(0, 1).mul(255).round().byte()
    delta = kernel.correction(z, ev.prepare(z, guide))
    proposal = official + ev.resize(delta, y.shape)
    old_map = official.argmax(1)[0].flatten()
    new_map = proposal.argmax(1)[0].flatten()
    ids = (old_map != new_map).nonzero().flatten()
    # Model cues use RGB and both predictions only. Labels restrict fitting and evaluation.
    label = y.flatten()
    if sample_key is not None:
        informative = (old_map[ids] == label[ids]) | (new_map[ids] == label[ids])
        ids = ids[informative]
        gen = torch.Generator(device="cuda").manual_seed(int.from_bytes(sha256(sample_key.encode()).digest()[:4], "big"))
        if len(ids) > 1536:
            ids = ids[torch.randperm(len(ids), generator=gen, device="cuda")[:1536]]
    old, new = old_map[ids], new_map[ids]
    n = old_map.numel()
    def margin(a, first, second):
        s = a[0].reshape(19, n)
        return (s[first, ids] - s[second, ids]).clamp_min(0)
    original_margin, proposal_margin = margin(official, old, new), margin(proposal, new, old)
    total = (original_margin + proposal_margin).clamp_min(1e-20)
    t = (original_margin/total).clamp(1e-5, 1-1e-5)
    vote, aligned, shifted = donor_cues(z, guide.float().div_(255), ids, old, new, y.shape)
    features = torch.stack((torch.logit(t), torch.full_like(t, -1 if side == 512 else 1),
                            vote, aligned, shifted), 1)
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite source or target cues")
    if sample_key is not None:
        return features.cpu().numpy(), (new == label[ids]).byte().cpu().numpy()
    valid = label[ids] < 19
    packed = [a[valid].cpu().numpy() for a in (label[ids], old, new, edge.flatten()[ids])]
    baseline = confusion(label.cpu().numpy(), old_map.cpu().numpy(), edge.flatten().cpu().numpy())
    return baseline, packed, features[valid]


def confusion(y, p, boundary):
    valid = y < 19
    counts = np.bincount((boundary[valid].astype(np.int64)*19 + y[valid])*19 + p[valid],
                         minlength=2*19*19).reshape(2, 19, 19)
    return np.stack((counts.sum(0), counts[1])).astype(np.int32)


def measure(baseline, packed, features, fits):
    y, old, new, edge = packed
    accepted = {"official": np.zeros(len(y), bool), "fixed": np.ones(len(y), bool)}
    for mode, cols in MODES.items():
        w = features.new_tensor(fits[mode]["weights"], dtype=torch.float64)
        score = w[0].expand(len(features))
        if cols:
            score = score + features[:, cols].double() @ w[1:]
        accepted[mode] = (score > features[:, 0].double()).cpu().numpy()
    cms, effects = [], []
    for mode in ARMS:
        take = accepted[mode]
        cm = baseline + confusion(y[take], new[take], edge[take]) - confusion(y[take], old[take], edge[take])
        masks = (take & (new == y), take & (old == y), take & (old != y) & (new != y))
        effects.append([[int(x.sum()) for x in masks], [int((x & edge).sum()) for x in masks]])
        assert np.all(cm >= 0) and np.all(cm[1] <= cm[0]) and np.array_equal(cm.sum(-1), baseline.sum(-1))
        assert np.array_equal(cm.diagonal(axis1=-1, axis2=-2).sum(-1) - baseline.diagonal(axis1=-1, axis2=-2).sum(-1),
                              np.array(effects[-1])[:, 0] - np.array(effects[-1])[:, 1])
        cms.append(cm)
    return np.stack(cms, axis=1), np.stack(effects, axis=1)

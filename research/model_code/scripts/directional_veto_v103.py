"""A source-trained decision to accept a frozen SegFix displacement at a class conflict."""
import torch
import torch.nn.functional as F
from torch import nn
from scripts import subpixel_transport_v92 as original
from scripts.cross_scale_rgb_v44 import resize

MODES = ("margin", "aligned", "shifted")
SEEDS = (101, 211, 307)


def paper_stencil(z, small):
    shape = tuple(4 * n for n in z.shape[-2:])
    guide = small if shape == small.shape[-2:] else resize(small, shape, antialias=True)
    guide = guide.clamp(0, 1).mul(255).round().byte()
    return original.guide_stencil(guide[[0, 0]], z.argmax(1)[0])


def paper_scores(z, stencil, weights, base):
    correction = original.prediction("dual_free", weights, z, stencil)
    return base + resize(correction, base.shape[-2:])


class DirectionalColors:
    def __init__(self, raw):
        smooth = F.avg_pool2d(F.pad(raw, (2, 2, 2, 2), mode="replicate"), 5, 1)
        self.h, self.w = raw.shape[-2:]
        self.colors = [v[0].permute(1, 2, 0).reshape(-1, 3) for v in (raw, smooth)]

    def pair(self, ids, offset, shift):
        y, x = ids // self.w, ids % self.w
        dy, dx = offset.reshape(-1, 2)[ids].unbind(-1)
        def index(k):
            # Spatial roll is applied to the guide, after clamping the geometric probes.
            yy = ((y + k * dy).clamp(0, self.h - 1) - shift[0]) % self.h
            xx = ((x + k * dx).clamp(0, self.w - 1) - shift[1]) % self.w
            return yy * self.w + xx
        near, opposite, center = index(2), index(-2), index(0)
        out = []
        for color in self.colors:
            a = (color[center] - color[near]).square().mean(-1)
            b = (color[center] - color[opposite]).square().mean(-1)
            out.append((b - a) / (a + b + .03**2))
        return out


def conflicts(scores, lookup, offset, colors):
    labels = scores.argmax(1)[0].byte()
    own = labels.flatten()
    moved = own[lookup].reshape_as(labels)
    ids = (own != moved.flatten()).nonzero().flatten()
    if not ids.numel():
        return labels, moved, ids, scores.new_empty(0, 6)
    other = lookup[ids]
    bank = scores[0].reshape(scores.shape[1], -1)
    def margin(points, rivals):
        values = bank[:, points].T
        best4 = values.topk(4, dim=-1).values
        gap = best4[:, 0] - values.gather(1, own[rivals, None].long()).flatten()
        return (gap / (best4[:, 0] - best4[:, 3]).clamp_min(1e-6)).clamp(0, 4)
    features = torch.stack((margin(ids, other), margin(other, ids),
                            *colors.pair(ids, offset, (0, 0)), *colors.pair(ids, offset, (17, 37))), -1)
    return labels, moved, ids, features


class MoveDecision(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.linear = nn.Linear(2 if mode == "margin" else 4, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features):
        if self.mode == "shifted":
            features = features[:, (0, 1, 4, 5)]
        elif self.mode == "aligned":
            features = features[:, :4]
        else:
            features = features[:, :2]
        return self.linear(features).flatten()

    def decide(self, labels, moved, ids, features):
        result = labels.clone()
        selected = ids[self(features) >= 0]
        result.view(-1)[selected] = moved.flatten()[selected]
        return result

"""Standalone range/spatial factorization for upsampling semantic scores. PyTorch only."""
import math

import torch
from torch import nn
import torch.nn.functional as F


FAMILIES = tuple(f"{space}_{color}" for space in ("gaussian", "bilinear")
                 for color in ("gaussian", "bounded1", "bounded4"))
RANGES = (.02, .05, .10, .20)


def prepare(logits, image, gated=True):
    """image: [1,3,4H,4W] uint8 (or float [0,1]); logits: [1,C,H,W]."""
    _, channels, h, w = logits.shape
    if image.shape != (1, 3, 4*h, 4*w):
        raise ValueError("Guide must match the 4x decoder lattice")
    centers = (torch.arange(4, device=logits.device) + .5) / 4 - .5
    offsets = torch.cartesian_prod(*(torch.arange(-1, 2, device=logits.device),) * 2)
    displacement = torch.cartesian_prod(centers, centers)[:, None] - offsets
    base = (1 - displacement.abs()).clamp_min(0).prod(-1)
    dist = displacement.square().sum(-1)
    idx = torch.arange(h*w, device=logits.device)
    neigh = ((idx[:, None] // w + offsets[:, 0]).clamp(0, h-1) * w +
             (idx[:, None] % w + offsets[:, 1]).clamp(0, w-1))
    if gated:
        labels = logits.argmax(1).flatten()
        idx = idx[(labels[neigh] != labels[:, None]).any(-1)]
    neigh = neigh[idx]
    raw = image[0].float()
    if image.dtype == torch.uint8:
        raw = raw / 255.
    pix = raw.reshape(3,h,4,w,4).permute(1,3,2,4,0).reshape(h*w,16,3)
    d = (pix[idx,:,None] - pix.mean(1)[neigh][:,None]).square().mean(-1)
    bands = logits.new_tensor(RANGES)
    features = 1 - (-.5 * d[...,None]/bands.square()).exp()
    donor = logits[0].permute(1,2,0).reshape(h*w,channels)[neigh]
    return idx, base, dist, d, features, donor


class RangeDecoder(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.space, self.color = family.split("_")
        self.spatial = nn.Parameter(torch.zeros(()))
        if self.color == "bounded4":
            self.mix = nn.Parameter(torch.full((4,), -.5))
        else:
            initial_band = .05 if self.color == "gaussian" else math.sqrt(4/sum(1/r**2 for r in RANGES))
            # Center the bounded positive bandwidth on its protocol-defined initialization.
            fraction = (initial_band-.005)/(.5-.005)
            self.width = nn.Parameter(torch.tensor(math.log(fraction/(1-fraction))))
            if self.color == "bounded1":
                self.mix = nn.Parameter(torch.tensor(-.5))

    def plan(self, st):
        _, base, dist, d, features, _ = st
        exponent = .25 + 1.5*self.spatial.sigmoid()
        if self.space == "gaussian":
            prior = -.5*dist/.65**2
        else:
            prior = base.clamp_min(1e-12).log()
        if self.color == "bounded4":
            cost = features @ (6*self.mix.sigmoid())
        else:
            bandwidth = .005 + .495*self.width.sigmoid()
            cost = .5*d/bandwidth.square()
            if self.color == "bounded1":
                cost = (24*self.mix.sigmoid())*(-torch.expm1(-cost))
        log_weights = prior*exponent - cost
        if self.space == "bilinear":
            log_weights = log_weights.masked_fill(base == 0, -torch.inf)
        return log_weights.softmax(-1)

    def correction(self, logits, st):
        _, channels, h, w = logits.shape
        idx, base, _, _, _, donor = st
        values = (self.plan(st)-base) @ donor
        grid = logits.new_zeros((h*w,16,channels)).index_copy(0,idx,values)
        return grid.reshape(h,w,4,4,channels).permute(4,0,2,1,3).reshape(1,channels,h*4,w*4)

    def forward(self, logits, guide, size=None):
        size = size or guide.shape[-2:]
        delta = self.correction(logits, prepare(logits, guide))
        return F.interpolate(logits,size=size,mode="bilinear",align_corners=False) + F.interpolate(
            delta,size=size,mode="bilinear",align_corners=False)

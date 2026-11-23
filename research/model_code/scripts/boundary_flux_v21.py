"""Multiscale RGB refinement with signed, class-aware local detail transfers."""
import torch
from torch import nn
import torch.nn.functional as F


METHODS = ("plain", "rgb", "semantic", "shifted", "oracle")


def edges(value):
    return value[..., 1:] - value[..., :-1], value[..., 1:, :] - value[..., :-1, :]


def divergence(horizontal, vertical):
    """Add each edge value to one endpoint and subtract it from the other."""
    x = F.pad(horizontal, (0, 1)) - F.pad(horizontal, (1, 0))
    y = F.pad(vertical, (0, 0, 0, 1)) - F.pad(vertical, (0, 0, 1, 0))
    return x + y


def raw_affinities(image):
    return tuple((-4 * delta.abs().mean(dim=1, keepdim=True)).exp() for delta in edges(image))


def predicted_affinities(logits):
    probability = logits.float().softmax(dim=1)
    return tuple((1 - 0.5 * delta.abs().sum(dim=1, keepdim=True)).clamp(0, 1).square()
                 for delta in edges(probability))


def label_affinities(labels):
    """Label 255 forms its own component; use no reference RGB at this interface."""
    if labels is None or labels.ndim != 3:
        raise ValueError("The label-guided diagnostic requires an indexed mask")
    return ((labels[..., 1:] == labels[..., :-1])[:, None].float(),
            (labels[..., 1:, :] == labels[..., :-1, :])[:, None].float())


def guidance(image, logits, labels=None):
    """All variants share RGB weights. Only the source of the edge restriction varies."""
    base = raw_affinities(image)
    semantic = predicted_affinities(logits)
    shifted = tuple(value.flip((-2, -1)) for value in semantic)
    result = dict(plain=base, rgb=base,
                  semantic=tuple(a * b for a, b in zip(base, semantic)),
                  shifted=tuple(a * b for a, b in zip(base, shifted)))
    if labels is not None:
        result["oracle"] = tuple(a * b for a, b in zip(base, label_affinities(labels)))
    return result


def coarse_edges(weights):
    """Average edges crossing each pair of neighboring 2x2 feature cells."""
    horizontal, vertical = weights
    return F.avg_pool2d(horizontal, (2, 1))[..., 1::2], F.avg_pool2d(vertical, (1, 2))[..., 1::2, :]


def transport(feature, weights):
    delta_x, delta_y = edges(feature.float())
    return 0.25 * divergence(weights[0] * delta_x, weights[1] * delta_y)


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1), nn.SiLU())


def upsample(value, shape):
    return F.interpolate(value, shape, mode="bilinear", align_corners=False)


class BoundaryRefiner(nn.Module):
    """Unrestricted coarse RGB changes with a separate local detail proposal."""
    def __init__(self, seed, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown boundary control {method}")
        self.method = method
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 21_100_000)
            self.stem = nn.Sequential(conv(9, 32, 2), conv(32, 32))
            self.mix = conv(64, 32)
            self.middle = nn.Sequential(conv(32, 48, 2), conv(48, 48))
            self.bottom = nn.Sequential(conv(48, 64, 2), conv(64, 64))
            self.middle_decoder = nn.Sequential(conv(64 + 48, 48), conv(48, 48))
            self.fine_decoder = nn.Sequential(conv(48 + 32, 36), conv(36, 36))
            self.coarse_output = nn.Conv2d(48, 3, 3, padding=1)
            self.fine_output = nn.Conv2d(36, 24, 3, padding=1)
        for module in (self.coarse_output, self.fine_output):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image, anchor, weights):
        if any(size % 8 for size in image.shape[-2:]):
            raise ValueError("Refiner expects image dimensions divisible by eight")
        half = self.stem(torch.cat((image, anchor, image - anchor), dim=1))
        half = self.mix(torch.cat((half, transport(half, coarse_edges(weights))), dim=1))
        middle = self.middle(half)
        bottom = self.bottom(middle)
        context = self.middle_decoder(torch.cat((middle, upsample(bottom, middle.shape[-2:])), dim=1))
        coarse = 0.35 * upsample(self.coarse_output(context).float().tanh(), image.shape[-2:])
        finest = self.fine_decoder(torch.cat((half, upsample(context, half.shape[-2:])), dim=1))
        values = F.pixel_shuffle(self.fine_output(finest), 2).float().tanh()
        pos_x, pos_y = values.chunk(2, dim=1)
        if self.method == "plain":
            detail = 0.16 * (pos_x + pos_y)
        else:
            detail = 0.16 * divergence(weights[0] * pos_x[..., :-1], weights[1] * pos_y[..., :-1, :])
        return anchor + coarse + detail, coarse, detail


@torch.no_grad()
def probe_math():
    generator = torch.Generator().manual_seed(610)
    raw = torch.rand((2, 3, 32, 48), generator=generator)
    logits = torch.rand((2, 11, 32, 48), generator=generator)
    labels = torch.zeros(2, 32, 48, dtype=torch.long)
    labels[..., :19] = 1
    labels[..., 45:] = 255
    weights = guidance(raw, logits, labels)
    for mode in METHODS:
        model = BoundaryRefiner(42, mode)
        initial, _, _ = model(raw, raw, weights[mode])
        if not torch.equal(initial, raw):
            raise ValueError("Refiner initialization is not an exact identity")
    horizontal = torch.rand((2, 3, 32, 47), generator=generator) * weights["oracle"][0]
    vertical = torch.rand((2, 3, 31, 48), generator=generator) * weights["oracle"][1]
    detail = divergence(horizontal, vertical).double()
    residuals = {value: float((detail * (labels == value)[:, None]).sum((-2, -1)).abs().max())
                 for value in (0, 1, 255)}
    if max(residuals.values()) > 1e-5 or torch.count_nonzero(weights["oracle"][0][..., 18]):
        raise ValueError("Oracle detail flows across a labeled boundary")
    return residuals

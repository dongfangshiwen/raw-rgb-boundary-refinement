"""Prospective image-space affine/complement split; not a novelty assertion.

Least-squares orthogonality is classical, and photometric/structural error
decomposition is explicitly prior art (PAL, arXiv:2604.08172). This candidate
changes the RGB parameterization at inference: a raw-image global branch owns
the shared gain / channel offsets, and only the orthogonal local residual uses
semantic context. It retains full reference RGB supervision. A PAL loss port
on the matched additive architecture is an actual planned prior-method control.
No classifier-null-space surrogate backward is used here.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from scripts.decision_joint_v15 import DecisionJoint


@dataclass(frozen=True)
class Spec:
    orthogonal: bool = False
    local_context: bool = True
    pal: bool = False
    classifier_supervision_only: bool = False


SPECS = {
    'additive': Spec(),
    'orthogonal': Spec(orthogonal=True),
    'orthogonal_noguide': Spec(orthogonal=True, local_context=False),
    'pal_additive': Spec(pal=True),
    'orthogonal_rgb_context': Spec(orthogonal=True, classifier_supervision_only=True),
}


def affine_components(image, value):
    """P_I value and Q_I value for span{shared I gain, RGB constants}.

    Channel offsets are independent; the gain is common across RGB. This is
    an algebraic image-space split, NOT identification of physical airlight,
    transmission or scene reflectance. Degenerate constant images use a=0
    (minimum-norm centered gain); no division by zero. Domain is the complete
    tensor supplied to the model, before output clipping / external cropping.
    """
    if image.shape != value.shape or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError('Equal B,3,H,W image/value tensors required')
    with torch.autocast(image.device.type, enabled=False):
        # Preserve float64 in analytic tests; training/inference use FP32.
        dtype = torch.float64 if image.dtype == value.dtype == torch.float64 else torch.float32
        x = image.to(dtype); y = value.to(dtype)
        xm = x.mean((-2, -1), keepdim=True); ym = y.mean((-2, -1), keepdim=True)
        xc = x - xm; yc = y - ym
        den = xc.square().sum((1, 2, 3), keepdim=True)
        numer = (xc * yc).sum((1, 2, 3), keepdim=True)
        # A per-pixel variance floor also handles FP32 reduction roundoff for
        # large constant images. Above this numerical-rank floor the shared
        # gain is unique; below it only channel means are identified.
        floor = image[0].numel() * 1e-12
        gain = torch.where(den > floor, numer / den.clamp_min(floor), torch.zeros_like(den))
        offset = ym - gain * xm
        fitted = gain * x + offset
        return fitted, y - fitted, gain, offset


class GlobalAffine(nn.Module):
    """Raw-RGB-only global gain/offset head; identity initialization."""
    def __init__(self):
        super().__init__()
        layers = []; channels = (3, 16, 32, 64, 64)
        for a, b in zip(channels, channels[1:]):
            layers.extend((nn.Conv2d(a, b, 3, stride=2, padding=1), nn.SiLU()))
        self.features = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Linear(70, 64), nn.SiLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def forward(self, image):
        z = self.features(image).mean((-2, -1)).float()
        raw = image.float(); mean = raw.mean((-2, -1)); std = raw.var((-2, -1), unbiased=False).clamp_min(0).sqrt()
        coefficients = self.output(torch.cat((z, mean, std), 1)).float()
        gain = 1 + coefficients[:, :1, None, None]
        offset = coefficients[:, 1:, None, None]
        return gain * raw + offset, gain, offset


class AffineSplitJoint(DecisionJoint):
    def __init__(self, seed=42, method='orthogonal', classes=11, pretrained=True):
        design = SPECS[method]
        super().__init__(seed, 'shared' if design.local_context else 'independent', classes, pretrained)
        self.method = method; self.design = design
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 16_100_000)
            self.global_affine = GlobalAffine()
        # All five candidates start at exact identity RGB, with identical NAF
        # parameters. This is a disclosed change to author initialization.
        nn.init.zeros_(self.rgb_net.ending.weight); nn.init.zeros_(self.rgb_net.ending.bias)

    def local_residual(self, image, context):
        """Author NAF bodies, four optional semantic additions, no RGB clipping."""
        net = self.rgb_net; h, w = image.shape[-2:]
        inp = net.check_image_size(image); x = net.intro(inp); skips = []
        for index, (encoder, down) in enumerate(zip(net.encoders, net.downs)):
            if self.design.local_context:
                value = F.interpolate(self.photo_context[index](context), x.shape[-2:], mode='bilinear', align_corners=False)
                x = x + value
            x = encoder(x); skips.append(x); x = down(x)
        x = net.middle_blks(x)
        for decoder, up, skip in zip(net.decoders, net.ups, skips[::-1]):
            x = decoder(up(x) + skip)
        return net.ending(x)[:, :, :h, :w].float()

    def forward(self, image):
        fused, low_logits = self.semantic(image)
        if self.design.classifier_supervision_only:
            # Capacity/initialization matched mechanistic control. RGB trains
            # the context encoder/fused head; labels train ONLY the classifier.
            # It is not a competitive fully supervised segmentation baseline.
            with torch.autocast(image.device.type, enabled=False):
                low_logits = self.seg_head.classifier(fused.detach())
        raw_local = self.local_residual(image, fused)
        global_rgb, gain, offset = self.global_affine(image)
        if self.design.orthogonal:
            fitted, local, _, _ = affine_components(image, raw_local)
        else:
            fitted = torch.zeros_like(raw_local); local = raw_local
        unbounded = global_rgb.float() + local.float()
        return unbounded.clamp(0, 1), F.interpolate(low_logits, image.shape[-2:], mode='bilinear', align_corners=False), dict(
            feature=fused, low_logits=low_logits, unbounded=unbounded, global_rgb=global_rgb,
            raw_local=raw_local, local=local, removed_affine=fitted, gain=gain, offset=offset)


def pal_penalty(prediction, reference, alpha=.8, eps=1e-3):
    """PAL eq12 / appendix D port, added to the common full RGB loss.

    Primary: https://arxiv.org/html/2604.08172v1 . Full 3x3+offset ridge fit,
    stop-gradient coefficients, unnormalised pixel sums and eps*I4 including
    the offset as in appendix D. FP64 statistics/solve stabilize the same
    algebra; aligned prediction is FP32 with gradients. No oracle at inference.
    alpha=.8 is the paper's restoration setting, not tuned on our validation.
    """
    with torch.autocast(prediction.device.type, enabled=False):
        p = prediction.float(); target = reference.float(); batch = len(p)
        with torch.no_grad():
            pixels = p.flatten(2).transpose(1, 2).double()
            truth = target.flatten(2).transpose(1, 2).double()
            design = torch.cat((pixels, torch.ones_like(pixels[:, :, :1])), -1)
            gram = design.transpose(1, 2) @ design
            rhs = design.transpose(1, 2) @ truth
            matrix = torch.linalg.solve(gram + eps * torch.eye(4, device=p.device, dtype=torch.float64)[None], rhs)
            matrix = matrix.transpose(1, 2).float()
        features = torch.cat((p.flatten(2), p.new_ones(batch, 1, p.shape[-2] * p.shape[-1])), 1)
        aligned = (matrix @ features).view_as(p)
        return alpha * (aligned - target).abs().mean()


def build_model(seed, method, classes=11, pretrained=True):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA PyTorch required')
    model = AffineSplitJoint(seed, method, classes, pretrained).cuda()
    torch.cuda.manual_seed_all(seed + 16_020_000)
    return model


def load_model(path, device='cuda'):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['format'] != 'affine-split-v16': raise ValueError('Wrong checkpoint format')
    model = AffineSplitJoint(value['seed'], value['method'], value['classes'], False)
    model.load_state_dict(value['model_state'], strict=True)
    return model.to(device).eval(), value

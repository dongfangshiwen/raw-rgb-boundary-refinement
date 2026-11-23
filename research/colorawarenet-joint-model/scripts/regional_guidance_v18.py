"""Two matched local RGB corrections and a budgeted selector for semantic guidance."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from scripts.affine_split_v16 import GlobalAffine
from scripts.decision_joint_v15 import DecisionJoint
from scripts.dual_coordinate_v17 import canonical_rgb


TILE = 16
CLASSES = 11
FEATURE_STRIDE = 4
SELECTORS = ("direct", "gain", "paired_gain")
CONTROLS = ("anchor", "rgb", "semantic", "entropy", *SELECTORS)
GAIN_SCALE = 100.0


def local_correction():
    """107 channels represent the same image evidence plus a matched context."""
    module = nn.Sequential(
        nn.Conv2d(96 + CLASSES, 40, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(40, 40, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(40, 3, 3, padding=1),
    )
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)
    return module


def descriptor_rgb(image, anchor):
    stride = FEATURE_STRIDE
    mean = F.avg_pool2d(image, stride)
    spread = (F.avg_pool2d(image.square(), stride) - mean.square()).clamp_min(0).sqrt()
    variations = local_differences(image)
    correction = F.avg_pool2d(anchor - image, stride)
    return torch.cat((mean, spread, F.avg_pool2d(variations, stride), correction), dim=1)


def local_differences(value):
    dx = F.pad((value[..., 1:] - value[..., :-1]).abs().mean(dim=1, keepdim=True), (0, 1, 0, 0))
    dy = F.pad((value[..., 1:, :] - value[..., :-1, :]).abs().mean(dim=1, keepdim=True), (0, 0, 0, 1))
    return torch.cat((dx, dy), dim=1)


def patch_mean(value):
    return F.avg_pool2d(value, TILE)


def patch_window(image):
    """Fade to the RGB result at tile interfaces, keeping decisions disjoint."""
    axis = torch.arange(TILE, device=image.device, dtype=image.dtype)
    ramp = torch.minimum(axis, TILE - 1 - axis).clamp(max=2) / 2
    tile = ramp[:, None] * ramp[None]
    return tile.repeat(image.shape[-2] // TILE, image.shape[-1] // TILE)


class RegionalGuidance(DecisionJoint):
    def __init__(self, seed, pretrained=True):
        super().__init__(seed, "independent", CLASSES, pretrained)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 18_110_000)
            self.global_affine = GlobalAffine()
            self.rgb_correction = local_correction()
            self.semantic_correction = local_correction()
        self.semantic_correction.load_state_dict(self.rgb_correction.state_dict())
        nn.init.zeros_(self.rgb_net.ending.weight)
        nn.init.zeros_(self.rgb_net.ending.bias)

    def forward(self, image):
        height, width = image.shape[-2:]
        if height % TILE or width % TILE or min(height, width) < 32:
            raise ValueError("Input must be padded to the tile grid and at least 32 pixels wide and high")
        semantic_input = canonical_rgb(image, self.mean, "channel")
        _, quarter_logits = self.semantic(semantic_input)
        logits = F.interpolate(quarter_logits, (height, width), mode="bilinear", align_corners=False)
        probabilities = quarter_logits.detach().softmax(dim=1)

        global_rgb, _, _ = self.global_affine(image)
        residual = self.rgb_net(image).float() - image
        anchor = (global_rgb.float() + residual).clamp(0, 1)
        base = anchor.detach()
        common = F.pixel_unshuffle(torch.cat((image, base), dim=1), FEATURE_STRIDE)
        rgb_context = descriptor_rgb(image, base)
        rgb = self.refine(common, rgb_context, base, self.rgb_correction)
        semantic = self.refine(common, probabilities, base, self.semantic_correction)
        # The semantic option begins from the same visible RGB value at tile boundaries.
        guided = rgb + patch_window(rgb) * (semantic - rgb)
        return anchor, rgb, semantic, guided, logits, probabilities

    @staticmethod
    def refine(common, context, anchor, head):
        residual = head(torch.cat((common, context), dim=1)).float()
        residual = F.interpolate(residual, anchor.shape[-2:], mode="bilinear", align_corners=False)
        return (anchor + 0.20 * residual.tanh()).clamp(0, 1)


def selector_features(image, anchor, rgb, guided, probabilities):
    """Inputs available from a single hazy image. No clean reference or fog map."""
    image_delta = patch_mean(local_differences(image))
    candidate_delta = patch_mean((guided - rgb).abs())
    rgb_delta = patch_mean((rgb - image).abs())
    input_sd = (patch_mean(image.square()) - patch_mean(image).square()).clamp_min(0).sqrt()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    confidence = 1 - entropy / math.log(CLASSES)
    feature_pool = TILE // FEATURE_STRIDE
    tile_confidence = F.avg_pool2d(confidence, feature_pool)
    boundaries = F.avg_pool2d(local_differences(probabilities), feature_pool)
    class_means = F.avg_pool2d(probabilities, feature_pool)
    features = torch.cat((
        patch_mean(image), input_sd, image_delta, patch_mean(anchor), rgb_delta, candidate_delta,
        tile_confidence, boundaries, class_means,
    ), dim=1)
    return features, tile_confidence


class TileSelector(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 18_220_000)
            self.score = nn.Sequential(
                nn.Conv2d(31, 32, 3, padding=1),
                nn.GroupNorm(4, 32),
                nn.SiLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 1, 1),
            )

    def forward(self, features):
        return self.score(features)


def choose_tiles(scores, fraction=0.5, straight_through=False):
    """Use the same exact tile count for every ranked comparison, including ties."""
    flat = scores.flatten(1)
    count = max(1, round(flat.shape[1] * fraction))
    indices = flat.argsort(dim=1, descending=True, stable=True)[:, :count]
    hard = torch.zeros_like(flat).scatter(1, indices, 1)
    if straight_through:
        cutoff = flat.gather(1, indices[:, -1:]).detach()
        soft = ((flat - cutoff) / 0.25).sigmoid()
        return (hard + (soft - soft.detach())).reshape_as(scores)
    return hard.reshape_as(scores)


def apply_choice(rgb, guided, tiles):
    mask = tiles.repeat_interleave(TILE, dim=-2).repeat_interleave(TILE, dim=-1)
    return rgb + mask * (guided - rgb)


def tile_gain(rgb, guided, reference):
    """Measured effect of replacing only the RGB correction on one tile."""
    error_before = (rgb - reference).square().mean(dim=1, keepdim=True)
    error_after = (guided - reference).square().mean(dim=1, keepdim=True)
    return patch_mean(error_before - error_after)


def selector_loss(name, scores, gain, paired, coverage):
    if name == "direct":
        weight = choose_tiles(scores, coverage, straight_through=True)
        return -(weight * (GAIN_SCALE * gain)).mean()
    if name == "gain":
        return F.smooth_l1_loss(scores, GAIN_SCALE * gain, beta=0.25)
    if name == "paired_gain":
        return F.smooth_l1_loss(scores, GAIN_SCALE * paired, beta=0.25)
    raise ValueError(f"Unknown selector {name}")

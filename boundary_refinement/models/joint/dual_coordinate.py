"""A fixed photometric quotient for semantics, with RGB still seeing raw input.

Mean/variance normalization and fog-invariant features are prior art. The
question here is whether a shared-gain normalization is useful at the interface
between the semantic encoder and a restoration branch under a matched recipe.
"""
import torch

from boundary_refinement.models.joint.affine_split import AffineSplitJoint


METHODS = ('raw', 'channel', 'shared_rms')


def canonical_rgb(image, center, mode):
    """Remove channel offsets and one positive gain (or three in `channel`).

    Preserve relative channel variances in shared mode. On constant or near
    constant inputs, use a variance floor; this region, clipping, spatially
    varying haze, and finite quantization are outside the invariance guarantee.
    """
    if mode not in METHODS or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError('Need a declared mode and B,3,H,W RGB')
    if mode == 'raw':
        return image
    with torch.autocast(image.device.type, enabled=False):
        value = image if image.dtype == torch.float64 else image.float()
        centered = value - value.mean((-2, -1), keepdim=True)
        dims = (1, 2, 3) if mode == 'shared_rms' else (-2, -1)
        scale = centered.square().mean(dims, keepdim=True).clamp_min(1e-12).sqrt()
        return center.to(value) + .20 * centered / scale


class DualCoordinateJoint(AffineSplitJoint):
    """Same parameter shapes as additive V16; only the semantic input changes."""

    def __init__(self, seed, mode, pretrained=True):
        if mode not in METHODS:
            raise ValueError(mode)
        super().__init__(seed, 'additive', 11, pretrained)
        self.mode = mode

    def semantic(self, image):
        return super().semantic(canonical_rgb(image, self.mean, self.mode))

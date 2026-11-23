"""A matched coarse correction with either free or observed detail."""
import torch
import torch.nn.functional as F

from boundary_refinement.models.joint.boundary_flux import BoundaryRefiner, raw_affinities


RECIPES = {
    "synthetic_free": ("synthetic", "free"),
    "synthetic_detail": ("synthetic", "observed"),
    "affine_mean": ("donor_mean", "free"),
    "affine_spatial": ("donor_spatial", "free"),
    "paired_free": ("paired", "free"),
    "paired_detail": ("paired", "observed"),
}


def high_pass(value):
    low = F.avg_pool2d(F.pad(value, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    return value - low


class AppearanceRefiner(BoundaryRefiner):
    """Constrain the fine correction direction to the observed detail difference."""
    def __init__(self, seed, choice):
        if choice not in ("free", "observed"):
            raise ValueError("Invalid detail mode")
        super().__init__(seed + 22_000_000, "plain")
        self.detail_mode = choice

    def forward(self, image, anchor, weights):
        _, coarse, proposal = super().forward(image, anchor, weights)
        if self.detail_mode == "free":
            detail = high_pass(proposal)
        else:
            # Both arms start exactly at the same anchor. The observed arm
            # can only borrow a bounded amount of detail from the raw input.
            alpha = (3.125 * proposal).clamp(0, 1)
            detail = high_pass(alpha * high_pass(image - anchor))
        return anchor + coarse + detail


@torch.no_grad()
def restoration_anchor(core, refiner, image):
    global_rgb, _, _ = core.global_affine(image)
    base = (global_rgb.float() + (core.rgb_net(image).float() - image)).clamp(0, 1)
    result, _, _ = refiner(image, base, raw_affinities(image))
    return result.clamp(0, 1)


def appearance_loss(restored, clean, valid):
    from boundary_refinement.data.camvid_data import photo_loss
    difference = restored - clean
    mask = valid[:, None].float()
    channel_bias = (difference * mask).sum((-2, -1)) / mask.sum((-2, -1)).clamp_min(1)
    return photo_loss(restored, clean, valid) + 0.25 * channel_bias.abs().mean()

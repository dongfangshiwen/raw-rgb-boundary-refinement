"""Corridor support / visible-surface factorization, an exploratory dual-output model.

Distance supervision, multiplicative masks, consistency and FPNs have prior art.
These are candidates for empirical investigation, not established novel contributions.
No reference image, mask, occlusion map or atmospheric parameter is an inference input.
"""
import copy
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from scripts.coefficient_joint_v9 import CoefficientJoint, apply_fields
from scripts.parallel_joint_v8 import conv_block, rgb_loss
from scripts.partial_joint_v7 import partial_segmentation_loss


@dataclass(frozen=True)
class Variant:
    factor: bool = False
    geometry: bool = False
    counterfactual: str = "none"
    task: str = "joint"
    decoder: str = "fpn"
    public_labels: bool = True


METHODS = {
    "base": Variant(),
    "factor": Variant(factor=True),
    "geometry": Variant(geometry=True),
    "counterfactual": Variant(counterfactual="support"),
    "factor_geometry": Variant(True, True),
    "factor_counterfactual": Variant(True, False, "support"),
    "geometry_counterfactual": Variant(False, True, "support"),
    "full": Variant(True, True, "support"),
    "full_aux_only": Variant(True, True, "aux"),
    "full_visible_kl": Variant(True, True, "visible"),
    "full_shuffled": Variant(True, True, "shuffled"),
    "seg_only": Variant(True, True, "support", "seg"),
    "rgb_only": Variant(True, True, "support", "rgb"),
    "independent": Variant(True, True, "support", "independent"),
    "deeplab": Variant(decoder="deeplab"),
    "base_no_public_labels": Variant(public_labels=False),
}
FACTORIAL = tuple(list(METHODS)[:8])
CONFIRM = ("base", "full", "geometry_counterfactual", "factor_counterfactual", "factor_geometry")


def visible_logits(support, visibility, factor):
    """Stable exact Bernoulli conjunction; additive capacity control uses both heads."""
    s, v = support.float(), visibility.float()
    if factor:
        # p/(1-p) = exp(s+v)/(1+exp(s)+exp(v)); finite at extreme logits.
        delta = s + v - torch.logsumexp(torch.stack((torch.zeros_like(s), s, v)), 0)
    else:
        delta = s + v - 4.
    return torch.cat((torch.zeros_like(delta), delta), 1)


class GeometryVisibilityJoint(CoefficientJoint):
    def __init__(self, seed=42, method="base", pretrained=True):
        if method not in METHODS:
            raise ValueError("Unknown V10 treatment")
        super().__init__(seed, "free_supervised", pretrained)
        self.method, self.variant = method, METHODS[method]
        self.independent_encoder = copy.deepcopy(self.encoder) if self.variant.task == "independent" else None
        if self.variant.decoder != "deeplab":
            del self.segmenter
        # Separate seed isolates new heads from the established RGB initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 10_000_000)
            self.seg_deep = conv_block(960, 64)
            self.seg_mid_project = nn.Conv2d(40, 32, 1)
            self.seg_mid = conv_block(96, 64)
            self.seg_low_project = nn.Conv2d(24, 32, 1)
            self.seg_low = conv_block(96, 32)
            self.support_head = nn.Conv2d(32, 1, 3, padding=1)
            self.visibility_head = nn.Conv2d(32, 1, 3, padding=1)
            self.distance_head = nn.Conv2d(32, 1, 3, padding=1)
            nn.init.zeros_(self.visibility_head.weight)
            nn.init.constant_(self.visibility_head.bias, 4.)
            nn.init.zeros_(self.distance_head.weight)
            nn.init.zeros_(self.distance_head.bias)
        if self.variant.task == "seg":
            for name in ("rgb_encoder", "rgb_project", "mid_project", "low_project", "rgb_mid", "rgb_low", "field_head", "airlight_head"):
                getattr(self, name).requires_grad_(False)
        elif self.variant.task == "rgb":
            for name in ("seg_deep", "seg_mid_project", "seg_mid", "seg_low_project", "seg_low", "support_head", "visibility_head", "distance_head", "aux_semantic"):
                getattr(self, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Pretrained running statistics remain fixed for ALL methods. Affine learns.
        for encoder in (self.encoder, self.rgb_encoder, self.independent_encoder):
            if encoder is not None:
                for layer in encoder.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
        return self

    @staticmethod
    def capture(encoder, value):
        result = {}
        for name, layer in encoder.items():
            value = layer(value)
            if name in {"3", "6", "12", "16"}:
                result[name] = value
        return result

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected NCHW RGB only")
        native = image.shape[-2:]
        canonical = image if native == (384, 384) else F.interpolate(image, (384, 384), mode="bilinear", align_corners=False, antialias=True)
        normalized = (canonical - self.mean) / self.std
        sem = self.capture(self.encoder, normalized)
        if self.independent_encoder is not None:
            photo = self.capture(self.independent_encoder, normalized)
        else:
            deep = sem["12"]
            for layer in self.rgb_encoder.values():
                deep = layer(deep)
            photo = {"3": sem["3"], "6": sem["6"], "16": deep}
        feature = self.seg_deep(sem["16"])
        mid = self.seg_mid_project(sem["6"])
        feature = self.seg_mid(torch.cat((F.interpolate(feature, mid.shape[-2:], mode="bilinear", align_corners=False), mid), 1))
        low = self.seg_low_project(sem["3"])
        feature = self.seg_low(torch.cat((F.interpolate(feature, low.shape[-2:], mode="bilinear", align_corners=False), low), 1))
        distance = self.distance_head(feature).float().tanh()
        support = self.support_head(feature).float()
        if self.variant.geometry:
            support = support + 4 * distance
        visibility = self.visibility_head(feature).float()
        logits = visible_logits(support, visibility, self.variant.factor)
        if self.variant.decoder == "deeplab":
            logits = self.segmenter(sem["16"])
        logits = F.interpolate(logits, native, mode="bilinear", align_corners=False)
        latent = self.rgb_project(photo["16"])
        airlight = self.airlight_head(latent.float().mean((2, 3))).sigmoid()[:, :, None, None]
        mid = self.mid_project(photo["6"])
        value = self.rgb_mid(torch.cat((F.interpolate(latent, mid.shape[-2:], mode="bilinear", align_corners=False), mid), 1))
        low = self.low_project(photo["3"])
        value = self.rgb_low(torch.cat((F.interpolate(value, low.shape[-2:], mode="bilinear", align_corners=False), low), 1))
        fields = F.adaptive_avg_pool2d(self.field_head(value).float(), (24, 24))
        log_gain, bias = math.log(6) * fields[:, :1].tanh(), 4 * fields[:, 1:].tanh()
        restored, _ = apply_fields(image, log_gain, bias, airlight, "free")
        return restored, logits, dict(support=support, visibility=visibility, distance=distance,
            semantic=self.aux_semantic(sem["16"]), log_gain=log_gain, bias=bias, airlight=airlight)


def balanced_bce(logits, target, known):
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    terms = [loss[known & (target == label)].mean() for label in (0, 1) if bool((known & (target == label)).any())]
    return sum(terms) / len(terms) if terms else logits.sum() * 0


def components(model, batch):
    image, reference, mask = (batch[k] for k in ("image", "reference", "visible_mask"))
    n = batch["scene_count"]
    restored, logits, aux = model(image)
    if not model.variant.public_labels:
        mask = mask.clone()
        mask[batch["complete"].repeat(4)] = 255
    semantic = partial_segmentation_loss(logits, mask)
    coarse = F.interpolate(aux["semantic"].float(), mask.shape[-2:], mode="bilinear", align_corners=False)
    semantic = semantic + .1 * partial_segmentation_loss(coarse, mask)
    t = F.interpolate(batch["transmission"], aux["log_gain"].shape[-2:], mode="bilinear", align_corners=False)
    coefficient = .1 * F.l1_loss(aux["log_gain"][:n], torch.zeros_like(aux["log_gain"][:n]))
    for view in (2, 3):
        region = slice(view * n, (view + 1) * n)
        coefficient = coefficient + .05 * F.l1_loss(aux["log_gain"][region], -t.log())
        coefficient = coefficient + .05 * F.l1_loss(aux["airlight"][region], batch["airlight"])
        coefficient = coefficient + .0125 * F.l1_loss(aux["bias"][region], (1 - t.reciprocal()) * batch["airlight"])
    photo = rgb_loss(restored, reference) + coefficient
    geometry = semantic * 0
    if model.variant.geometry:
        target = batch["distance"].repeat(4, 1, 1, 1)
        complete = batch["complete"].repeat(4)
        if bool(complete.any()):
            error = (aux["distance"][complete] - target[complete]).abs()
            # A band prevents the much larger far-background area dominating the SDF.
            band = (target[complete].abs() < 1)
            geometry = .1 * (error[band].mean() if bool(band.any()) else error.mean())
    counterfactual = semantic * 0
    mode = model.variant.counterfactual
    if mode != "none":
        size = aux["support"].shape[-2:]
        original = F.interpolate(batch["support_mask"][:, None].float(), size, mode="nearest")
        known = original != 255
        occlusion = F.interpolate(batch["occlusion"][:, None].float(), size, mode="nearest") > .5
        support = aux["support"]
        counterfactual = .2 * balanced_bce(support[3*n:4*n], (original == 1).float(), known)
        vis_target = (~occlusion).float()
        vis_known = occlusion | (original == 1)
        counterfactual = counterfactual + .2 * balanced_bce(aux["visibility"][3*n:4*n], vis_target, vis_known)
        consistency = semantic * 0
        if mode in {"support", "shuffled"}:
            teacher = support[2*n:3*n].detach().sigmoid()
            if mode == "shuffled":
                teacher = torch.roll(teacher, (size[0]//2, size[1]//2), (-2, -1))
            error = F.binary_cross_entropy_with_logits(support[3*n:4*n], teacher, reduction="none")
            consistency = error[known].mean()
        elif mode == "visible":
            small = F.interpolate(logits.float(), size, mode="bilinear", align_corners=False)
            teacher = small[2*n:3*n].detach().softmax(1)
            error = -(teacher * small[3*n:4*n].log_softmax(1)).sum(1, keepdim=True)
            valid = known & ~occlusion
            consistency = error[valid].mean() if bool(valid.any()) else error.sum() * 0
        counterfactual = counterfactual + .1 * consistency
    semantic_total = semantic + geometry + counterfactual
    if model.variant.task == "seg":
        total = semantic_total
    elif model.variant.task == "rgb":
        total = photo
    else:
        total = semantic_total + photo
    return total, dict(total=float(total.detach()), semantic=float(semantic.detach()), rgb=float(photo.detach()),
        geometry=float(geometry.detach()), counterfactual=float(counterfactual.detach()))


def load_model(path, device="cuda"):
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if packet.get("format") != "geometry-visibility-v10":
        raise ValueError("Wrong V10 checkpoint")
    model = GeometryVisibilityJoint(packet["seed"], packet["method"], False)
    model.load_state_dict(packet["model_state"], strict=True)
    return model.to(device).eval(), packet

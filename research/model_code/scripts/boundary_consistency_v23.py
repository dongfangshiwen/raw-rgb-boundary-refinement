"""Trainable RGB restoration and clean-reference adjacent-class consistency."""
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from scripts.boundary_flux_v21 import raw_affinities
from scripts.dual_coordinate_v17 import canonical_rgb
from scripts.infer_real_transfer_v22 import load_model


SCOPES = ("head_only", "decoder_tail", "full_rgb")
LOSSES = ("photo_only", "global_semantic", "rgb_boundary", "class_boundary")


class AdaptiveRestorer(nn.Module):
    def __init__(self, core, anchor, correction):
        super().__init__()
        self.affine = core.global_affine
        self.naf = core.rgb_net
        self.anchor = anchor
        self.correction = correction

    def forward(self, image, weights):
        global_rgb, _, _ = self.affine(image)
        base = (global_rgb.float() + (self.naf(image).float() - image)).clamp(0, 1)
        refined, _, _ = self.anchor(image, base, weights)
        return self.correction(image, refined.clamp(0, 1), weights)

    def set_training_scope(self, scope):
        if scope not in SCOPES:
            raise ValueError("Unknown RGB training scope")
        self.requires_grad_(scope == "full_rgb")
        self.correction.requires_grad_(True)
        if scope == "decoder_tail":
            for module in (self.affine.output, self.naf.ups[-1], self.naf.decoders[-1], self.naf.ending, self.anchor):
                module.requires_grad_(True)
        # The inherited restorer has no train-mode statistics or stochastic layers.
        return {key: sum(p.numel() for p in module.parameters() if p.requires_grad)
                for key, module in self.named_children()}


def from_v22(checkpoint):
    core, anchor, heads, metadata = load_model(checkpoint, ("paired_free",))
    return core, AdaptiveRestorer(core, anchor, heads["paired_free"]).eval(), metadata


def copy_restorer(model, scope):
    clone = deepcopy(model)
    clone.set_training_scope(scope)
    return clone


def trainable_weights(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def load_adaptation(model, values, scope):
    model.set_training_scope(scope)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(values) != set(parameters):
        raise ValueError("Checkpoint contains the wrong parameter set")
    with torch.no_grad():
        for name, param in parameters.items():
            if param.shape != values[name].shape or param.dtype != values[name].dtype:
                raise ValueError("Checkpoint parameter differs in shape or type")
            param.copy_(values[name])
    return model


def semantic_scores(core, image):
    _, logits = core.semantic(canonical_rgb(image, core.mean, "channel"))
    return logits.float()


def adjacent(value, axis):
    length = value.shape[axis] - 1
    return value.narrow(axis, 0, length), value.narrow(axis, 1, length)


def adjacent_margins(logits, classes_a, classes_b, axis):
    left, right = adjacent(logits, axis)
    def value(score, label):
        return score.gather(1, label[:, None]).squeeze(1)
    m_a = value(left, classes_a) - value(left, classes_b)
    m_b = value(right, classes_b) - value(right, classes_a)
    return torch.stack((m_a, m_b), dim=1)


def quarter_rgb(image, size):
    return F.adaptive_avg_pool2d(image, size)


@torch.no_grad()
def teacher_targets(core, clean, labels, repeats):
    logits = semantic_scores(core, clean)
    size = logits.shape[-2:]
    labels = F.interpolate(labels[:, None].float(), size=size, mode="nearest-exact")[:, 0].long()
    known = labels < logits.shape[1]
    safe = labels.masked_fill(~known, 0)
    quarter = quarter_rgb(clean, size)
    pairs = []
    for axis in (-1, -2):
        a, b = adjacent(safe, axis)
        known_a, known_b = adjacent(known, axis)
        clear_margins = adjacent_margins(logits, a, b, axis)
        mask = known_a & known_b & (a != b) & (clear_margins > 0).all(dim=1)
        rgb_a, rgb_b = adjacent(quarter, axis)
        pairs.append(dict(axis=axis, a=a.repeat(repeats, 1, 1), b=b.repeat(repeats, 1, 1),
                          mask=mask.repeat(repeats, 1, 1),
                          margin=(clear_margins/2).sigmoid().repeat(repeats, 1, 1, 1),
                          rgb_delta=(rgb_a-rgb_b).repeat(repeats, 1, 1, 1)))
    return dict(known=known.repeat(repeats, 1, 1), probability=(logits/2).softmax(1).repeat(repeats, 1, 1, 1),
                pairs=pairs, size=size)


def semantic_auxiliary(name, core, restored, targets):
    if name == "rgb_boundary":
        values = quarter_rgb(restored, targets["size"])
    else:
        values = semantic_scores(core, restored)
    if name == "global_semantic":
        difference = F.kl_div((values/2).log_softmax(1), targets["probability"], reduction="none").sum(1)
        known = targets["known"]
        return (difference * known).sum() / known.sum().clamp_min(1)
    total, count = 0, 0
    for pair in targets["pairs"]:
        mask = pair["mask"]
        if name == "class_boundary":
            margin = adjacent_margins(values, pair["a"], pair["b"], pair["axis"])
            difference = F.smooth_l1_loss((margin/2).sigmoid(), pair["margin"], beta=.10, reduction="none")
        elif name == "rgb_boundary":
            a, b = adjacent(values, pair["axis"])
            difference = F.smooth_l1_loss(a-b, pair["rgb_delta"], beta=.10, reduction="none")
        else:
            raise ValueError("Unknown consistency loss")
        total = total + (difference.mean(1) * mask).sum()
        count = count + mask.sum()
    return total / count.clamp_min(1)


def edge_count(targets):
    return sum(int(item["mask"].sum()) for item in targets["pairs"])


@torch.no_grad()
def rgb(restorer, image):
    return restorer(image, raw_affinities(image)).clamp(0, 1)

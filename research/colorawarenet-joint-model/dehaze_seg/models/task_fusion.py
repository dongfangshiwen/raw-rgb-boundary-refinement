"""Shared-encoder raw/restored feature fusion for task-compatible dehazing.

The spatial gates are learned selectors, not calibrated uncertainty estimates.
The raw image is an inference input; paired clear images are training-only.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .LiteAttentionUnet import LiteAttentionUNet
from .joint import JointDehazeSegModel, unpack_dehaze_output


class SpatialFeatureGate(nn.Module):
    """Predict one gate per pixel from channel summaries and disagreement."""

    def __init__(self, hidden=8):
        super().__init__()
        self.predict = nn.Sequential(nn.Conv2d(6, hidden, 3, padding=1),
                                     nn.ReLU(inplace=False), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, raw, restored):
        # Float32 summaries avoid overflow in half-precision squares/reductions.
        h, d = raw.float(), restored.float()
        mh, md = h.mean(1, keepdim=True), d.mean(1, keepdim=True)
        difference = (h-d).abs().mean(1, keepdim=True)
        cues = torch.cat((mh, md, h.amax(1, keepdim=True), d.amax(1, keepdim=True),
                          difference, difference/(mh.abs()+md.abs()+1e-4)), 1)
        gate = self.predict(cues).sigmoid().to(raw.dtype)
        return raw + gate*(restored-raw), gate


class SharedFusionSegmenter(nn.Module):
    def __init__(self, mode="gated", **segmenter_config):
        super().__init__()
        if mode not in ("restored", "average", "gated"):
            raise ValueError("Fusion mode must be restored, average or gated")
        self.mode = mode
        self.network = LiteAttentionUNet(**segmenter_config)
        self.gates = nn.ModuleList(SpatialFeatureGate() for _ in range(5)) if mode == "gated" else nn.ModuleList()

    def encode(self, image):
        n = self.network
        features = [n.inc(image)]
        for down in (n.down1, n.down2, n.down3, n.down4):
            features.append(down(features[-1]))
        return features

    def decode(self, features, size):
        n = self.network
        x1, x2, x3, x4, x5 = features
        value = n.up3(n.bottleneck(x5), x4)
        value = n.up2(value, x3)
        value = n.up1(value, x2)
        value = n.up0(value, x1)
        return F.interpolate(n.outc(value), size=size, mode="bilinear", align_corners=False)

    def forward(self, restored, raw=None, return_gates=False):
        if raw is None or self.mode == "restored":
            logits = self.network(restored)
            if isinstance(logits, dict):
                logits = logits["out"]
            return (logits, []) if return_gates else logits
        if restored.shape != raw.shape:
            raise ValueError("Raw and restored inputs must have identical shapes")
        batch = len(raw)
        # One mixed-domain encoder pass: average/gated controls share BN policy.
        pairs = [value.split(batch, 0) for value in self.encode(torch.cat((raw, restored), 0))]
        features, gate_maps = [], []
        for index, (h, d) in enumerate(pairs):
            if self.mode == "gated":
                fused, gate = self.gates[index](h, d)
                gate_maps.append(gate)
            else:
                fused = (h+d)*.5
            features.append(fused)
        logits = self.decode(features, restored.shape[-2:])
        return (logits, gate_maps) if return_gates else logits


class FusionDehazeSegModel(JointDehazeSegModel):
    def __init__(self, dehazer, segmenter, imagenet_norm=True, detach_restored=True):
        super().__init__(dehazer, segmenter, imagenet_norm)
        self.detach_restored = bool(detach_restored)

    def normalize(self, image):
        return (image-self.imagenet_mean)/self.imagenet_std if self.imagenet_norm else image

    def forward(self, hazy):
        restored, aux = unpack_dehaze_output(self.dehazer(hazy))
        task_image = restored.detach() if self.detach_restored else restored
        logits, gates = self.segmenter(self.normalize(task_image), self.normalize(hazy), return_gates=True)
        return restored, logits, {**aux, "fusion_gates": gates}

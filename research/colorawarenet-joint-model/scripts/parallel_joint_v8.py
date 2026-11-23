"""Non-serial shared-encoder RGB/semantic research model.

All proposals remain hypotheses. Statistics conditioning, feature interaction and
paired intervention have substantial prior art; this file does not establish novelty.
No restored RGB, clean target, mask or fog parameter is an inference input.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from scripts.vision_baseline_v1 import build_model as vision_model
from scripts.partial_joint_v7 import partial_segmentation_loss


@dataclass(frozen=True)
class Variant:
    factor: str = "none"
    exchange: str = "none"
    intervention: str = "none"
    task: str = "joint"


METHODS = {
    "joint": Variant(),
    "factor": Variant("moments"),
    "exchange": Variant(exchange="reliable"),
    "intervention": Variant(intervention="both"),
    "factor_exchange": Variant("moments", "reliable"),
    "factor_intervention": Variant("moments", intervention="both"),
    "exchange_intervention": Variant(exchange="reliable", intervention="both"),
    "full": Variant("moments", "reliable", "both"),
    "factor_no_stats": Variant("no_stats"),
    "factor_raw_adapter": Variant("raw"),
    "full_ungated": Variant("moments", "ungated", "both"),
    "full_learned_gate": Variant("moments", "learned", "both"),
    "full_feature_only": Variant("moments", "reliable", "feature"),
    "full_weather_only": Variant("moments", "reliable", "weather"),
    "full_shuffled": Variant("moments", "shuffled", "both"),
    "full_rgb_to_seg": Variant("moments", "rgb_to_seg", "both"),
    "full_seg_to_rgb": Variant("moments", "seg_to_rgb", "both"),
    "rgb_only": Variant(task="rgb"),
    "seg_only": Variant(task="seg"),
    "independent": Variant(task="independent"),
}
FACTORIAL = tuple(list(METHODS)[:8])


def conv_block(n_in, n_out):
    return nn.Sequential(nn.Conv2d(n_in, n_out, 3, padding=1, bias=False),
                         nn.GroupNorm(8, n_out), nn.SiLU())


def moments(feature):
    value = feature.float()
    mean = value.mean((2, 3), keepdim=True)
    scale = (value.var((2, 3), unbiased=False, keepdim=True) + 1e-4).sqrt()
    return mean, scale


class ParallelJoint(nn.Module):
    def __init__(self, seed=42, method="joint", pretrained=True):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown non-serial variant")
        self.method, self.variant, self.seed = method, METHODS[method], seed
        segmenter = vision_model("deeplab_mobile_imagenet" if pretrained else "deeplab_mobile_scratch", seed)
        self.encoder = segmenter.network.backbone
        self.segmenter = segmenter.network.classifier
        self.register_buffer("mean", segmenter.mean.clone())
        self.register_buffer("std", segmenter.std.clone())
        # A separate encoder is a capacity/compute control, initialized identically.
        self.rgb_encoder = None
        if self.variant.task == "independent":
            import copy
            self.rgb_encoder = copy.deepcopy(self.encoder)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 8_000_000)
            self.factor_adapter = nn.Sequential(nn.Conv2d(960, 32, 1), nn.SiLU(), nn.Conv2d(32, 960, 1))
            self.rgb_project = conv_block(960, 64)
            self.stats_project = nn.Sequential(nn.Conv2d(1920, 32, 1), nn.SiLU(), nn.Conv2d(32, 64, 1))
            self.rgb_from_semantic = nn.Sequential(nn.Conv2d(960, 32, 1), nn.SiLU(), nn.Conv2d(32, 64, 1))
            self.semantic_from_rgb = nn.Sequential(nn.Conv2d(64, 32, 1), nn.SiLU(), nn.Conv2d(32, 960, 1))
            self.aux_semantic = nn.Conv2d(960, 2, 1)
            self.aux_error = nn.Conv2d(64, 1, 1)
            self.aux_t = nn.Conv2d(64, 1, 1)
            self.aux_airlight = nn.Linear(64, 3)
            self.low_project = nn.Conv2d(24, 32, 1)
            self.mid_project = nn.Conv2d(40, 32, 1)
            self.rgb_mid = conv_block(96, 64)
            self.rgb_low = conv_block(96, 32)
            self.rgb_refine = nn.Sequential(conv_block(35, 24), nn.Conv2d(24, 3, 3, padding=1))
            # Equal initial RGB identity and equal semantic initialization in all variants.
            for module in (self.factor_adapter[-1], self.stats_project[-1],
                           self.rgb_from_semantic[-1], self.semantic_from_rgb[-1], self.rgb_refine[-1]):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def encode(encoder, value):
        features = {}
        for name, layer in encoder.items():
            value = layer(value)
            if name in ("3", "6", "16"):
                features[name] = value
        if set(features) != {"3", "6", "16"}:
            raise ValueError("Backbone feature topology changed")
        return features

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected NCHW RGB image only")
        features = self.encode(self.encoder, (image - self.mean) / self.std)
        rgb_features = features if self.rgb_encoder is None else self.encode(self.rgb_encoder, (image-self.mean)/self.std)
        original = features["16"]
        center, scale = moments(original)
        semantic = original
        if self.variant.factor != "none":
            conditioning = original.float() if self.variant.factor == "raw" else (original.float()-center)/scale
            bound = .25*(original.float().square().mean((2,3),keepdim=True)+1e-6).sqrt().detach()
            semantic = original + (bound*torch.tanh(self.factor_adapter(conditioning))).to(original.dtype)
        appearance = self.rgb_project(rgb_features["16"])
        rgb = appearance
        if self.variant.factor in ("moments", "raw"):
            rgb = rgb + self.stats_project(torch.cat((center, scale.log()), 1)).to(rgb.dtype)
        aux_semantic = self.aux_semantic(semantic)
        error_probability = self.aux_error(rgb).float().sigmoid()
        probabilities = aux_semantic.float().softmax(1)
        semantic_confidence = (1 + (probabilities*probabilities.clamp_min(1e-8).log()).sum(1,keepdim=True)/.6931471805599453).clamp(0,1)
        rgb_confidence = (1-error_probability).clamp(0,1)
        before_exchange = semantic
        if self.variant.exchange != "none":
            sr_gate = semantic_confidence.detach()
            rs_gate = rgb_confidence.detach()
            if self.variant.exchange == "ungated":
                sr_gate=torch.ones_like(sr_gate);rs_gate=torch.ones_like(rs_gate)
            elif self.variant.exchange == "learned":
                sr_gate=semantic_confidence;rs_gate=rgb_confidence
            elif self.variant.exchange == "shuffled":
                shift=(max(1,sr_gate.shape[-2]//2),max(1,sr_gate.shape[-1]//2))
                sr_gate=torch.roll(sr_gate,shift,(-2,-1));rs_gate=torch.roll(rs_gate,shift,(-2,-1))
            elif self.variant.exchange == "rgb_to_seg":sr_gate=torch.zeros_like(sr_gate)
            elif self.variant.exchange == "seg_to_rgb":rs_gate=torch.zeros_like(rs_gate)
            semantic_delta = self.semantic_from_rgb(rgb)
            rgb_delta = self.rgb_from_semantic(semantic)
            # Both messages use pre-exchange representations (no RGB-image cascade).
            semantic = semantic + rs_gate*semantic_delta
            rgb = rgb + sr_gate*rgb_delta
        logits = F.interpolate(self.segmenter(semantic),image.shape[-2:],mode="bilinear",align_corners=False)
        mid = self.mid_project(rgb_features["6"])
        rgb = self.rgb_mid(torch.cat((F.interpolate(rgb,mid.shape[-2:],mode="bilinear",align_corners=False),mid),1))
        low = self.low_project(rgb_features["3"])
        rgb = self.rgb_low(torch.cat((F.interpolate(rgb,low.shape[-2:],mode="bilinear",align_corners=False),low),1))
        rgb = F.interpolate(rgb,image.shape[-2:],mode="bilinear",align_corners=False)
        residual = self.rgb_refine(torch.cat((rgb,image),1)).float().tanh()
        restored = (image.float()+residual).clamp(0,1)
        # Physical auxiliaries concern controlled synthetic inputs only; no real t/depth claim.
        aux = dict(content=before_exchange,semantic=aux_semantic,error=error_probability,
                   transmission=.2+.8*self.aux_t(appearance).float().sigmoid(),
                   airlight=.7+.3*self.aux_airlight(appearance.float().mean((2,3))).sigmoid(),
                   semantic_confidence=semantic_confidence,rgb_confidence=rgb_confidence)
        return restored, logits, aux


def rgb_loss(restored, target):
    # RGB opponent differences constrain color without assuming chromaticity survives colored haze.
    l1 = F.l1_loss(restored, target)
    opponent = F.l1_loss(restored[:,0:1]-restored[:,1:2], target[:,0:1]-target[:,1:2])
    opponent = opponent + F.l1_loss(restored[:,2:3]-restored[:,1:2], target[:,2:3]-target[:,1:2])
    edge = F.l1_loss(restored[:,:,1:]-restored[:,:,:-1],target[:,:,1:]-target[:,:,:-1])
    edge = edge + F.l1_loss(restored[:,:,:,1:]-restored[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])
    return l1 + .1*opponent + .05*edge


def intervention_loss(aux, scene_count, transmission, airlight, mode):
    if mode == "none":
        return aux["content"].sum()*0, dict(feature=0.,weather=0.)
    # Batch layout: clear N, original hazy N, controlled synthetic N.
    content = F.normalize(F.adaptive_avg_pool2d(aux["content"].float(),(8,8)),dim=1)
    reference = content[:scene_count].detach()
    feature = F.mse_loss(content[scene_count:2*scene_count],reference) + F.mse_loss(content[2*scene_count:],reference)
    truth = F.interpolate(transmission,aux["transmission"].shape[-2:],mode="bilinear",align_corners=False)
    weather = F.l1_loss(aux["transmission"][2*scene_count:],truth) + F.l1_loss(aux["airlight"][2*scene_count:],airlight.flatten(1))
    total = (10*feature if mode in ("both","feature") else 0) + (.1*weather if mode in ("both","weather") else 0)
    return total, dict(feature=float(feature.detach()),weather=float(weather.detach()))


def joint_loss(model, image, target_rgb, target_mask, scene_count, transmission, airlight):
    restored, logits, aux = model(image)
    semantic = partial_segmentation_loss(logits,target_mask)
    auxiliary = F.interpolate(aux["semantic"].float(),target_mask.shape[-2:],mode="bilinear",align_corners=False)
    semantic = semantic + .1*partial_segmentation_loss(auxiliary,target_mask)
    photo = rgb_loss(restored,target_rgb)
    error_target = F.adaptive_avg_pool2d((restored.detach()-target_rgb).abs().mean(1,keepdim=True),aux["error"].shape[-2:])
    calibration = .05*F.l1_loss(aux["error"],error_target)
    if model.variant.exchange == "learned":calibration=calibration*0
    inter,details = intervention_loss(aux,scene_count,transmission,airlight,model.variant.intervention)
    loss = photo + calibration if model.variant.task == "rgb" else semantic if model.variant.task == "seg" else semantic+photo+calibration+inter
    return loss,dict(total=float(loss.detach()),semantic=float(semantic.detach()),rgb=float(photo.detach()),calibration=float(calibration.detach()),**details)


def load_model(path, device="cuda"):
    packet = torch.load(path,map_location="cpu",weights_only=True)
    if packet.get("format") != "parallel-joint-v8":
        raise ValueError("Wrong V8 checkpoint")
    model = ParallelJoint(packet["seed"],packet["method"],pretrained=False)
    model.load_state_dict(packet["model_state"],strict=True)
    return model.to(device).eval(),packet

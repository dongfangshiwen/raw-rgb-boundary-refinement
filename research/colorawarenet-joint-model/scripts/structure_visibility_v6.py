"""Source-only structure recovery, support risk, and bounded visibility training.

This is an experimental segmentation model. RGB restoration is deliberately not
represented by the identity first return value; an existing RGB dehazer can be
attached separately. No ground truth or clear reference is used at inference.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from scripts.vision_baseline_v1 import build_model as build_reference


@dataclass(frozen=True)
class Variant:
    geometry: str = "none"
    support: str = "positive"
    fog: str = "random"


METHODS = {
    "control": Variant(),
    "geometry_aux": Variant("aux"),
    "geometry_isotropic": Variant("isotropic"),
    "geometry": Variant("steer"),
    "negative_crops": Variant(support="crops"),
    "support": Variant(support="risk"),
    "fog_unconstrained": Variant(fog="adversarial_free"),
    "fog": Variant(fog="adversarial"),
    "geometry_support": Variant("steer", "risk"),
    "geometry_fog": Variant("steer", fog="adversarial"),
    "support_fog": Variant(support="risk", fog="adversarial"),
    "full": Variant("steer", "risk", "adversarial"),
}
FACTORIAL = ("control", "geometry", "support", "fog", "geometry_support",
             "geometry_fog", "support_fog", "full")


def gradients(value):
    padded = F.pad(value, (1,1,1,1), mode="reflect")
    return ((padded[...,1:-1,2:]-padded[...,1:-1,:-2])*.5,
            (padded[...,2:,1:-1]-padded[...,:-2,1:-1])*.5)


def structure_tensor(image, size=None):
    """Return trace-normalized tensor anisotropy and scale-free reliability.

    Q = .5*[[1+d0,d1],[d1,1-d0]]. Constant-transmission affine fog
    scales the unnormalized tensor by t**2. The relative ridge also scales
    by t**2; the 1e-12 numerical floor makes this approximately invariant.
    Spatially varying transmission is NOT exactly invariant.
    """
    image = image.float()
    if size is None:
        size = (image.shape[-2]//4, image.shape[-1]//4)
    gray = (image*image.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    gray = F.interpolate(gray, size=size, mode="bilinear", align_corners=False)
    dx, dy = gradients(gray)
    xx, yy, xy = [F.avg_pool2d(F.pad(z,(2,2,2,2),mode="reflect"),5,1)
                  for z in (dx.square(),dy.square(),dx*dy)]
    trace = xx+yy
    scale = trace.mean((2,3),keepdim=True)
    denominator = trace + scale*.001 + 1e-12
    vector = torch.cat(((xx-yy)/denominator,2*xy/denominator),1)
    reliability = trace/(trace+scale+1e-12)
    return vector, reliability


def structure_distance(image, reference):
    value, _ = structure_tensor(image)
    target, weight = structure_tensor(reference)
    return ((value-target).abs()*weight).sum((1,2,3))/(2*weight.sum((1,2,3))+1e-12)


def scatter_fog(clear, transmission, airlight):
    field = F.interpolate(transmission, size=clear.shape[-2:], mode="bilinear", align_corners=False)
    return clear*field+airlight*(1-field)


@torch.no_grad()
def project_structure(clear, transmission, airlight, tolerance=.025):
    """Shrink spatial variation, preserving mean transmission and airlight.

    The fallback uniform transmission satisfies the affine-fog invariant.
    This constrains the chosen descriptor, not semantic correctness in general.
    """
    center = transmission.mean((2,3),keepdim=True)
    current = transmission.clone()
    scale = torch.ones(len(clear),device=clear.device)
    for step in range(7):
        distance = structure_distance(scatter_fog(clear,current,airlight),clear)
        reject = distance > tolerance
        if not bool(reject.any()):
            break
        scale = torch.where(reject,scale*.5,scale)
        if step == 6:
            scale = torch.where(reject,torch.zeros_like(scale),scale)
        current = center+(transmission-center)*scale[:,None,None,None]
    final = structure_distance(scatter_fog(clear,current,airlight),clear)
    if bool((final > tolerance+1e-4).any()):
        raise RuntimeError("Descriptor projection failed its stated constraint")
    return current, final, scale


class GeometrySteering(nn.Module):
    def __init__(self, channels=24):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(channels,channels,3,padding=1,groups=channels),
                                  nn.SiLU(),nn.Conv2d(channels,16,1),nn.SiLU(),nn.Conv2d(16,2,1))
        self.project = nn.Conv2d(channels,channels,1,bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, feature, mode):
        vector = self.head(feature).float()
        vector = vector/(1+vector.square().sum(1,keepdim=True)).sqrt()
        if mode == "aux":
            return feature, vector
        dx, dy = gradients(feature.float())
        energy = dx.square()+dy.square()
        if mode == "steer":
            signal = (vector[:,0:1]*(dx.square()-dy.square())+2*vector[:,1:2]*dx*dy)/(energy+1e-6)
        elif mode == "isotropic":
            signal = (energy/(energy.mean((2,3),keepdim=True)+1e-6)).clamp(0,2)-1
        else:
            raise ValueError(mode)
        bound = .25*(feature.float().square().mean((2,3),keepdim=True)+1e-6).sqrt().detach()
        delta = bound*torch.tanh(self.project(signal))
        return feature+delta.to(feature.dtype), vector


class StructureSegmenter(nn.Module):
    def __init__(self, reference, variant):
        super().__init__()
        self.network = reference.network
        self.register_buffer("mean",reference.mean.clone())
        self.register_buffer("std",reference.std.clone())
        self.variant = variant
        self.geometry = GeometrySteering()
        if variant.geometry == "none":
            self.geometry.requires_grad_(False)
        elif variant.geometry == "aux":
            self.geometry.project.requires_grad_(False)

    def forward(self, image):
        feature = (image-self.mean)/self.std
        auxiliary = {}
        for name, block in self.network.backbone.items():
            feature = block(feature)
            if name == "3" and self.variant.geometry != "none":
                feature, vector = self.geometry(feature,self.variant.geometry)
                auxiliary["structure"] = vector
        logits = self.network.classifier(feature)
        logits = F.interpolate(logits,size=image.shape[-2:],mode="bilinear",align_corners=False)
        return image, logits, auxiliary


def build_model(method, seed, pretrained=True):
    if method not in METHODS:
        raise ValueError(method)
    reference = build_reference("deeplab_mobile_imagenet" if pretrained else "deeplab_mobile_scratch",seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+7_000_000)
        return StructureSegmenter(reference,METHODS[method])


def load_model(path, device="cuda"):
    packet = torch.load(path,map_location="cpu",weights_only=True)
    if packet["format"] != "structure-visibility-v6":
        raise ValueError("Not a V6 checkpoint")
    model = build_model(packet["method"],packet["seed"],pretrained=False)
    model.load_state_dict(packet["model_state"],strict=True)
    return model.to(device).eval(), packet


def geometry_loss(prediction, clear):
    with torch.no_grad():
        target, weight = structure_tensor(clear,size=prediction.shape[-2:])
    return (F.smooth_l1_loss(prediction.float(),target,reduction="none")*weight).sum()/(2*weight.sum()+1e-6)


def support_risk(logits, target):
    """Last sample is an audited absent crop; positives supply a detached anchor.

    Tail averaging/ranking are established tools, not a novelty claim here.
    No extra negative penalty is placed on potentially incomplete source masks.
    """
    logits = F.interpolate(logits.float(),size=(128,128),mode="bilinear",align_corners=False)
    target = F.interpolate(target[:,None].float(),size=(128,128),mode="nearest")[:,0].bool()
    if bool(target[-1].any()):
        raise ValueError("Support risk needs an exactly empty reviewed crop")
    margin = logits[:,1]-logits[:,0]
    negative = margin[-1].flatten()
    tail = negative.topk(max(1,int(.01*len(negative)))).values.mean()
    positives = margin[:-1][target[:-1]]
    if not len(positives):
        raise ValueError("No annotated foreground for the positive anchor")
    anchor = torch.quantile(positives.detach(),.2)
    return .5*(F.softplus(tail+.5)+F.softplus(1+tail-anchor))


def visibility_step(model, clear, target, transmission, airlight, random_t, random_a,
                    mode, loss_fn, tolerance=.025):
    """One-step input-only ascent; random-direction control uses identical work.

    Inner evaluation does not update BatchNorm/dropout or model parameters.
    Clear references/labels/transmission are training-only, never inference inputs.
    """
    was_training = model.training
    model.eval()
    if mode != "adversarial_free":
        transmission,_,_ = project_structure(clear,transmission,airlight,tolerance)
    transmission = transmission.detach().requires_grad_(True)
    airlight = airlight.detach().requires_grad_(True)
    try:
        with torch.enable_grad(),torch.autocast(clear.device.type,enabled=False):
            inner = loss_fn(model(scatter_fog(clear,transmission,airlight))[1],target)
            gt,ga = torch.autograd.grad(inner,(transmission,airlight),only_inputs=True)
        if mode == "random":
            gt,ga = random_t,random_a
        elif mode not in ("adversarial", "adversarial_free"):
            raise ValueError(mode)
        if not bool(torch.isfinite(gt).all() and torch.isfinite(ga).all()):
            raise RuntimeError("Non-finite visibility gradient")
        next_t = (transmission.detach()+.08*gt.sign()).clamp(.35,.95)
        next_a = (airlight.detach()+.025*ga.sign()).clamp(.75,1.)
        if mode == "adversarial_free":
            distance = structure_distance(scatter_fog(clear,next_t,next_a),clear)
            shrink = torch.ones_like(distance)
        else:
            next_t,distance,shrink = project_structure(clear,next_t,next_a,tolerance)
        image = scatter_fog(clear,next_t,next_a).detach()
        stats = dict(inner_loss=float(inner.detach()),structure_distance=float(distance.mean()),
                     spatial_scale=float(shrink.mean()),mean_transmission=float(next_t.mean()),
                     mean_airlight=float(next_a.mean()))
    finally:
        model.train(was_training)
    return image, stats

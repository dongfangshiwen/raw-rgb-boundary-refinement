"""Structural control: shared early layers, private deep task encoders.

Caches share the same autograd graph, not recomputed BatchNorm passes. The
restoration branch never sends decoded RGB to the semantic branch. This is an
established multitask design family, not by itself a proven novel contribution.
"""
import copy
import torch
from torch import nn

from scripts.parallel_joint_v8 import ParallelJoint

METHODS={"branch6":6,"branch12":12}


class PartialSharing(ParallelJoint):
    def __init__(self,seed=42,method="branch12",pretrained=True):
        if method not in METHODS:raise ValueError("Unknown partial-sharing cut")
        super().__init__(seed,"joint",pretrained)
        self.method=method;self.cut=METHODS[method]
        self.rgb_encoder=nn.ModuleDict({name:copy.deepcopy(module) for name,module in self.encoder.items() if int(name)>self.cut})
        self._shared_features=None

    def encode(self,encoder,value):
        if encoder is self.encoder:
            captured={}
            for name,layer in encoder.items():
                value=layer(value)
                if name in {"3","6","16",str(self.cut)}:captured[name]=value
            self._shared_features=captured
            return {name:captured[name] for name in ("3","6","16")}
        if encoder is not self.rgb_encoder or self._shared_features is None:
            raise ValueError("Private encoder needs the current shared feature graph")
        shared=self._shared_features;self._shared_features=None
        private=shared[str(self.cut)]
        for name,layer in encoder.items():private=layer(private)
        return {"3":shared["3"],"6":shared["6"],"16":private}


def load_model(path,device="cuda"):
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if packet.get("format")!="parallel-joint-v8" or packet["method"] not in METHODS:
        raise ValueError("Wrong partial-sharing checkpoint")
    model=PartialSharing(packet["seed"],packet["method"],pretrained=False)
    model.load_state_dict(packet["model_state"],strict=True)
    return model.to(device).eval(),packet

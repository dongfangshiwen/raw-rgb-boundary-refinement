"""Single-scale inference from the authors' SegFix HRNet-W48 release on new images."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.projective_ledger_v12 import REPO,digest

ROOT=REPO.parent/"benchmark_data/segfix-author-v102"
SOURCE=ROOT/"source"
WEIGHT=ROOT/"segfix_hrnet_hrnet48_segfix_loss_iter80000_1_latest.pth"
SHA="cc7332acb4c1c5a30d97a061bb8d27023747011c6cd7b1d2fb83ece664a43d8b"
REV="aefc75517b09068d7131a69420bc5f66cb41f0ee"


class EvaluationABN(nn.BatchNorm2d):
    """Equivalent of the release's legacy InPlaceABN inference kernel."""
    def forward(self,x):
        return F.batch_norm(x,self.running_mean,self.running_var,self.weight.abs()+self.eps,
                            self.bias,False,0.,self.eps)


class ActivatedABN(EvaluationABN):
    def forward(self,x):
        return F.leaky_relu(super().forward(x),negative_slope=.01)


class Config:
    def __init__(self,network):
        self.network=network

    def get(self,section,key):
        assert section=="network"
        return self.network[key]


class SegFix:
    def __init__(self):
        assert digest(WEIGHT)==SHA
        for p in (SOURCE,ROOT/"dependencies"):
            if str(p) not in sys.path:
                sys.path.insert(0,str(p))
        from lib.models.nets.segfix import SegFix_HRNet
        from lib.models.tools.module_helper import ModuleHelper
        from lib.utils.helpers.offset_helper import label_to_vector_mapping
        scalar=np._core.multiarray.scalar
        safe=[(scalar,"numpy.core.multiarray.scalar"),np.dtype,type(np.dtype("float32")),
              type(np.dtype("float64")),type(np.dtype("int64"))]
        with torch.serialization.safe_globals(safe):
            release=torch.load(WEIGHT,map_location="cpu",weights_only=True)
        spec=release["config_dict"]
        assert spec["data"]["input_mode"]=="BGR" and spec["network"]["backbone"]=="hrnet48"
        with patch.object(ModuleHelper,"BatchNorm2d",side_effect=lambda **_:EvaluationABN), \
             patch.object(ModuleHelper,"BNReLU",side_effect=lambda n,**_:ActivatedABN(n)):
            model=SegFix_HRNet(Config({**spec["network"],"pretrained":None}))
        state={k.removeprefix("module."):v for k,v in release["state_dict"].items()}
        model.load_state_dict(state,strict=True)
        self.model=model.cuda().eval().requires_grad_(False)
        self.directions=torch.tensor(label_to_vector_mapping[8],device="cuda",dtype=torch.long)
        self.mean,self.std=(torch.tensor(spec["normalize"][k],device="cuda")[None,:,None,None]
                            for k in ("mean","std"))
        self.div=spec["normalize"]["div_value"]
        assert self.div==255 and self.directions.shape==(8,2)

    @torch.inference_mode()
    def lookup(self,rgb,return_offset=False):
        h,w=rgb.shape[:2]
        x=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float().flip(1)/self.div
        mask,direction=self.model((x-self.mean)/self.std)
        edge=F.interpolate(mask.softmax(1)[:,1:2],(h,w),mode="bilinear",align_corners=True)[0,0]>.5
        direction=F.interpolate(direction,(h,w),mode="bilinear",align_corners=True).argmax(1)[0]
        offset=self.directions[direction]*edge[:,:,None]*2
        y=torch.arange(h,device=x.device)[:,None]+offset[:,:,0]
        xx=torch.arange(w,device=x.device)[None,:]+offset[:,:,1]
        lookup=(y.clamp(0,h-1)*w+xx.clamp(0,w-1)).flatten()
        if return_offset:
            return lookup,offset
        return lookup

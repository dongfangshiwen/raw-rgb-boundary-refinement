"""Official torchvision DeepLabV3-MobileNetV3; binary crosswalk adaptation."""
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


METHODS=("deeplab_mobile_scratch","deeplab_mobile_imagenet")


class DirectVisionSegmenter(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        self.network=deeplabv3_mobilenet_v3_large(weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None,
            num_classes=2,aux_loss=False,progress=False)
        self.register_buffer("mean",torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer("std",torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,image):
        return image,self.network((image-self.mean)/self.std)["out"],{}


def build_model(method,seed):
    if method not in METHODS:raise ValueError("Unknown vision control")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+1_000_000)
        return DirectVisionSegmenter(pretrained=method=="deeplab_mobile_imagenet")


def load_model(path,device="cuda"):
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if packet["format"]!="torchvision-crosswalk-v1":raise ValueError("Wrong vision baseline checkpoint")
    model=DirectVisionSegmenter(pretrained=False)
    model.load_state_dict(packet["model_state"],strict=True)
    return model.to(device).eval(),packet

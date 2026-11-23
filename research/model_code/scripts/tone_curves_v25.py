"""Author-style spatially varying color curves and paired supervision controls."""
import torch
from torch import nn
import torch.nn.functional as F


KNOTS = 8


def sample_curves(image, curves):
    """Evaluate the three input-channel curve contributions on the original grid."""
    b, _, h, w = image.shape
    vertical = torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype)
    horizontal = torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype)
    y, x = torch.meshgrid(vertical, horizontal, indexing="ij")
    xy = torch.stack((x, y), dim=-1)[None].expand(b, h, w, 2)
    results = []
    for c in range(3):
        grid = torch.cat((xy, image[:, c, :, :, None].detach()), dim=-1)[:, None]
        source = curves[:, :, c].contiguous()
        value = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
        results.append(value.squeeze(2))
    return torch.stack(results).sum(0)


def positive_diagonal(raw):
    """Rectify each own-channel increment, holding the curve's average fixed."""
    output = []
    for c in range(3):
        own = raw[:, c, c]
        steps = own.diff(dim=1).clamp_min(0)
        monotone = torch.cat((own[:, :1], own[:, :1]+steps.cumsum(dim=1)), dim=1)
        centered = monotone + (own-monotone).mean(dim=1, keepdim=True)
        output.append(torch.stack([centered if j==c else raw[:, c, j] for j in range(3)], dim=1))
    return torch.stack(output, dim=1)


def replace_affine_head(network):
    """Initialize the diagonal curves to the original gain and offset exactly."""
    head = network.patch_unembed.proj[0]
    if head.out_channels != 4 or network.patch_unembed.proj[1].upscale_factor != 1:
        raise ValueError("Expected the author's final four-channel affine projection")
    curves = nn.Conv2d(head.in_channels, 9*KNOTS, head.kernel_size, stride=head.stride,
                       padding=head.padding, padding_mode=head.padding_mode,
                       dilation=head.dilation, bias=True, device=head.weight.device)
    with torch.no_grad():
        curves.weight.zero_()
        curves.bias.zero_()
        knots = torch.linspace(-1,1,KNOTS,device=head.weight.device)
        for c in range(3):
            first = (3*c+c)*KNOTS
            curves.weight[first:first+KNOTS].copy_(knots[:,None,None,None]*head.weight[0] - head.weight[c+1])
            curves.bias[first:first+KNOTS].copy_(knots*(1+head.bias[0]) - head.bias[c+1])
    network.patch_unembed.proj[0] = curves
    network.patch_unembed.out_chans = 9*KNOTS


class CurveRestoration(nn.Module):
    def __init__(self, backbone, monotone=False):
        super().__init__()
        self.backbone = backbone
        self.monotone = monotone
        replace_affine_head(backbone)

    @staticmethod
    def context_size(image):
        h, w = image.shape[-2:]
        ratio = min(.5, 256/max(h,w))
        return max(16,round(h*ratio)), max(16,round(w*ratio))

    def coefficients(self, source):
        side = self.context_size(source)
        middle = F.interpolate(source, side, mode="area")
        features = self.backbone.forward_features(self.backbone.check_image_size(middle)).float()
        raw = features[..., :side[0], :side[1]].unflatten(1, (3,3,KNOTS))
        return positive_diagonal(raw) if self.monotone else raw

    def forward(self, image):
        source = 2*image-1
        curves = self.coefficients(source)
        return .5*(sample_curves(source, curves)+1)


def affine_rgb(backbone, image):
    source = 2*image-1
    height, width = CurveRestoration.context_size(source)
    small = F.interpolate(source, (height, width), mode="area")
    fields = backbone.forward_features(backbone.check_image_size(small)).float()[..., :height, :width]
    full = F.interpolate(fields, source.shape[-2:], mode="bilinear", align_corners=True)
    return .5*((1+full[:, :1])*source-full[:, 1:]+1)


def edge_loss(hazy, restored, clean, valid, weighted):
    """Match paired gradients, with an option to favor visible reference edges."""
    costs = []
    for axis in (-1,-2):
        observed = hazy.diff(dim=axis)
        reference = clean.diff(dim=axis)
        prediction = restored.diff(dim=axis)
        n = valid.shape[axis]-1
        weights = (valid.narrow(axis,0,n)&valid.narrow(axis,1,n)).float()
        if weighted:
            obs = observed.square().sum(1).clamp_min(1e-8).sqrt()
            ref = reference.square().sum(1).clamp_min(1e-8).sqrt()
            direction = ((observed*reference).sum(1)/(obs*ref)).clamp(0,1)
            strength = ((obs+.001)/(ref+.001)).clamp(0,1).sqrt()
            visibility = (.15+.85*direction*strength)*(ref/.06).clamp(.2,1)
            weights = weights*visibility
        costs.append(((prediction-reference).abs().mean(1)*weights).sum()/weights.sum().clamp_min(1))
    return sum(costs)/len(costs)

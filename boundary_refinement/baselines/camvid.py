"""CamVid11 transfer of the released Dehaze-Seg-S architecture.

Author backbone and frequency neck are used through the reviewed V12 adapter.
The library supplies the Mask2Former head under disclosed framework, padding,
class count and recipe changes. No UAV checkpoints, pickle or install scripts.
"""
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.baselines.published_runtime import AuthorFrequencyNeck, dehazeseg_backbone


class CamVidQueryHead(nn.Module):
    def __init__(self, classes=11):
        super().__init__()
        from transformers import Mask2FormerConfig
        from transformers.models.mask2former.modeling_mask2former import (
            Mask2FormerPixelDecoder, Mask2FormerPreTrainedModel, Mask2FormerTransformerModule,
        )

        class Head(Mask2FormerPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.pixel_decoder = Mask2FormerPixelDecoder(config, [192, 384, 768, 768])
                self.query_decoder = Mask2FormerTransformerModule(config.feature_size, config)
                self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
                self.post_init()

        # The library counts its initial queries in decoder_layers. Ten entries
        # means nine transformer blocks, matching the author's config.
        self.head = Head(Mask2FormerConfig(num_labels=classes, use_auxiliary_loss=False))

    def forward(self, features):
        pixel = self.head.pixel_decoder(features, return_dict=True)
        result = self.head.query_decoder(pixel.multi_scale_features, pixel.mask_features)
        classes = self.head.class_predictor(result.intermediate_hidden_states[-1].transpose(0, 1))
        return classes.float(), result.masks_queries_logits[-1].float()


def query_to_scores(classes, masks, padded_shape, output_shape):
    """Author upsample-then-sigmoid aggregation, chunked along the query axis."""
    height, width = output_shape
    with torch.autocast(classes.device.type, enabled=False):
        probabilities = classes.float().softmax(-1)[..., :-1]
        result = None
        for label_weight, mask in zip(probabilities.split(20, 1), masks.float().split(20, 1)):
            value = F.interpolate(mask, padded_shape, mode='bilinear', align_corners=False)
            value = value[..., :height, :width].sigmoid()
            term = torch.einsum('bqc,bqhw->bchw', label_weight, value)
            result = term if result is None else result + term
        # Log preserves argmax and turns normalized semantic sums into the
        # logits expected by the shared CamVid pixel CE + Dice implementation.
        return result.clamp_min(1e-6).log()


def rectangular_dct_blocks(value, block=16):
    """Released global DCT then block-major flatten, including partial blocks.

    For shapes divisible by `block` this is the author's stack(...).view. At
    384 input the /16 feature is 24x24: padding the whole image to 512 would
    add more artificial pixels, so retain the original coefficient count.
    """
    import torch_dct
    with torch.autocast(value.device.type, enabled=False):
        h, w = value.shape[-2:]
        coeff = torch_dct.dct_2d(value.float(), norm='ortho')
        flat = [coeff[..., y:y + block, x:x + block].flatten(-2)
                for y in range(0, h, block) for x in range(0, w, block)]
        return torch.cat(flat, dim=-1).view(value.shape)


class PublishedCamVid11(nn.Module):
    classes = 11

    def __init__(self, seed):
        super().__init__()
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 17_100_000)
            self.backbone = dehazeseg_backbone()
            self.freq_neck = AuthorFrequencyNeck()
            self.freq_neck.original.semantic_enhance.dct_processor = rectangular_dct_blocks
            self.seg_head = CamVidQueryHead(self.classes)

    def forward(self, image):
        h, w = image.shape[-2:]
        # /8 stages have windows of 16. Rectangular DCT blocks are handled above.
        # CAM288x384 therefore uses 384x384 internally. Crop to shared target.
        padded = F.pad(image, (0, (-w) % 128, 0, (-h) % 128), mode='replicate')
        raw, *features = self.backbone((padded - self.mean) / self.std)
        cls, mask = self.seg_head(self.freq_neck(features))
        rgb = (raw.float() * self.std + self.mean)[..., :h, :w].clamp(0, 1)
        return rgb, query_to_scores(cls, mask, padded.shape[-2:], (h, w)), {}

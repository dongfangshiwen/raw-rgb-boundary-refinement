"""Inspected author architectures with explicit framework/resolution adapters.

Author files remain untouched. These adapters are not the authors' training
recipes, and must not be described as published benchmark scores.
"""
from boundary_refinement.paths import WORKSPACE
from pathlib import Path
import types

import torch
from torch import nn
import torch.nn.functional as F

SOURCE=(WORKSPACE / "benchmark_data" / 'published-models-v12')


class Registry:
    def register_module(self,*args,**kwargs):return lambda cls:cls


class BaseModule(nn.Module):
    def __init__(self,init_cfg=None,**kwargs):super().__init__()


def author_module(name):
    path=SOURCE/name;source=path.read_text(encoding='utf-8-sig')
    for line in ('from mmengine.model import BaseModule','from mmseg.models.builder import BACKBONES',
                 'from mmseg.models.builder import NECKS','from timm.models.layers import to_2tuple, trunc_normal_'):
        source=source.replace(line,'# Framework import supplied by inspected V12 adapter')
    namespace=dict(__name__='v12_author_'+path.parent.name.replace('-','_'),BaseModule=BaseModule,
        BACKBONES=Registry(),NECKS=Registry(),to_2tuple=lambda x:(x,x) if isinstance(x,int) else x,
        trunc_normal_=nn.init.trunc_normal_)
    exec(compile(source,str(path),'exec'),namespace)
    return types.SimpleNamespace(**namespace)


def dehazeformer_t():return author_module('author-dehazeformer/dehazeformer.py').dehazeformer_t()


def dehazeseg_backbone():return author_module('author-dehaze-seg/dehazeformer.py').DehazeFormerS()


def efficient_window_attention(self,x):
    """Same author algebra/layout, using PyTorch's fused attention primitive."""
    b,c,h,w=x.shape;ws=self.window_size
    x=x.view(b,c,h//ws,ws,w//ws,ws).permute(0,2,4,3,5,1).contiguous().view(-1,ws*ws,c)
    qkv=self.qkv(x).reshape(-1,ws*ws,3,self.num_heads,self.head_dim)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
    bias=self.relative_position_bias_table[self.relative_position_index.view(-1)].view(ws*ws,ws*ws,-1).permute(2,0,1)
    # Released head_dim=12 prevents the CUDA memory-efficient kernel. Zero
    # padding to16 leaves dot products/values unchanged; keep original scale.
    width=q.shape[-1];padding=(-width)%8
    if padding:q,k,v=(F.pad(z,(0,padding)) for z in (q,k,v))
    x=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype).contiguous()[None],dropout_p=0.,scale=self.scale)[...,:width].transpose(1,2)
    x=x.reshape(-1,ws,ws,c).view(b,h//ws,w//ws,ws,ws,c)
    # Deliberately retain the released author's merge/view layout.
    x=x.permute(0,5,1,3,2,4).contiguous().view(b,h,w,c)
    return self.proj(x).view(b,c,h,w)


class AuthorFrequencyNeck(nn.Module):
    """Original LFM/IFM/SEM; interpolate positional grids for 256 inputs.

    Preserve the author's attention layout and wavelet/DCT operations. DCT runs
    in FP32 because CUDA half FFT does not support arbitrary spatial sizes.
    """
    def __init__(self):
        super().__init__();module=author_module('author-dehaze-seg/dehaze_seg_neck.py')
        module.WindowSelfAttention.forward=efficient_window_attention
        original_dct=module.DCT2D_Processor.__call__
        def fft_float32(processor,x):
            with torch.autocast('cuda',enabled=False):return original_dct(processor,x.float())
        module.DCT2D_Processor.__call__=fft_float32
        self.original=module.DehazeFreqFusionNeck([24,48,96,48,24],[192,384,768,768])

    def forward(self,features):
        x=features;m=self.original
        # Original code has fixed 512-input positional and layer-norm sizes.
        # All learned positional tensors are retained and interpolated.
        a=m.fusion_1_4(x[0],x[4]);b=m.fusion_1_8(x[1],x[3]);c,d=m.semantic_enhance(x[2])
        values=[]
        for key,value in zip(('1_4','1_8','1_16','1_32'),(a,b,c,d)):
            position=F.interpolate(m.pos_embed[key],value.shape[-2:],mode='bilinear',align_corners=False)
            value=value+position;values.append(F.layer_norm(value,value.shape[1:]))
        return values


class Mask2FormerHead(nn.Module):
    """Library Mask2Former pixel/transformer decoders, full default dimensions.

    The common partial semantic loss replaces set/Hungarian supervision, which
    cannot treat unknown own pixels as background. No segmentation checkpoint
    or additional labelled dataset is imported.
    """
    def __init__(self,channels):
        super().__init__()
        from transformers import Mask2FormerConfig
        from transformers.models.mask2former.modeling_mask2former import (
            Mask2FormerPreTrainedModel,Mask2FormerPixelDecoder,Mask2FormerTransformerModule)
        config=Mask2FormerConfig(num_labels=2)
        class InitializedHead(Mask2FormerPreTrainedModel):
            def __init__(self,cfg):
                super().__init__(cfg);self.pixel_decoder=Mask2FormerPixelDecoder(cfg,channels)
                self.transformer=Mask2FormerTransformerModule(cfg.feature_size,cfg)
                self.classifier=nn.Linear(cfg.hidden_dim,3);self.post_init()
        self.model=InitializedHead(config)

    def forward(self,features):
        pixel=self.model.pixel_decoder(features,return_dict=True)
        result=self.model.transformer(pixel.multi_scale_features,pixel.mask_features)
        values=[]
        for hidden,mask in zip(result.intermediate_hidden_states,result.masks_queries_logits):
            cls=self.model.classifier(hidden.transpose(0,1)).float().softmax(-1)[...,:2]
            if cls.shape[1]!=mask.shape[1]:raise ValueError('Mask2Former query dimensions mismatch')
            values.append(torch.einsum('bqc,bqhw->bchw',cls,mask.float().sigmoid()).clamp_min(1e-6).log())
        return values[-1],values[:-1]

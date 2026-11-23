"""Keep MCAF-Net cross-attention global while avoiding its discarded dense attention map."""
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def attention(head, query, key, value):
    """Attention with the original projection, heads, and logit scale."""
    weights = head.in_proj_weight.chunk(3)
    biases = head.in_proj_bias.chunk(3)
    per_head = head.head_dim
    shape = head.num_heads
    padding = (-per_head) % 8
    q, k, v = (F.pad(F.linear(x, w, b).transpose(0,1).unflatten(-1, (shape, per_head)).transpose(1,2), (0,padding))
               for x,w,b in zip((query,key,value),weights,biases,strict=True))
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        response = F.scaled_dot_product_attention(q.contiguous(),k.contiguous(),v.contiguous(),scale=per_head**-.5)
    flat = response[...,:per_head].transpose(1,2).flatten(-2).transpose(0,1)
    return head.out_proj(flat), None


def memory_efficient(model):
    if model.training:
        raise ValueError("This adapter requires the frozen author-architecture mode")
    for head in (model.fusion1.cross_attn, model.fusion2.cross_attn):
        if head.batch_first or head.in_proj_bias is None or head.bias_k is not None or head.bias_v is not None:
            raise ValueError("The published cross-attention layout changed")
        head.forward = MethodType(attention,head)
    return model


@torch.inference_mode()
def check():
    from statistics import median
    from time import perf_counter
    from scripts.mcaf_reference_v59 import trained

    model = next(iter(trained().values()))
    torch.manual_seed(59)
    lite = torch.rand(1,3,40,52,device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        original = model(lite)[0].float()
        memory_efficient(model)
        reduced = model(lite)[0].float()
        torch.testing.assert_close(original,reduced,rtol=.055,atol=.018)
        full = torch.rand(1,3,256,256,device="cuda")
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            start = perf_counter()
            result = model(full)[0]
            torch.cuda.synchronize()
            times.append((perf_counter()-start)*1000)
    assert result.shape==full.shape and bool(result.isfinite().all())
    print(dict(maximum_bfloat16_difference=float((original-reduced).abs().max()),
               typical_ms=median(times[1:]),cuda_GiB=torch.cuda.max_memory_allocated()/2**30),flush=True)


if __name__=="__main__":
    check()

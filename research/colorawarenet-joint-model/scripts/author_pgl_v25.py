"""Load the exact official PGL-Net-T implementation and RW2AH checkpoint."""
from importlib.metadata import version
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

from scripts.projective_ledger_v12 import REPO, digest


STORE = REPO.parent/"benchmark_data/pgl-net-v25"
SOURCE_SHA256 = "57f389376de1aa89474f57bc6b4f4e8c50feeff18e6c2161d25da580509f5c10"
WEIGHTS_SHA256 = "1ea04c0d993cc2e5aa6a36a6c77f5ac7719d4faa1fede551c3bc8db23f89526e"


def load_pgl():
    archive, weights = STORE/"author.zip", STORE/"rw2ah_pglnet_t.pk"
    if digest(archive)!=SOURCE_SHA256 or digest(weights)!=WEIGHTS_SHA256:
        raise ValueError("Official PGL code or checkpoint changed")
    member = "PGL-Net-main/models/PGLNet.py"
    with ZipFile(archive) as source:
        module = {"__name__": "official_pglnet_v25"}
        exec(compile(source.read(member),member,"exec"),module)
    # NumPy scalar metadata accompanies the author's model tensors.
    allowed = [np.dtype, (np._core.multiarray.scalar,"numpy.core.multiarray.scalar"),
               np.dtypes.Float64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(weights,map_location="cpu",weights_only=True)
    model = module["pglnet_t"]()
    model.load_state_dict({k.removeprefix("module."):v for k,v in checkpoint["model"].items()},strict=True)
    registration = dict(paper="https://arxiv.org/abs/2606.25732",source="https://github.com/sc-30-bit/PGL-Net",
                        weights="https://huggingface.co/klay11/PGL-Net",source_SHA256=SOURCE_SHA256,
                        checkpoint_SHA256=WEIGHTS_SHA256,author_step=int(checkpoint["step"]),timm=version("timm"),
                        params=sum(p.numel() for p in model.parameters()),
                        scope="Author RW2AH checkpoint with official full-data configuration and evaluation on the original RW2AH test set. Training and selection budget differs from the controlled arms.")
    return model.cuda().eval(), registration


def pgl_rgb(model, image):
    height, width = image.shape[-2:]
    # Author test.py uses normalized-by-255 RGB, then reflection pads to 32.
    padded = F.pad(image,(0,(-width)%32,0,(-height)%32),mode="reflect")
    return model(padded)[...,:height,:width]

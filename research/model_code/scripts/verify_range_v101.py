"""Independent integrity, algorithm and deployment checks for the follow-up study."""
import json
from hashlib import file_digest
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

from scripts import subpixel_transport_v92 as legacy
from scripts import axis_range_v101 as ax
from scripts.fit_range_v101 import OUT
from scripts.range_decoder_v101 import FAMILIES, RangeDecoder,prepare

def test():
    torch.manual_seed(712)
    z=torch.randn(1,19,7,9,device="cuda")
    rgb=torch.randint(0,256,(1,3,28,36),dtype=torch.uint8,device="cuda")
    st=prepare(z,rgb)
    for f in FAMILIES:
        net=RangeDecoder(f).cuda()
        p=net.plan(st)
        torch.testing.assert_close(p.sum(-1),torch.ones_like(p[...,0]),rtol=1e-6,atol=1e-6)
        assert (p>=0).all() and torch.isfinite(p).all()
        if f.startswith("bilinear"):
            assert (p[:,st[1]==0]==0).all()
        val=net.correction(z,st).square().mean();val.backward()
        assert all(torch.isfinite(v.grad).all() for v in net.parameters())
    old=legacy.ColorTransport().cuda()
    prev=legacy.prediction("dual_free",old,z,legacy.guide_stencil(rgb[[0,0]],z.argmax(1)[0]))
    new=RangeDecoder("bilinear_bounded4").cuda().correction(z,st)
    torch.testing.assert_close(prev,new,atol=2.e-6,rtol=1.e-5)
    flat=torch.zeros_like(z);empty=prepare(flat,rgb)
    assert empty[0].numel()==0 and RangeDecoder("bilinear_bounded4").cuda().correction(flat,empty).abs().max()==0
    flatguide=torch.full_like(rgb,118)
    flatst=prepare(z,flatguide)
    axes=ax.basis(z,flatguide,flatst)
    for v in axes.values():
        assert (v==0).all()
    assert len(np.unique([r["sampler_end"] for r in legacy.read(OUT/"training.json")["runs"]]))==3
    for stem,expected in (("bdd",128),("bddval",128),("fogreserve",4)):
        rows=legacy.read(OUT/f"{stem}-manifest.json")["rows"]
        assert len(rows)==expected
    b=legacy.read(OUT/"bdd-manifest.json")["rows"]
    v=legacy.read(OUT/"bddval-manifest.json")["rows"]
    f=legacy.read(OUT/"fogreserve-manifest.json")["rows"]
    groups=[{r["id"].split("-")[0] for r in items} for items in (b,v,f)]
    assert [len(x) for x in groups]==[126,128,4]
    assert all(groups[i].isdisjoint(groups[j]) for i,j in ((0,1),(0,2),(1,2)))
    assert not {r["rgb_sha"] for r in v+f}&{r["rgb_sha"] for r in b}
    photo=legacy.read(OUT/"photo-weights.json")
    for city,fold in photo["folds"].items():
        assert city not in photo["models"][f"{fold}/scalar"]["training_cities"]
        assert city not in photo["models"][f"{fold}/conditional"]["training_cities"]
    for t in ("mask2former","segformer"):
        a=np.asarray(Image.open(OUT/f"prediction-{t}-bdd-candidate.png"))
        bmask=np.asarray(Image.open(OUT/f"example-{t}/mask_train_ids.png"))
        np.testing.assert_array_equal(a,bmask)
        assert bmask.shape==(720,1280)
    assert (OUT/"example-mask2former/restored.png").read_bytes()==(OUT/"example-segformer/restored.png").read_bytes()
    sys.path.insert(0,str(OUT/"portable"))
    from predict import predict
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp);im=d/"odd.png"
        arr=np.random.default_rng(701).integers(0,256,(99,151,3),dtype=np.uint8)
        Image.fromarray(arr).save(im)
        record=predict(im,d/"tested",OUT/"portable/assets")
        assert record["shape"]==[99,151]
        ids=np.asarray(Image.open(d/"tested/mask_train_ids.png"))
        assert ids.shape==(99,151) and ids.max()<19
    legacy.atomic_write(OUT/"checks.json",dict(protocol=legacy.digest(OUT/"protocol.json"),CUDA=str(torch.__version__),
        verified=["simplex and finite gradients for 6 families","zero bilinear support","legacy/fixed 4-range equivalence",
        "empty transitions", "flat-guide axis degeneracy", "matched seed sampler receipts", "disjoint prefix sets (126,128,4)",
        "no cross-cohort RGB hashes","all five BeDDE city splits for both gates","two portable mask PNGs pixel-identical to evaluation",
        "photographic output identical with either semantic backbone","odd-size dual-output inference"]))
    print("Algorithm, cohort and two-output checks passed",flush=True)
if __name__=="__main__":test()

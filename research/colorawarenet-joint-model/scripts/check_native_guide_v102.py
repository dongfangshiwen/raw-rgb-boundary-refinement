"""Verify geometry, the authors' offset convention and independent pixel counts."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.native_guide_v102 import PixelGrid,interpolate
from scripts.native_reserve_v102 import ROOT
from scripts.range_decoder_v101 import RangeDecoder
from scripts.segfix_peer_v102 import SOURCE,SegFix,ActivatedABN
from scripts.projective_ledger_v12 import REPO,digest,read
from scripts.train_regional_guidance_v18 import atomic_write,configure_cuda


def author_method(name,tree):
    node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
    return ast.Module(body=[node],type_ignores=[])


@torch.inference_mode()
def preflight():
    configure_cuda()
    torch.manual_seed(1202)
    z=torch.randn(1,19,7,11,device="cuda")
    guide=torch.randint(0,256,(1,3,28,44),device="cuda").float()/255
    grid=PixelGrid(z,guide,guide)
    decoder=RangeDecoder("bilinear_bounded4").cuda()
    ids=grid.active().nonzero().flatten()
    prior,cost,gaps,donors=grid.four(ids)
    expected=decoder(z,(guide*255).round().byte())[0].permute(1,2,0).reshape(-1,19)[ids]
    err=float((interpolate(prior,cost,donors)-expected).abs().max())
    assert err<2e-5
    constant=torch.full((1,3,51,73),.4,device="cuda")
    co=PixelGrid(z,torch.full_like(guide,.4),constant)
    ids2=torch.arange(51*73,device="cuda")
    pr,c,_,v=co.four(ids2)
    target=F.interpolate(z,(51,73),mode="bilinear",align_corners=False)[0].permute(1,2,0).reshape(-1,19)
    err2=float((target-interpolate(pr,c,v)).abs().max());assert err2<2e-5
    assert not PixelGrid(torch.ones_like(z),guide,constant).active().any()
    sf=SegFix()
    from lib.utils.helpers.offset_helper import DTOffsetHelper
    from lib.datasets.tools.transforms import Normalize,ToTensor
    rgb=np.random.default_rng(324).integers(0,256,(96,128,3),dtype=np.uint8)
    x=Normalize(255,[.485,.456,.406],[.229,.224,.225])(ToTensor()(rgb[:,:,::-1].copy())).cuda()[None]
    mask,dr=sf.model(x)
    prob=F.interpolate(mask.softmax(1),rgb.shape[:2],mode="bilinear",align_corners=True)
    dr=F.interpolate(dr,rgb.shape[:2],mode="bilinear",align_corners=True)
    tree=ast.parse((SOURCE/"segmentor/tester_offset.py").read_text(encoding="utf8"))
    args={"torch":torch,"DTOffsetHelper":DTOffsetHelper}
    exec(compile(author_method("_get_offset",tree),"<authors' offset conversion>","exec"),args)
    expected_off=args["_get_offset"](None,prob,dr)[0]*2
    lookup,off=sf.lookup(rgb,True)
    torch.testing.assert_close(off,expected_off,rtol=0,atol=0)
    # The source normalizes integer coordinates by (size-1)/2, i.e. pixel-center-aligned.
    ff=SimpleNamespace(grid_sample=lambda *a,**kw:F.grid_sample(*a,align_corners=True,**kw))
    tree=ast.parse((SOURCE/"scripts/cityscapes/segfix.py").read_text(encoding="utf8"))
    warp={"np":np,"torch":torch,"F":ff}
    for n in ("gen_coord_map","shift"):
        exec(compile(author_method(n,tree),"<authors' warp>","exec"),warp)
    pattern=np.random.default_rng(54).integers(0,19,rgb.shape[:2],dtype=np.uint8)
    reference=warp["shift"](pattern,off.permute(2,0,1).float().cpu().numpy())
    np.testing.assert_array_equal(pattern.ravel()[lookup.cpu().numpy()].reshape(pattern.shape),reference)
    bn=ActivatedABN(3).cuda().eval()
    bn.weight.copy_(torch.tensor([-.5,.1,-.9],device="cuda"))
    bn.bias.copy_(torch.tensor([.7,-.3,.2],device="cuda"))
    arr=(x-bn.running_mean[None,:,None,None])/(bn.running_var[None,:,None,None]+bn.eps).sqrt()
    explicit=F.leaky_relu(arr*(bn.weight.abs()+bn.eps)[None,:,None,None]+bn.bias[None,:,None,None],.01)
    assert torch.allclose(bn(x),explicit,atol=2e-6)
    atomic_write(ROOT/"preflight.json",dict(four_times_max_abs_error=err,constant_on_odd_grid_error=err2,
                author_offset_matches=True,author_two_pixel_warp_matches=True,abs_affine_bn_matches=True,
                segfix_code=digest(SOURCE/"lib/models/nets/segfix.py"),segfix_offset_code=digest(SOURCE/"segmentor/tester_offset.py")))
    print(read(ROOT/"preflight.json"),flush=True)


def saved_masks():
    from scipy.ndimage import maximum_filter
    from scripts.idd_fog_data_v100 import MAP
    manifest=read(ROOT/"manifest.json")["rows"]
    checked=0
    for cohort in ("bdd","fog"):
        r=next(r for r in manifest if r["cohort"]==cohort)
        gt=np.array(Image.open(REPO.parent/r["mask_path"]))
        if cohort=="fog":
            lut=np.full(256,255,dtype=np.uint8)
            for k,v in MAP.items():lut[k]=v
            gt=lut[gt]
        edge=np.zeros(gt.shape,dtype=bool)
        v=(gt[:-1]!=gt[1:])&(gt[:-1]<19)&(gt[1:]<19)
        h=(gt[:,:-1]!=gt[:,1:])&(gt[:,:-1]<19)&(gt[:,1:]<19)
        edge[:-1]|=v;edge[1:]|=v;edge[:,:-1]|=h;edge[:,1:]|=h
        band=maximum_filter(edge,size=13,mode="constant")
        with np.load(ROOT/"counts"/cohort/(r["id"].replace("/","_")+".npz")) as z:
            for j,target in enumerate(z["targets"].tolist()):
                for i,arm in enumerate(z["arms"].tolist()):
                    labels=np.array(Image.open(ROOT/"examples"/cohort/target/(arm+".png")))
                    for e,valid in enumerate((gt<19,(gt<19)&band)):
                        flat=gt[valid].astype(np.int32)*19+labels[valid]
                        cm=np.bincount(flat,minlength=361).reshape(19,19)
                        np.testing.assert_array_equal(cm,z["counts"][j,e,i,0])
                        checked+=1
    atomic_write(ROOT/"mask-audit.json",dict(independent_integer_confusions=checked))
    print("Independent native-mask confusion checks",checked,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("step",choices=("pre","post"))
    preflight() if cli.parse_args().step=="pre" else saved_masks()

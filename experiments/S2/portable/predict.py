"""Native semantic IDs and restored RGB for the follow-up two-output model."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, SegformerConfig, SegformerForSemanticSegmentation
from range_decoder_v101 import RangeDecoder, prepare

HERE=Path(__file__).resolve().parent
MCT_SHA="225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304"
MCT_SOURCE="56712bf3bb949d5c4c5a068d8a301b8057179c828a5a1acfb813af373e8bf207"
MASK_SHA="489e8dcd9b46f991ba0daa00a9dc3d780094168937dad3ff3a1c0dd884664754"
SEG_SHA="9a7c40538e41d92f93ad6501d54e6333833a1d96bda7b0906c7255240e927832"
PALETTE=np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],
                  [250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],
                  [0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],dtype=np.uint8)

def filehash(p):
    with p.open("rb") as f:return __import__("hashlib").file_digest(f,"sha256").hexdigest()

def resize(x,dims,antialias=False):
    return F.interpolate(x,dims,mode="bilinear",align_corners=False,antialias=antialias)

def shorter(x,long_side):
    h,w=x.shape[-2:];ratio=min(1.,long_side/max(h,w))
    return resize(x,(max(1,round(h*ratio)),max(1,round(w*ratio))),True)

def mct_from(directory):
    src=directory/"models/dehazeformer.py";weights=directory/"saved_models/dehazeformer.pth"
    assert filehash(src)==MCT_SOURCE and filehash(weights)==MCT_SHA
    spec=importlib.util.spec_from_file_location("official_mct",src)
    code=importlib.util.module_from_spec(spec);spec.loader.exec_module(code)
    class MCT(code.MCT):
        def get_coord(self,x):
            u,v=super().get_coord(x)
            return u.to(x.device),v.to(x.device)
    model=MCT().cuda().eval()
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True)["state_dict"])
    return model

def semantic_scores(q):
    probs=q.class_queries_logits.float().softmax(-1)[...,:-1]
    z=None
    for pr,mask in zip(probs.split(20,1),q.masks_queries_logits.float().split(20,1)):
        x=torch.einsum("bqc,bqhw->bchw",pr,mask.sigmoid())
        z=x if z is None else z+x
    return z.clamp_min(1.e-6).log()

def native(z,delta,img_shape,dims):
    h,w=img_shape;gh,gw=z.shape[-2:]
    if (4*gh,4*gw)==(h,w):return (resize(z,dims)+resize(delta,dims)).argmax(1)[0]
    # Partial border cells in odd images cover only the original image extent.
    y=(torch.arange(dims[0],device=z.device)+.5)*(2*h/(dims[0]*4*gh))-1
    x=(torch.arange(dims[1],device=z.device)+.5)*(2*w/(dims[1]*4*gw))-1
    yy,xx=torch.meshgrid(y,x,indexing="ij");grid=torch.stack((xx,yy),-1)[None]
    lift=lambda a:F.grid_sample(a,grid,mode="bilinear",padding_mode="border",align_corners=False)
    return (lift(z)+lift(delta)).argmax(1)[0]

@torch.inference_mode()
def predict(image_path,out,assets,backbone="mask2former",seed=101,photo_fold="all",decoder="fitted",fit_dir=None):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("CUDA BF16 GPU required")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with Image.open(image_path) as im:rgb=np.array(ImageOps.exif_transpose(im).convert("RGB"))
    tensor=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255.
    fit_dir = Path(fit_dir) if fit_dir is not None else HERE
    dims=rgb.shape[:2]
    raw=shorter(tensor,576);mct=mct_from(assets/"mct")
    base=(.5*(mct(2*raw-1).float()+1)).clamp(0,1)
    p=json.loads((fit_dir/"photo-weights.json").read_text(encoding="utf-8"))
    alpha=torch.sigmoid(raw.new_tensor(p["models"][f"{photo_fold}/scalar"]["bias"]))
    recovered=raw+alpha*(base-raw)
    folder=assets/backbone
    prep=json.loads((folder/"preprocessor_config.json").read_text(encoding="utf-8"))
    mean,std=(tensor.new_tensor(prep[k])[None,:,None,None] for k in ("image_mean","image_std"))
    x=shorter(tensor,512);h,w=x.shape[-2:];pad=(0,(-w)%32,0,(-h)%32)
    model_in=F.pad((x-mean)/std,pad)
    if backbone=="mask2former":
        assert filehash(folder/"model.safetensors")==MASK_SHA
        net=Mask2FormerForUniversalSegmentation.from_pretrained(str(folder),local_files_only=True,use_safetensors=True).cuda().eval()
        valid=F.pad(torch.ones((1,h,w),device="cuda",dtype=torch.long),pad)
        with torch.autocast("cuda",dtype=torch.bfloat16):q=net(pixel_values=model_in,pixel_mask=valid)
        z=semantic_scores(q)
    else:
        assert filehash(folder/"pytorch_model.bin")==SEG_SHA
        net=SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(str(folder),local_files_only=True)).cuda().eval()
        net.load_state_dict(torch.load(folder/"pytorch_model.bin",map_location="cpu",weights_only=True))
        with torch.autocast("cuda",dtype=torch.bfloat16):z=net(pixel_values=model_in).logits.float()
    z=z[...,:((h+3)//4),:((w+3)//4)].contiguous()
    guide=F.pad(x,(0,(-w)%4,0,(-h)%4),mode="replicate").mul(255).round().clamp(0,255).byte()
    model=RangeDecoder("bilinear_bounded4").cuda()
    if decoder=="fitted":
        model.load_state_dict(torch.load(fit_dir/f"bilinear_bounded4-{seed}.pth",weights_only=True))
    elif decoder == "original":
        profile_file = HERE.parents[1] / "S3" / "design" / "fixed-evaluation.json"
        profile = json.loads(profile_file.read_text(encoding="utf-8"))["settings"][str(seed)]
        model.mix.copy_(torch.logit(z.new_tensor(profile["colour"]) / 6))
        model.spatial.copy_(torch.logit(z.new_tensor((profile["exponent"] - .25) / 1.5)))
    elif decoder != "fixed":
        raise ValueError(f"Unknown decoder {decoder}")
    mask=native(z,model.correction(z,prepare(z,guide)),(h,w),dims).byte().cpu().numpy()
    photo=resize(recovered,dims)[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
    out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask).save(out/"mask_train_ids.png");Image.fromarray(PALETTE[mask]).save(out/"mask_color.png")
    Image.fromarray(photo).save(out/"restored.png")
    manifest=dict(model="follow-up parallel RGB and semantic outputs",input_sha256=filehash(image_path),shape=list(dims),
        backbone=backbone,semantic_decoder=decoder,source_seed=seed,RGB_fold=photo_fold,RGB_MCT_fraction=float(alpha),
        RGB_size=list(raw.shape[-2:]),semantic_size=[h,w],torch=str(torch.__version__),cuda=torch.version.cuda,
        output_sha256={k:filehash(out/k) for k in ("mask_train_ids.png","mask_color.png","restored.png")})
    (out/"prediction.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image",type=Path);p.add_argument("output",type=Path);p.add_argument("--assets",type=Path,default=HERE/"assets")
    p.add_argument("--backbone",choices=["mask2former","segformer"],default="mask2former")
    p.add_argument("--seed",type=int,choices=[101,211,307],default=101);p.add_argument("--decoder",choices=["original","fitted","fixed"],default="fitted")
    p.add_argument("--photo-fold",choices=["all","0","1","2","3","4"],default="all")
    p.add_argument("--fit-dir", type=Path, default=HERE,
                   help="S2/portable directory containing the authors' fitted heads and photographic blend")
    args = p.parse_args()
    print(json.dumps(predict(args.image, args.output, args.assets, args.backbone,
                             args.seed, args.photo_fold, args.decoder, args.fit_dir)))

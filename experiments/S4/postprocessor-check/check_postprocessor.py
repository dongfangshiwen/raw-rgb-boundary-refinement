"""Audit the Mask2Former postprocessor with fixed model inputs on the original IDD-AW fog cohort."""
import argparse, hashlib, json, os, sys
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np

LABEL_MAP = {0:0,2:1,4:11,5:12,6:17,7:18,9:13,10:14,11:15,14:3,15:4,18:7,19:6,20:5,22:2,24:8,25:10}
SEEDS = [101,211,307]
def h(x):return hashlib.sha256(x).hexdigest()
def load_s1(path):
    with ZipFile(path) as z:
        rows=json.loads(z.read("idd/frames.json"))["fog"]
        with np.load(BytesIO(z.read("idd/fog-counts.npz")),allow_pickle=False) as f:
            assert [r["id"] for r in rows]==f["ids"].tolist() and [r["group"] for r in rows]==f["groups"].tolist()
            assert f["seeds"].tolist()==SEEDS
            return rows,f["counts"][:,:,f["arms"].tolist().index("raw")],f["counts"][:,:,f["arms"].tolist().index("author"),0]
def scoring(y):
    from scipy.ndimage import maximum_filter
    valid=y<19;e=np.zeros(y.shape,dtype=bool)
    a=valid[:,:-1]&valid[:,1:]&(y[:,:-1]!=y[:,1:]);b=valid[:-1]&valid[1:]&(y[:-1]!=y[1:])
    e[:,:-1]|=a;e[:,1:]|=a;e[:-1]|=b;e[1:]|=b
    e=maximum_filter(e,size=13,mode="constant",cval=0)
    keys=(19*e[valid].astype(np.int32)+y[valid])*19
    def cm(v):
        v=v[valid];n=np.bincount(keys+v,minlength=2*19*19).reshape(2,19,19)
        return np.stack([n.sum(0),n[1]])
    return cm
def stats(rows,raw,logged,api):
    groups=np.array([r["group"] for r in rows]);unique=np.unique(groups)
    assert len(rows)==146 and len(unique)==24
    arms=np.stack([raw,np.repeat(logged[:,:,None],3,axis=2),np.repeat(api[:,:,None],3,axis=2)],axis=2)
    assert np.all(arms.sum(-1)==arms[:,:,:1,:1].sum(-1))
    vals=np.stack([arms[groups==g].sum(0) for g in unique]);cl=sorted(LABEL_MAP.values()); hit=vals.diagonal(axis1=-2,axis2=-1);union=vals.sum(-1)+vals.sum(-2)-hit
    sufficient=np.stack((hit[...,cl],union[...,cl]),axis=-1)
    def score(n):
        count=np.tensordot(n,sufficient,axes=([-1],[0]));x,y=count[...,0],count[...,1]
        return 100*np.divide(x,y,out=np.zeros_like(x,dtype=float),where=y>0).mean((-1,-2))
    n=len(unique); point=score(np.ones(n)); draws=np.random.default_rng(100481).multinomial(n,np.full(n,1/n),size=5000);rep=np.concatenate([score(a) for a in np.array_split(draws,80)])
    out={}
    for k,reg in enumerate(["whole","boundary"]):
        parts=dict(zip(["guided","study_reference","library_default"],point[k].tolist()))
        for a,b,key in [(0,1,"guided_minus_study"),(0,2,"guided_minus_default"),(2,1,"default_minus_study")]:
            parts[key]=[float(point[k,a]-point[k,b]),*np.quantile(rep[:,k,a]-rep[:,k,b],[.025,.975]).tolist()]
        out[reg]=parts
    return out
def evaluate(s1,data,assets,folder):
    from PIL import Image
    import torch, torch.nn.functional as F, transformers
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    assert transformers.__version__=="4.57.1" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert h((assets/"model.safetensors").read_bytes())=="489e8dcd9b46f991ba0daa00a9dc3d780094168937dad3ff3a1c0dd884664754"
    rows,guided,study=load_s1(s1)
    folder.mkdir(parents=True,exist_ok=True)
    pr=Mask2FormerImageProcessor.from_pretrained(str(assets),local_files_only=True)
    net=Mask2FormerForUniversalSegmentation.from_pretrained(str(assets),local_files_only=True,use_safetensors=True).cuda().eval().requires_grad_(False)
    mu,sd=[torch.tensor(getattr(pr,k),device="cuda")[None,:,None,None] for k in ("image_mean","image_std")]
    lut=np.full(256,255,dtype=np.uint8)
    for a,b in LABEL_MAP.items():lut[a]=b
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cm=[];redo=[];delta=[];agree=[]
    with torch.inference_mode():
      for i,r in enumerate(rows):
        p=data/r["rgb"];m=data/r["mask"]
        assert h(p.read_bytes())==r["rgb_sha"] and h(m.read_bytes())==r["mask_sha"]
        with Image.open(m) as im:y=lut[np.array(im)]
        with Image.open(p) as im:x=np.array(im.convert("RGB"))
        ih,iw=y.shape; assert x.shape[:2]==(ih,iw)
        shape=tuple(round(a*min(1.,512/max(ih,iw))) for a in (ih,iw));assert all(a%32==0 for a in shape)
        x=torch.from_numpy(x).cuda().permute(2,0,1)[None].float()/255.
        x=F.interpolate(x,shape,mode="bilinear",align_corners=False,antialias=True)
        valid=torch.ones((1,*shape),dtype=torch.long,device="cuda")
        with torch.autocast("cuda",dtype=torch.bfloat16): pred=net(pixel_values=(x-mu)/sd,pixel_mask=valid)
        # Use the library postprocessor's default geometry, with the same FP32
        # class and mask logits supplied to both postprocessors.
        pred.class_queries_logits=pred.class_queries_logits.float();pred.masks_queries_logits=pred.masks_queries_logits.float()
        cls=pred.class_queries_logits.softmax(-1)[...,:-1];masks=pred.masks_queries_logits
        scores=None
        for cw,mk in zip(cls.split(20,1),masks.split(20,1)):
            local=torch.einsum("bqc,bqhw->bchw",cw,F.interpolate(mk,shape,mode="bilinear",align_corners=False).sigmoid())
            scores=local if scores is None else scores+local
        logged=F.interpolate(scores.clamp_min(1e-6).log(),(ih,iw),mode="bilinear",align_corners=False).argmax(1)[0].byte().cpu().numpy()
        api=pr.post_process_semantic_segmentation(pred,target_sizes=[(ih,iw)])[0].byte().cpu().numpy()
        compute=scoring(y);ca=compute(api);cr=compute(logged);cm.append(ca);redo.append(cr)
        agree.append(bool(np.array_equal(cr,study[i])))
        delta.append(int(np.count_nonzero((api!=logged)&(y<19))))
        if (i+1)%24==0:print(f"{i+1}/{len(rows)}; historical reference exact for {sum(agree)}/{len(agree)}",flush=True)
    meta=dict(torch=str(torch.__version__),transformers=transformers.__version__,GPU=torch.cuda.get_device_name(),source_s1=h(s1.read_bytes()),library="Mask2FormerImageProcessor.post_process_semantic_segmentation",numeric_precision="BF16 model forward; FP32 postprocessing; label grid 2048 x 1536",preprocessing="identical cached study protocol: 512-long-edge, no target refitting or input-size change",exact_historical_replays=sum(agree),frames=len(rows),drives=len(set(r["group"] for r in rows)),valid_pixels_changed=sum(delta),valid_pixels=int(study[:,0].sum()))
    np.savez_compressed(folder/"postprocess-counts.npz",ids=np.asarray([r["id"] for r in rows]),groups=np.asarray([r["group"] for r in rows]),library=np.stack(cm),rerun=np.stack(redo),per_image_changed=np.asarray(delta))
    (folder/"run.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--s1",required=True,type=Path);ap.add_argument("--data",type=Path);ap.add_argument("--assets",type=Path);ap.add_argument("--out",type=Path,default=Path(__file__).resolve().parent);ap.add_argument("--replay",action="store_true");a=ap.parse_args()
    if not a.replay:
        if not a.data or not a.assets:ap.error("CUDA inference requires --data and --assets")
        print(evaluate(a.s1,a.data,a.assets,a.out),flush=True)
    rows,raw,old=load_s1(a.s1)
    with np.load(a.out/"postprocess-counts.npz",allow_pickle=False) as t:
        assert [r["id"] for r in rows]==t["ids"].tolist()
        assert np.array_equal(t["rerun"],old),"Native study reference does not replicate bit-for-bit; investigate before using head-to-head contrasts."
        out=stats(rows,raw,old,t["library"])
    dest=a.out/"comparisons.json"
    if a.replay:
        if not dest.is_file():raise ValueError("Missing archived comparisons")
        expected=json.loads(dest.read_text(encoding="utf-8"))
        for reg in out:
            for k in out[reg]:np.testing.assert_allclose(expected[reg][k],out[reg][k],rtol=0,atol=1e-10)
    else:dest.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()

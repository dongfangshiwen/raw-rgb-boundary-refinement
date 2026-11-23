"""Frozen target comparisons with one immutable confusion packet per frame."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from transformers import Mask2FormerForUniversalSegmentation

from scripts import dual_view_inference_v43 as seg
from scripts import idd_fog_evaluate_v100 as ev
from scripts import idd_fog_models_v100 as setup
from scripts import idd_peer_source_v100 as masknet
from scripts import semantic_subpixel_v94 as original
from scripts import subpixel_transport_v92 as legacy
from scripts.cross_scale_rgb_v44 import resize
from scripts.fit_range_v101 import OUT, PLAN
from scripts.idd_fog_data_v100 import DATA as IDD
from scripts.projective_ledger_v12 import REPO, digest, read
from scripts.range_decoder_v101 import RangeDecoder, FAMILIES, prepare

ARMS=("released","bilinear_grid","fixed_jbu","fixed_four","paper_fit",*FAMILIES,"anyup")
SEEDS=(101,211,307)
ANY = REPO.parent / "benchmark_data/anyup-v101"
PEER_SHA="b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59"
SUPPLEMENT=OUT.parent/"manuscript-2026-11-19/Boundary transfer revision/Supplementary Data S1.zip"

class Target:
    def __init__(self,name):
        self.name=name
        if name=="mask2former":
            assert digest(masknet.BASE/"model.safetensors")==masknet.SHA
            self.net=Mask2FormerForUniversalSegmentation.from_pretrained(str(masknet.BASE),local_files_only=True,
                        use_safetensors=True).cuda().eval().requires_grad_(False)
            p=read(masknet.BASE/"preprocessor_config.json")
            self.mean,self.std=(torch.tensor(p[k],device="cuda")[None,:,None,None] for k in ("image_mean","image_std"))
        else:
            self.net,self.mean,self.std=seg.load_b1()

    def __call__(self,image,native_size):
        size=image.shape[-2:]
        x=(image-self.mean)/self.std
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if self.name=="segformer":
                z=self.net(pixel_values=x).logits.float()
                return z,resize(z,native_size)
            pm=torch.ones((1,*size),device="cuda",dtype=torch.long)
            q=self.net(pixel_values=x,pixel_mask=pm)
        def scores(sz):
            return setup.peer.query_to_scores(q.class_queries_logits,q.masks_queries_logits,sz,sz)
        z=scores(q.masks_queries_logits.shape[-2:])
        return z,resize(scores(size),native_size)

def peer():
    sys.path.insert(0,str(ANY))
    from anyup.model import AnyUp
    assert digest(ANY/"anyup_multi_backbone.pth")==PEER_SHA
    model=AnyUp().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(ANY/"anyup_multi_backbone.pth",map_location="cpu",weights_only=True))
    return model

def rows(cohort):
    if cohort=="idd":
        with ZipFile(SUPPLEMENT) as archive:
            return json.loads(archive.read("idd/frames.json"))["fog"]
    return read(OUT/f"{cohort}-manifest.json")["rows"]

def scene(row,cohort):
    path=IDD/row["rgb"] if cohort=="idd" else REPO.parent/row["rgb_path"]
    assert digest(path)==row["rgb_sha"]
    with Image.open(path) as im:
        rgb=np.array(im.convert("RGB"))
    if cohort=="idd":
        truth=ev.native_truth(row)
    else:
        label=REPO.parent/row["mask_path"]
        assert digest(label)==row["mask_sha"]
        with Image.open(label) as im:
            truth=torch.from_numpy(np.array(im)).cuda().long()
    assert tuple(truth.shape)==rgb.shape[:2]
    return rgb,truth

def load_fits():
    packet=read(OUT/"training.json")
    assert packet["inputs_sha256"]["protocol"]==digest(PLAN) and packet["updates"]==5760
    fitted={}
    for r in packet["runs"]:
        path=OUT/r["weights"]
        assert r["sha256"]==digest(path)
        net=RangeDecoder(r["arm"]).cuda().eval().requires_grad_(False)
        net.load_state_dict(torch.load(path,weights_only=True))
        fitted[r["arm"],r["seed"]]=net
    assert len(fitted)==18
    return fitted

@torch.inference_mode()
def run(target,cohort,only_peer=False):
    setup.configure_cuda()
    cases=rows(cohort)
    directory=OUT/("counts-"+target+"-"+cohort+("-peer" if only_peer else ""))
    directory.mkdir(exist_ok=True)
    segmenter=Target(target)
    if only_peer:
        nets=peer()
        mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    else:
        fitted=load_fits()
        defaults=[RangeDecoder(f).cuda() for f in ("gaussian_gaussian","bilinear_bounded4")]
        old=original.load()
    lock=sha256((digest(PLAN)+digest(__file__)+digest(OUT/"training.json")+digest(Path(__file__).with_name("range_decoder_v101.py"))+PEER_SHA).encode()).hexdigest()
    t=time.monotonic()
    for i,row in enumerate(cases):
        key=row["id"]
        dest=directory/(key.replace("/","_")+".npz")
        wanted=["anyup"] if only_peer else list(ARMS[:-1])
        if dest.exists():
            with np.load(dest,allow_pickle=False) as cached:
                if cached["lock"].item()!=lock or cached["id"].item()!=key or cached["arms"].tolist()!=wanted:
                    raise RuntimeError("Existing counts use another protocol")
            continue
        rgb,truth=scene(row,cohort)
        size=tuple(truth.shape)
        data=ev.Counts(truth)
        batch,(h,w)=setup.peer_input(rgb)
        image=batch[...,:h,:w].contiguous()
        z,released=segmenter(image,size)
        assert z.shape[-2:]==(h//4,w//4) and h%4==0 and w%4==0
        def cm(scores):
            return np.stack(data.score(scores.argmax(1)[0]))
        def deterministic(c):
            return np.repeat(c[:,None],3,axis=1)
        if only_peer:
            # Released original window implementation with query chunking; same 4x output lattice.
            refined=nets((image-mean)/std,z,q_chunk_size=512)
            values={"anyup":deterministic(cm(resize(refined.float(),size)))}
        else:
            base=resize(z,size)
            guide=image.clamp(0,1).mul(255).round().byte()
            st=prepare(z,guide)
            def net_scores(net):
                return base+resize(net.correction(z,st),size)
            values={"released":deterministic(cm(released)),"bilinear_grid":deterministic(cm(base))}
            for arm,net in zip(("fixed_jbu","fixed_four"),defaults):
                values[arm]=deterministic(cm(net_scores(net)))
            shared=legacy.guide_stencil(guide[[0,0]],z.argmax(1)[0])
            values["paper_fit"]=np.stack([cm(base+resize(legacy.prediction("dual_free",old["dual_core",s],z,shared),size)) for s in SEEDS],axis=1)
            for family in FAMILIES:
                values[family]=np.stack([cm(net_scores(fitted[family,s])) for s in SEEDS],axis=1)
            if i==0:
                for name,score in (("candidate",net_scores(fitted["bilinear_bounded4",101])),("released",released)):
                    lab=score.argmax(1)[0].byte().cpu().numpy()
                    pic=OUT/f"prediction-{target}-{cohort}-{name}.png"
                    Image.fromarray(lab).save(pic)
                    with Image.open(pic) as disk:
                        np.testing.assert_array_equal(np.stack(data.score(torch.from_numpy(np.array(disk)).cuda())),
                            values["bilinear_bounded4" if name=="candidate" else "released"][:,0])
        counts=np.stack([values[k] for k in wanted],axis=1).astype(np.int32)
        assert np.all(counts.sum(axis=-1)==counts[:,:1,:1].sum(axis=-1)) and np.all(counts[1]<=counts[0])
        temp=dest.with_suffix(".part")
        with temp.open("wb") as f:
            np.savez_compressed(f,id=key,group=row["group"],weather=row.get("weather","fog"),
                arms=wanted,seeds=SEEDS,counts=counts,lock=lock)
        temp.replace(dest)
        if i==0 or (i+1)%32==0 or i+1==len(cases):
            print(target,cohort,"peer" if only_peer else "factorial",i+1,len(cases),round(time.monotonic()-t,1),flush=True)

if __name__=="__main__":
    cmd=argparse.ArgumentParser()
    cmd.add_argument("cohort",choices=("idd","bdd","acdc","fogreserve"))
    cmd.add_argument("--peer",action="store_true")
    cmd.add_argument("--targets",nargs="+",default=["mask2former","segformer"],choices=("mask2former","segformer"))
    args=cmd.parse_args()
    for name in args.targets:
        run(name,args.cohort,args.peer)

"""Frozen two-backbone evaluation; write paired integer counts and preselected example masks."""
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts import eval_range_v101 as prior
from scripts import native_gaussian_v102 as gaussian
from scripts import semantic_subpixel_v94 as original
from scripts import subpixel_transport_v92 as legacy
from scripts.cross_scale_rgb_v44 import resize
from scripts.idd_fog_data_v100 import MAP
from scripts.idd_fog_evaluate_v100 import Counts
from scripts.idd_fog_models_v100 import peer_input
from scripts.native_guide_v102 import PixelGrid,Strength,interpolate
from scripts.native_reserve_v102 import ARMS,ROOT,SEEDS
from scripts.projective_ledger_v12 import REPO,digest,read,now
from scripts.range_decoder_v101 import RangeDecoder
from scripts.segfix_peer_v102 import SegFix,SHA
from scripts.train_regional_guidance_v18 import atomic_write,configure_cuda

ALL=(*ARMS,"matched_gaussian_fixed","matched_gaussian_fit")
TRAINED=("scalar","rgb_gap","class_gap","matched_gaussian_fit")
REPEATED={"released","bilinear","segfix","jbu_native","fixed_native","fixed_lowpass","matched_gaussian_fixed"}
TARGETS=("mask2former","segformer")


def load_fits():
    fitted={}
    for name in ("training.json","gaussian-training.json"):
        for r in read(ROOT/name)["records"]:
            path=ROOT/r["weights"]
            assert digest(path)==r["sha256"]
            model=(RangeDecoder("bilinear_gaussian") if r["mode"]=="matched_gaussian_fit" else Strength(r["mode"]))
            model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
            fitted[r["mode"],r["seed"]]=model.cuda().eval().requires_grad_(False)
    return fitted


def load_image(row):
    a,b=(REPO.parent/row[k+"_path"] for k in ("rgb","mask"))
    assert digest(a)==row["rgb_sha"] and digest(b)==row["mask_sha"]
    with Image.open(a) as img, Image.open(b) as label:
        rgb=np.array(img.convert("RGB"))
        ids=np.array(label)
    if row["cohort"]=="fog":
        lut=np.full(256,255,dtype=np.uint8)
        for k,v in MAP.items():
            lut[k]=v
        ids=lut[ids]
    assert ids.shape==rgb.shape[:2]
    return rgb,torch.from_numpy(ids.copy()).cuda().long()


def predict(z,released,small,raw,offsets,old,nets):
    shape=raw.shape[-2:]
    base=resize(z,shape)
    bilinear=base.argmax(1)[0].byte()
    baseline=released.argmax(1)[0].byte()
    out={"bilinear":[bilinear],"released":[baseline],"segfix":[baseline.flatten()[offsets].reshape(shape)]}
    guide=small.clamp(0,1).mul(255).round().byte()
    stencil=legacy.guide_stencil(guide[[0,0]],z.argmax(1)[0])
    out["paper_512"]=[]
    for s in SEEDS:
        delta=legacy.prediction("dual_free",old["dual_core",s],z,stencil)
        out["paper_512"].append((base+resize(delta,shape)).argmax(1)[0].byte())
    del base
    native=PixelGrid(z,small,raw)
    for arm in ALL:
        if arm not in out:
            out[arm]=[bilinear.clone() for _ in range(1 if arm in REPEATED else len(SEEDS))]
    for ids in native.active().nonzero().flatten().split(32768):
        if len(ids)==0:
            continue
        native_features=native.four(ids,include_distance=True)
        low=native.four(ids,lowpass=True)
        high=native_features[:4]
        def put(arm,i,scores):
            out[arm][i].view(-1)[ids]=scores.argmax(-1).byte()
        put("jbu_native",0,native.jbu(ids))
        put("matched_gaussian_fixed",0,gaussian.scores(native_features))
        put("fixed_native",0,interpolate(high[0],high[1],high[3]))
        put("fixed_lowpass",0,interpolate(low[0],low[1],low[3]))
        for i,s in enumerate(SEEDS):
            for mode in TRAINED[:-1]:
                put(mode,i,nets[mode,s](high))
            put("class_gap_lowpass",i,nets["class_gap",s](low))
            put("matched_gaussian_fit",i,gaussian.scores(native_features,nets["matched_gaussian_fit",s]))
    return out


def sample_counts(out,scorer):
    mats=[]
    for arm in ALL:
        m=np.stack([np.stack(scorer.score(lab)) for lab in out[arm]],axis=1)
        mats.append(np.repeat(m,3,axis=1) if len(out[arm])==1 else m)
    counts=np.stack(mats,axis=1).astype(np.int32)
    assert np.all(counts.sum(-1)==counts[:,:1,:1].sum(-1)) and np.all(counts[1]<=counts[0])
    return counts


@torch.inference_mode()
def run():
    configure_cuda()
    manifest=read(ROOT/"manifest.json")
    assert manifest["selection"]==digest(ROOT/"selection.json")
    paths=[ROOT/n for n in ("protocol.json","manifest.json","training.json","matched-controls.json","gaussian-training.json")]
    paths.extend(Path(__file__).with_name(f) for f in ("native_guide_v102.py","native_gaussian_v102.py","segfix_peer_v102.py"))
    lock=sha256(("".join(digest(p) for p in paths)+digest(__file__)+SHA).encode()).hexdigest()
    models={name:prior.Target(name) for name in TARGETS}
    segfix=SegFix()
    nets=load_fits()
    old=original.load()
    rows=manifest["rows"]
    examples={next(r["id"] for r in rows if r["cohort"]==c) for c in ("bdd","fog")}
    tic=monotonic()
    for i,row in enumerate(rows):
        dest=ROOT/"counts"/row["cohort"]/(row["id"].replace("/","_")+".npz")
        if dest.exists():
            with np.load(dest,allow_pickle=False) as cached:
                assert cached["lock"].item()==lock and cached["id"].item()==row["id"]
            continue
        rgb,truth=load_image(row)
        scorer=Counts(truth)
        offsets=segfix.lookup(rgb)
        raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        padded,(h,w)=peer_input(rgb)
        small=padded[...,:h,:w].contiguous()
        all_counts=[]
        for name,model in models.items():
            z,released=model(small,truth.shape)
            labels=predict(z,released,small,raw,offsets,old,nets)
            all_counts.append(sample_counts(labels,scorer))
            if row["id"] in examples:
                directory=ROOT/"examples"/row["cohort"]/name
                directory.mkdir(parents=True,exist_ok=True)
                for arm in ALL:
                    Image.fromarray(labels[arm][0].cpu().numpy()).save(directory/(arm+".png"))
            del released,labels,z
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(".part")
        with tmp.open("wb") as f:
            np.savez_compressed(f,counts=np.stack(all_counts),id=row["id"],group=row["group"],seeds=SEEDS,
                                targets=TARGETS,arms=ALL,lock=lock,edge_pixels=scorer.edge_count)
        tmp.replace(dest)
        if i==0 or (i+1)%16==0 or i+1==len(rows):
            print("Native evaluation",i+1,len(rows),row["cohort"],round(monotonic()-tic,1),flush=True)
    atomic_write(ROOT/"evaluation.json",dict(lock=lock,files=len(rows),examples=sorted(examples),targets=TARGETS,arms=ALL,
                 segfix_release=SHA,native_device=torch.cuda.get_device_name(),finished_utc=now()))


if __name__=="__main__":
    run()

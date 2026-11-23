"""Test increments on original manuscript scores, including a 150-class frozen backbone."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation
from scripts import eval_path_v104 as prior
from scripts import path_decoder_v104 as op
from scripts import train_anchor_v105 as fit
from scripts.reserve_anchor_v105 import OUT
from scripts.weatherproof_release_v103 import WORK,read,write

CORE=("released","paper","fixed_grid","gaussian","gaussian_path",*fit.MODES)
AFTER={"segfix":"released","fixed_grid_segfix":"fixed_grid","gaussian_segfix":"gaussian","paper_segfix":"paper","path_segfix":"path"}
ROAD=(*CORE,*AFTER)
LABELS=("building","road","sky","tree","grass","rock")
WP_IDS=(1,3,4,9,6,5)

class ADE:
    def __init__(self):
        folder=WORK/"colorawarenet-joint-model/pretrained/segformer-b0-ade"
        for f,v in read(OUT/"ade-weights.json")["files"].items():assert prior.base.digest(folder/f)==v
        self.net=SegformerForSemanticSegmentation.from_pretrained(str(folder),local_files_only=True,use_safetensors=True).cuda().eval().requires_grad_(False)
        ids={v:int(k) for k,v in self.net.config.id2label.items()}
        self.ids=[ids[k] for k in LABELS];assert len(ids)==150
        pp=read(folder/"preprocessor_config.json")
        self.mean,self.std=(torch.tensor(pp[k],device="cuda")[None,:,None,None] for k in ("image_mean","image_std"))

    def __call__(self,small,native):
        with torch.autocast("cuda",dtype=torch.bfloat16):z=self.net(pixel_values=(small-self.mean)/self.std).logits.float()
        return z,prior.base.resize(z,native)

def maps(y,source_ids):
    count=len(source_ids);to_label=np.full(256,255,np.uint8);to_label[list(WP_IDS[:count])]=np.arange(count)
    to_prediction=np.full(256,18,np.uint8);to_prediction[list(source_ids)]=np.arange(count)
    return to_label[y],to_prediction

def predictions(z,released,small,old,anchors,strength,trained,is_road):
    shape=released.shape[-2:];native=prior.base.resize(z,shape)
    to_mask=lambda x:x.argmax(1)[0].byte().cpu().numpy()
    out={"released":[to_mask(released)]};st=op.Stencil(z,small);ids=st.active.nonzero().flatten()
    if len(ids):feat=tuple(torch.cat(items) for items in zip(*(st.features(i) for i in ids.split(6144))))
    else:feat=None
    penalties=fit.cues(feat) if feat is not None else None
    def delta(weight,baseline):
        work=z.new_zeros(16*st.h*st.w,st.c)
        if feat is not None:work[ids]=((weight-baseline)[:,None]@feat[-1]).squeeze(1)
        grid=work.reshape(4*st.h,4*st.w,st.c).permute(2,0,1).contiguous()[None]
        return prior.base.resize(grid,shape)
    stencil=prior.paper.paper_stencil(z,small) if is_road else None
    for seed in op.SEEDS:
        logp=anchors[seed](feat) if feat is not None else None
        w=logp.softmax(-1) if feat is not None else None
        if is_road:
            paper=prior.paper.paper_scores(z,stencil,old["dual_core",seed],native)
        else:paper=native+delta(w,feat[0] if feat is not None else None)
        out.setdefault("paper",[]).append(to_mask(paper))
        for mode in fit.MODES:
            extended=(logp-strength[mode,seed]*penalties[mode]).softmax(-1) if feat is not None else None
            out.setdefault(mode,[]).append(to_mask(paper+delta(extended,w)))
        del paper
        for mode,net in (("fixed_grid",trained["fixed_grid",0]),("gaussian",trained["gaussian",seed]),("gaussian_path",trained["path",seed])):
            if mode=="fixed_grid" and seed!=op.SEEDS[0]:continue
            w=net.weights(feat) if feat is not None else None
            out.setdefault(mode,[]).append(to_mask(native+delta(w,feat[0] if feat is not None else None)))
    return out

def count(masks,truth,lut,arms):
    scorer=prior.old.SparseCounts(truth);pick=lambda item: item.ravel() if lut is None else lut[item.ravel()]
    ref=pick(masks["released"][0]);ref_cm=scorer.cm(ref);rows=[]
    for mode in arms:
        arr=[scorer.change(ref,pick(mask),ref_cm)[0] for mask in masks[mode]]
        if len(arr)==1:arr*=len(op.SEEDS)
        rows.append(np.stack(arr,axis=1))
    cm=np.stack(rows,axis=1).astype(np.int32)
    assert (cm>=0).all() and (cm[1]<=cm[0]).all() and (cm.sum(-1)==cm[:,:1,:1].sum(-1)).all()
    return cm

@torch.inference_mode()
def evaluate(panel):
    prior.base.configure_cuda();tic=monotonic()
    old,anchors,strength=fit.load();trained=prior.train.load();trained["fixed_grid",0]=op.Decoder("gaussian").cuda().eval()
    models={k:prior.base.prior.Target(k) for k in ("mask2former","segformer")}
    if panel=="wp":models["ade"]=ADE()
    fix=prior.base.SegFix();rows=read(OUT/(panel+".json"))["rows"]
    lock=sha256("".join(prior.base.digest(f) for f in [OUT/(panel+".json"),OUT/"training.json",OUT/"protocol.json",prior.OUT/"training.json",Path(__file__),Path(fit.__file__)]).encode()).hexdigest()
    suites={"mask2former":list(ROAD),"segformer":list(ROAD)}
    if panel=="wp":suites.update({f"ade{k}":list(CORE) for k in (3,5,6)})
    for i,r in enumerate(rows):
        dest=OUT/"counts"/panel/(r["id"].replace("/","_")+".npz")
        if dest.exists():
            with np.load(dest,allow_pickle=False) as f:assert f["lock"].item()==lock
            continue
        rgb,ys=prior.old.open_frame(r)
        raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        padded,(h,w)=prior.base.peer_input(rgb);small=padded[...,:h,:w].contiguous()
        if panel=="wp":
            small=prior.base.resize(raw,tuple(max(32,min(512,32*round(d/32))) for d in (h,w)),antialias=True)
            with Image.open(WORK/r["paths"][-1]) as im:y=np.array(im)
            if y.ndim==3:y=y[:,:,0]
        lookup=fix.lookup(rgb,True)[0].cpu().numpy()
        out={}
        for name,net in models.items():
            z,release=net(small,raw.shape[-2:]);rd=name!="ade"
            masks=predictions(z,release,small,old,anchors,strength,trained,rd)
            del z,release
            if rd:
                for k,src in AFTER.items():masks[k]=[m.ravel()[lookup].reshape(m.shape) for m in masks[src]]
                yy,lut=maps(y,(2,0,10)) if panel=="wp" else (ys[0],None)
                out[name]=count(masks,yy,lut,ROAD)
            else:
                for k in (3,5,6):
                    yy,lut=maps(y,net.ids[:k]);out[f"ade{k}"]=count(masks,yy,lut,CORE)
            if i==0 and rd and name=="mask2former":
                folder=OUT/"examples"/panel;folder.mkdir(exist_ok=True,parents=True)
                Image.fromarray(rgb).save(folder/"input.jpg",quality=94);Image.fromarray(ys[0]).save(folder/"truth.png")
                for k in ("released","paper","path","gaussian","paper_segfix","gaussian_segfix"):Image.fromarray(masks[k][0]).save(folder/(k+".png"))
                write(folder/"frame.json",dict(id=r["id"],group=r["group"],seed=101,target=name))
            del masks
        assert set(out)==set(suites);dest.parent.mkdir(exist_ok=True,parents=True);tmp=dest.with_suffix(".part")
        with tmp.open("wb") as f:np.savez_compressed(f,**out,id=r["id"],group=r["group"],weather=r.get("weather","unrestricted"),lock=lock,seeds=op.SEEDS)
        tmp.replace(dest)
        if i==0 or (i+1)%20==0 or i+1==len(rows):print(panel,i+1,len(rows),round(monotonic()-tic,1),"s",flush=True)
    write(OUT/(panel+"-evaluation.json"),dict(suites=suites,lock=lock,frames=len(rows),groups=len({r["group"] for r in rows}),
             seeds=list(op.SEEDS),seconds=monotonic()-tic,ade_direct_taxonomy=dict(zip(LABELS,models["ade"].ids)) if "ade" in models else {}))

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("panel",choices=("wp","bdd"));evaluate(cli.parse_args().panel)

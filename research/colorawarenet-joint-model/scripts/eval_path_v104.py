"""Frozen dual-backbone inference. All decoder contrasts share the same coarse scores."""
import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
from PIL import Image
import torch
from scripts import directional_veto_v103 as paper
from scripts import eval_native_guide_v102 as base
from scripts import native_gaussian_v102 as native
from scripts import path_decoder_v104 as candidate
from scripts import train_path_v104 as train
from scripts import validate_fog_v103 as old
from scripts.holdout_path_v104 import OUT
from scripts.weatherproof_release_v103 import read,write

ARMS=("released","bilinear","paper","native_gaussian","segfix","native_gaussian_segfix","paper_segfix",
      "fixed_grid",*candidate.MODES,"path_segfix")
MATCH={"released":"released","paper":"paper","native_gaussian":"gaussian","segfix":"segfix",
       "native_gaussian_segfix":"gaussian_segfix","paper_segfix":"paper_segfix"}
PLAN={"new_random":"random.json","new_fog":"fog-new.json","old_fog":"dense-roads/manifest.json",
      "idd":"manifest.json","weatherproof":"wp-manifest.json"}

def rows_for(panel):
    p=(OUT if panel.startswith("new_") else (old.LAST if panel=="idd" else old.OUT))/PLAN[panel]
    rows=read(p)["rows"]
    if panel=="idd":rows=[r for r in rows if r["cohort"]=="fog"]
    if panel=="weatherproof":rows=[r for r in rows if r["weather"] in ("fog","mixed")]
    return rows,p

def trained_predictions(z,small,native_base,nets):
    st=candidate.Stencil(z,small)
    ids=st.active.nonzero().flatten()
    feat=tuple(torch.cat(seq) for seq in zip(*(st.features(ix) for ix in ids.split(12288) if len(ix)))) if len(ids) else ()
    for (mode,seed),net in nets.items():
        delta=z.new_zeros((16*st.h*st.w,st.c))
        if len(ids):
            delta[ids]=((net.weights(feat)-feat[0]).unsqueeze(-2)@feat[-1]).squeeze(-2)
        working=delta.reshape(4*st.h,4*st.w,st.c).permute(2,0,1)[None].contiguous()
        yield mode,seed,(native_base+base.resize(working,native_base.shape[-2:])).argmax(1)[0].byte()

def infer(z,released,small,raw,lookup,weights,nets):
    shape=raw.shape[-2:]; initial=base.resize(z,shape)
    move=lambda m:m.flatten()[lookup].reshape(shape)
    released=released.argmax(1)[0].byte();bilinear=initial.argmax(1)[0].byte()
    gaussian=bilinear.clone();grid=base.PixelGrid(z,small,raw)
    for ix in grid.active().nonzero().flatten().split(32768):
        if len(ix):gaussian.flatten()[ix]=native.scores(grid.four(ix,include_distance=True)).argmax(-1).byte()
    masks={k:[v.cpu().numpy()] for k,v in {"released":released,"bilinear":bilinear,"segfix":move(released),
           "native_gaussian":gaussian,"native_gaussian_segfix":move(gaussian)}.items()}
    stencil=paper.paper_stencil(z,small)
    for seed in candidate.SEEDS:
        m=paper.paper_scores(z,stencil,weights["dual_core",seed],initial).argmax(1)[0].byte()
        for name,mask in (("paper",m),("paper_segfix",move(m))):masks.setdefault(name,[]).append(mask.cpu().numpy())
    for mode,seed,m in trained_predictions(z,small,initial,nets):
        masks.setdefault(mode,[]).append(m.cpu().numpy())
        if mode=="path":masks.setdefault("path_segfix",[]).append(move(m).cpu().numpy())
    assert set(masks)==set(ARMS)
    return masks

def packet(masks,scorer):
    reference=masks["released"][0].ravel();baseline=scorer.cm(reference)
    arrays=[]
    for k in ARMS:
        cms=[scorer.change(reference,x.ravel(),baseline)[0] for x in masks[k]]
        if len(cms)==1:cms*=len(candidate.SEEDS)
        arrays.append(np.stack(cms,axis=1))
    counts=np.stack(arrays,axis=1).astype(np.int32)
    assert (counts>=0).all() and (counts[1]<=counts[0]).all()
    assert (counts.sum(-1)==counts[:,:1,:1].sum(-1)).all()
    return counts

@torch.inference_mode()
def run(panels):
    base.configure_cuda()
    nets=train.load();nets["fixed_grid",0]=candidate.Decoder("gaussian").cuda().eval()
    models={t:base.prior.Target(t) for t in base.TARGETS}
    fix=base.SegFix();weights=base.original.load()
    for panel in panels:
        rows,p=rows_for(panel);tic=monotonic()
        lock=sha256("".join(base.digest(f) for f in [p,OUT/"protocol.json",OUT/"training.json",Path(__file__),Path(candidate.__file__)]).encode()).hexdigest()
        checked=0
        for i,r in enumerate(rows):
            key=r["id"].replace("/","_")+".npz";dest=OUT/"counts"/panel/key
            if dest.exists():
                with np.load(dest,allow_pickle=False) as f:assert f["lock"].item()==lock
                if not panel.startswith("new_"):checked+=1
                continue
            rgb,truth=old.open_frame(r);raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
            padded,(h,w)=base.peer_input(rgb);small=padded[...,:h,:w].contiguous()
            if panel=="weatherproof":
                small=base.resize(raw,tuple(max(32,min(512,32*round(d/32))) for d in (h,w)),antialias=True)
            lookup=fix.lookup(rgb,True)[0]
            scorers=[old.SparseCounts(y) for y in truth];slices=[]
            for name,model in models.items():
                z,released=model(small,raw.shape[-2:]);masks=infer(z,released,small,raw,lookup,weights,nets)
                slices.append(np.stack([packet(masks,s) for s in scorers]))
                if i==0 and name=="mask2former":
                    folder=OUT/"examples"/panel;folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rgb).save(folder/"input.jpg",quality=94)
                    Image.fromarray(truth[0]).save(folder/"ground-truth.png")
                    for method in ARMS:Image.fromarray(masks[method][0]).save(folder/(method+".png"))
                    write(folder/"frame.json",dict(id=r["id"],group=r["group"],seed=candidate.SEEDS[0],target=name))
                del released,z,masks
            counts=np.stack(slices)
            if not panel.startswith("new_"):
                oldpath=old.OUT/"dense-roads/counts"/key if panel=="old_fog" else old.OUT/"counts"/r["cohort"]/key
                with np.load(oldpath,allow_pickle=False) as f:
                    data=f["counts"][:,None] if panel=="old_fog" else f["counts"]
                    for a,b in MATCH.items():
                        np.testing.assert_array_equal(counts[:,:,:,ARMS.index(a)],data[:,:,:,f["arms"].tolist().index(b)])
                checked+=1
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(".part")
            with tmp.open("wb") as fp:np.savez_compressed(fp,counts=counts,id=r["id"],group=r["group"],lock=lock,
                            targets=base.TARGETS,arms=ARMS,seeds=candidate.SEEDS,shape=raw.shape[-2:],working=small.shape[-2:])
            tmp.replace(dest)
            if i==0 or (i+1)%16==0 or i+1==len(rows):
                print(panel,i+1,len(rows),round(monotonic()-tic,1),"s",flush=True)
        write(OUT/(panel+"-evaluation.json"),dict(panel=panel,frames=len(rows),groups=len({r["group"] for r in rows}),lock=lock,
            archived_six_arm_count_matches=checked,targets=list(base.TARGETS),arms=list(ARMS),seeds=list(candidate.SEEDS),seconds=monotonic()-tic))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("panels",nargs="+",choices=PLAN);run(parser.parse_args().panels)

"""Evaluate frozen SegFix after each color reconstruction, verifying prior integer counts."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from scripts import eval_native_guide_v102 as base
from scripts import native_gaussian_v102 as gaussian
from scripts import subpixel_transport_v92 as legacy
from scripts.cross_scale_rgb_v44 import resize
from scripts.native_guide_v102 import PixelGrid
from scripts.native_reserve_v102 import ROOT, SEEDS
from scripts.projective_ledger_v12 import digest, read, now
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda

ARMS=("released","segfix","paper_512","matched_gaussian_fixed","paper_then_segfix","gaussian_then_segfix")
def warp(mask,offset):
    return mask.flatten()[offset].reshape(mask.shape)

def infer(z,released,small,raw,offset,old):
    shape=raw.shape[-2:]
    ref=released.argmax(1)[0].byte()
    scores=resize(z,shape)
    g=scores.argmax(1)[0].byte()
    low=(small.clamp(0,1)*255).round().byte()
    stencil=legacy.guide_stencil(low[[0,0]],z.argmax(1)[0])
    paper=[]
    for seed in SEEDS:
        correction=legacy.prediction("dual_free",old["dual_core",seed],z,stencil)
        paper.append((scores+resize(correction,shape)).argmax(1)[0].byte())
    del scores
    grid=PixelGrid(z,small,raw)
    for ids in grid.active().nonzero().flatten().split(32768):
        if len(ids):
            g.view(-1)[ids]=gaussian.scores(grid.four(ids,include_distance=True)).argmax(-1).byte()
    return {"released":[ref],"segfix":[warp(ref,offset)],"paper_512":paper,"matched_gaussian_fixed":[g],
            "paper_then_segfix":[warp(m,offset) for m in paper],"gaussian_then_segfix":[warp(g,offset)]}

def packet(out,scorer):
    parts=[]
    for name in ARMS:
        labels=out[name]
        counts=np.stack([np.stack(scorer.score(mask)) for mask in labels],axis=1)
        parts.append(np.repeat(counts,3,axis=1) if len(labels)==1 else counts)
    return np.stack(parts,axis=1).astype(np.int32)

@torch.inference_mode()
def run():
    configure_cuda()
    dest=ROOT/"composition"
    source=read(ROOT/"evaluation.json")
    lock=sha256("".join(digest(p) for p in [ROOT/"composition-protocol.json",ROOT/"evaluation.json",
                 Path(__file__),Path(base.__file__)]).encode()).hexdigest()
    targets={name:base.prior.Target(name) for name in base.TARGETS}
    modifier=base.SegFix()
    old=base.original.load()
    examples=set(source["examples"])
    matched_arms=np.array([source["arms"].index(a) for a in ARMS[:4]])
    rows=read(ROOT/"manifest.json")["rows"]
    for i,row in enumerate(rows):
        key=row["id"].replace("/","_")+".npz"
        path=dest/"counts"/row["cohort"]/key
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                assert d["lock"].item()==lock and d["id"].item()==row["id"]
            continue
        rgb,gt=base.load_image(row)
        scorer=base.Counts(gt)
        padded,(h,w)=base.peer_input(rgb)
        small=padded[...,:h,:w].contiguous()
        raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
        offset=modifier.lookup(rgb)
        parts=[]
        for name,model in targets.items():
            z,released=model(small,gt.shape)
            out=infer(z,released,small,raw,offset,old)
            parts.append(packet(out,scorer))
            if row["id"] in examples:
                folder=dest/"examples"/row["cohort"]/name
                folder.mkdir(parents=True,exist_ok=True)
                for a in ARMS[-2:]:
                    Image.fromarray(out[a][0].cpu().numpy()).save(folder/(a+".png"))
            del out,z,released
        counts=np.stack(parts)
        with np.load(ROOT/"counts"/row["cohort"]/key,allow_pickle=False) as recorded:
            assert recorded["lock"].item()==source["lock"]
            np.testing.assert_array_equal(counts[:,:,:4],np.take(recorded["counts"],matched_arms,axis=2))
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(".part")
        with tmp.open("wb") as f:
            np.savez_compressed(f,counts=counts,id=row["id"],group=row["group"],seeds=SEEDS,
                                targets=base.TARGETS,arms=ARMS,lock=lock,edge_pixels=scorer.edge_count)
        tmp.replace(path)
        if i==0 or (i+1)%32==0 or i+1==len(rows):
            print("Composition and four independent baseline comparisons",i+1,len(rows),flush=True)
    atomic_write(dest/"evaluation.json",dict(lock=lock,arms=ARMS,targets=base.TARGETS,files=len(rows),
                 parent=source["lock"],independently_recomputed_baseline_families=4,finished=now()))

if __name__=="__main__":
    run()

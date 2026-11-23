"""Score the original transferred decoder on the exact same two artificial veils."""
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from boundary_refinement.evaluation.semantic import eval_native_guide as ev
from boundary_refinement.evaluation.paired import paired_veil as paired
from boundary_refinement.data.cohorts.native_reserve import ROOT,SEEDS
from boundary_refinement.data.provenance import digest,read,now
from boundary_refinement.training.train_regional_guidance import configure_cuda,atomic_write

@torch.inference_mode()
def main():
    configure_cuda()
    note=read(ROOT/"paired-evaluation.json")
    paths=(ROOT/"paired-evaluation.json",ROOT/"paired-manuscript-protocol.json",Path(__file__),Path(paired.__file__))
    lock=sha256("".join(digest(p) for p in paths).encode()).hexdigest()
    model=ev.prior.Target("mask2former")
    old=ev.original.load()
    rows=[r for r in read(ROOT/"manifest.json")["rows"] if r["cohort"]=="bdd"]
    dest=ROOT/"paired-manuscript"
    dest.mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        key=r["id"]+".npz"
        path=dest/key
        if path.exists():
            with np.load(path,allow_pickle=False) as prior:
                assert prior["lock"].item()==lock
            continue
        rgb,gt=ev.load_image(r)
        scorer=ev.Counts(gt)
        points=[]
        with np.load(ROOT/"paired-counts"/key,allow_pickle=False) as ref:
            assert ref["lock"].item()==note["lock"]
            released=ref["counts"][:,0,:,0]
        for k,t in enumerate(paired.STRENGTH):
            image=np.rint(t*rgb.astype(np.float32)+(1-t)*.90*255).clip(0,255).astype(np.uint8)
            padded,(h,w)=ev.peer_input(image)
            small=padded[...,:h,:w].contiguous()
            z,base=model(small,gt.shape)
            np.testing.assert_array_equal(np.stack(scorer.score(base.argmax(1)[0].byte())),released[k])
            del base
            base=ev.resize(z,gt.shape)
            guide=(small.clamp(0,1)*255).round().byte()
            stencil=ev.legacy.guide_stencil(guide[[0,0]],z.argmax(1)[0])
            bank=[]
            for seed in SEEDS:
                d=ev.legacy.prediction("dual_free",old["dual_core",seed],z,stencil)
                prediction=(base+ev.resize(d,gt.shape)).argmax(1)[0].byte()
                bank.append(np.stack(scorer.score(prediction)))
                if i==0 and seed==SEEDS[0]:
                    Image.fromarray(prediction.cpu().numpy()).save(ROOT/"paired-examples"/str(t)/"paper_512-labels.png")
            points.append(np.stack(bank,axis=1))
        tmp=path.with_suffix(".part")
        with tmp.open("wb") as f:
            np.savez_compressed(f,counts=np.stack(points).astype(np.int32),id=r["id"],group=r["group"],lock=lock)
        tmp.replace(path)
    atomic_write(dest/"evaluation.json",dict(lock=lock,parent=note["lock"],frames=len(rows),seeds=SEEDS,finished=now()))
    print("Original manuscript decoder on 128 x 2 matched inputs; all released anchors matched.")

if __name__=="__main__":
    main()

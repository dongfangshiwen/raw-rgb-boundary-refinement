"""Paired secondary check on the already-inspected real fog driving cohorts."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from scripts import eval_anchor_v105 as ext
from scripts import eval_path_v104 as discovery
from scripts.reserve_anchor_v105 import OUT
from scripts.weatherproof_release_v103 import read,write

PANELS=("new_fog","old_fog","idd")
RULE={"stage":"Secondary re-evaluation after seeing both V104 and V105 main results. No fresh-road or selection claim.",
      "cohorts":"All four visually foggy BDD keyframes from V104, all four from V103, and all 62 frames in 16 previously screened IDD-AW fog drives. Keep the two BDD panels separate.",
      "methods":"Every V105 frozen road arm, the same workspace, native projection, SegFix offsets, three seeds and held-in MUSES coefficients. Primary descriptive contrasts for this secondary evaluation are extra path versus archived paper and aligned versus shifted, and paper+SegFix versus paper, SegFix and fitted same-grid Gaussian+SegFix. Report both target backbones, full mIoU and the six-pixel native GT boundary band. Use the existing 17-ID IDD mapping. No interval below six drives; no target updates."}

@torch.inference_mode()
def run():
    plan=OUT/"road-recheck-protocol.json"
    if plan.exists():assert read(plan)==RULE
    else:write(plan,RULE)
    ext.prior.base.configure_cuda();models={k:ext.prior.base.prior.Target(k) for k in ("mask2former","segformer")}
    fix=ext.prior.base.SegFix();old,anchors,strength=ext.fit.load();nets=ext.prior.train.load();nets["fixed_grid",0]=ext.op.Decoder("gaussian").cuda().eval()
    cases={}
    for panel in PANELS:
        rows,source=discovery.rows_for(panel);lock=sha256("".join(ext.prior.base.digest(p) for p in [plan,source,OUT/"training.json",Path(__file__),Path(ext.__file__)]).encode()).hexdigest()
        for i,row in enumerate(rows):
            key=row["id"].replace("/","_")+".npz";dest=OUT/"road-recheck"/panel/key
            if dest.exists():
                with np.load(dest,allow_pickle=False) as f:assert f["lock"].item()==lock
                continue
            rgb,truth=discovery.old.open_frame(row);padded,(h,w)=ext.prior.base.peer_input(rgb);small=padded[...,:h,:w].contiguous()
            lut=fix.lookup(rgb,True)[0].cpu().numpy();values=[]
            for name,net in models.items():
                z,rel=net(small,rgb.shape[:2]);masks=ext.predictions(z,rel,small,old,anchors,strength,nets,True)
                for k,s in ext.AFTER.items():masks[k]=[m.ravel()[lut].reshape(m.shape) for m in masks[s]]
                values.append(ext.count(masks,truth[0],None,ext.ROAD));del masks,z,rel
            values=np.stack(values)
            with np.load(discovery.OUT/"counts"/panel/key,allow_pickle=False) as history:
                for name in ("released","paper","segfix","paper_segfix"):
                    np.testing.assert_array_equal(values[:,:,ext.ROAD.index(name)],history["counts"][:,0,:,history["arms"].tolist().index(name)])
            dest.parent.mkdir(exist_ok=True,parents=True);tmp=dest.with_suffix(".part")
            with tmp.open("wb") as f:np.savez_compressed(f,counts=values,id=row["id"],group=row["group"],lock=lock)
            tmp.replace(dest)
            if i==0 or (i+1)%24==0 or i+1==len(rows):print(panel,i+1,len(rows),flush=True)
        cases[panel]=dict(frames=len(rows),groups=len({r["group"] for r in rows}),lock=lock,matched_existing_arms=4)
    write(OUT/"road-recheck-evaluation.json",dict(panels=cases,targets=list(models),arms=list(ext.ROAD),seeds=list(ext.op.SEEDS)))
if __name__=="__main__":run()

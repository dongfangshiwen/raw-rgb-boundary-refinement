"""Replicate the identical frozen contrasts on the separately screened 19-class clips."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from scripts import validate_fog_v103 as ev
from scripts import score_fog_v103 as score
from scripts import directional_veto_v103 as route
from scripts.projective_ledger_v12 import digest
from scripts.weatherproof_release_v103 import OUT,read,write

R=OUT/"dense-roads"

@torch.inference_mode()
def run():
    ev.configure_cuda()
    rows=read(R/"manifest.json")["rows"]
    components=[R/"protocol.json",R/"reserve.json",R/"screen.json",R/"shortlist.json",R/"weather-lock.json",R/"manifest.json",
                OUT/"training.json",OUT/"evaluation.json",Path(ev.__file__),Path(__file__)]
    lock=sha256("".join(digest(p) for p in components).encode()).hexdigest()
    if rows:
        models={t:ev.base.prior.Target(t) for t in ev.base.TARGETS}
        modifier=ev.base.SegFix();old=ev.base.original.load();nets=ev.fit.load()
    (R/"counts").mkdir(exist_ok=True)
    for index,row in enumerate(rows):
        dest=R/"counts"/(row["id"]+".npz")
        if dest.exists():
            with np.load(dest,allow_pickle=False) as f:assert f["lock"].item()==lock
            continue
        rgb,(truth,)=ev.open_frame(row)
        native=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        padded,(h,w)=ev.base.peer_input(rgb)
        small=padded[...,:h,:w].contiguous()
        if rgb.shape[:2]!=(720,1280):
            small=ev.base.resize(native,tuple(max(32,min(512,32*round(d/32))) for d in (h,w)),antialias=True)
        lookup,offset=modifier.lookup(rgb,True)
        scorer=ev.SparseCounts(truth)
        mats=[];edits=[]
        for name,m in models.items():
            masks=ev.native_predictions(m,small,native,lookup,offset,old,nets)
            counts,delta=ev.packet(masks,scorer,False)
            mats.append(counts);edits.append(delta)
            if index==0:ev.save_examples(row,name,masks,rgb,truth,"dense-road")
        tmp=dest.with_suffix(".part")
        with tmp.open("wb") as fp:
            np.savez_compressed(fp,counts=np.stack(mats),edits=np.stack(edits),group=row["group"],id=row["id"],lock=lock,
                                arms=ev.ARMS,targets=ev.base.TARGETS,seeds=route.SEEDS,native_size=truth.shape)
        tmp.replace(dest)
    write(R/"evaluation.json",dict(files=len(rows),lock=lock,source_training=digest(OUT/"training.json"),arms=ev.ARMS))
    score.rescore_dense(R)
    print("Completed full 19-class independent clips",len(rows),flush=True)
if __name__=="__main__":
    run()

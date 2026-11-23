"""Match native-resolution donor-strength fits using only nine MUSES source recordings."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.audit_dual_output_v66 import AUTHOR_HASH,author_loader
from scripts.idd_fog_models_v100 import peer_input
from scripts.native_guide_v102 import PixelGrid,Strength
from scripts.native_reserve_v102 import ROOT,SEEDS,protocol
from scripts.projective_ledger_v12 import digest,now
from scripts.real_fog_correction_v70 import DATA,paths
from scripts.restoration_affinity_v75 import boundary_band
from scripts.train_regional_guidance_v18 import configure_cuda,atomic_write,torch_save

MODES=("scalar","rgb_gap","class_gap")
PER_STRATUM=8192


@torch.no_grad()
def source_bank(include_distance=False):
    author,model,source_hash=author_loader()
    model.eval()
    mean,std=(torch.tensor(x,device="cuda")[None,:,None,None] for x in (author.MEAN,author.STD))
    bank=defaultdict(list)
    for index,(key,group,_,_,truth,rgb) in enumerate(paths("fit")):
        padded,(h,w)=peer_input(rgb)
        small=padded[...,:h,:w].contiguous()
        z=model.predict((small-mean)/std).float()
        z=F.interpolate(z,(h//4,w//4),mode="bilinear",align_corners=False)
        raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        grid=PixelGrid(z,small,raw)
        valid=(truth.flatten()<19)&grid.active()
        edge=boundary_band(truth).flatten()&valid
        gen=torch.Generator(device="cuda").manual_seed(int.from_bytes(sha256(key.encode()).digest()[:4],"big"))
        ids=[]
        for region in (valid,edge):
            eligible=region.nonzero().flatten()
            if not len(eligible):
                eligible=valid.nonzero().flatten()
            assert len(eligible)>0
            ids.append(eligible[torch.randint(len(eligible),(PER_STRATUM,),generator=gen,device="cuda")])
        ids=torch.cat(ids)
        values=grid.four(ids,include_distance=include_distance)
        # Store score/color statistics in float32, preserving the low-contrast comparisons.
        bank[group].append((*values,truth.flatten()[ids]))
        if (index+1)%24==0:
            print("Encoded source",index+1,flush=True)
    assert len(bank)==9 and sum(map(len,bank.values()))==96
    return list(bank.values()),source_hash


def fit():
    configure_cuda()
    plan=protocol()
    assert (ROOT/"selection.json").exists()
    tic=monotonic()
    bank,source_hash=source_bank()
    receipts=[]
    for seed in SEEDS:
        set_seed(seed)
        rng=np.random.default_rng(seed+10200)
        generator=torch.Generator(device="cuda").manual_seed(seed+10200)
        models={m:Strength(m).cuda() for m in MODES}
        opt=torch.optim.AdamW([p for n in models.values() for p in n.parameters()],lr=.035,weight_decay=0)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,320,eta_min=.0035)
        traces={m:[] for m in MODES}
        for step in range(320):
            batch=[]
            for stratum in (0,1):
                group=bank[rng.integers(len(bank))]
                frame=group[rng.integers(len(group))]
                chosen=torch.randint(PER_STRATUM,(2048,),generator=generator,device="cuda")+PER_STRATUM*stratum
                batch.append([v[chosen] for v in frame])
            *features,y=(torch.cat(v) for v in zip(*batch))
            opt.zero_grad(set_to_none=True)
            losses=[]
            for mode,net in models.items():
                loss=F.cross_entropy(net(features),y)
                losses.append(loss)
                traces[mode].append(float(loss.detach()))
            sum(losses).backward()
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            opt.step()
            schedule.step()
        for mode,model in models.items():
            path=ROOT/f"{mode}-{seed}.pth"
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch_save(path,state)
            receipts.append(dict(mode=mode,seed=seed,steps=320,weights=path.name,sha256=digest(path),
                                 fitted={k:float(v) for k,v in state.items()},loss=traces[mode]))
        print("Fitted three matched arms",seed,flush=True)
    atomic_write(ROOT/"training.json",dict(source_frames=96,source_recordings=9,records=receipts,
                 protocol=digest(plan),selection=digest(ROOT/"selection.json"),source_manifest=digest(DATA/"fit.json"),
                 lidseg=AUTHOR_HASH,lidseg_inference_code=source_hash,trainer=digest(__file__),
                 operator=digest(Path(__file__).with_name("native_guide_v102.py")),cuda=torch.version.cuda,
                 pytorch=str(torch.__version__),gpu=torch.cuda.get_device_name(),seconds=monotonic()-tic,finished_utc=now()))


if __name__=="__main__":
    fit()

"""Single-Gaussian controls using exactly the candidate's four bilinear donors."""
from boundary_refinement.paths import SOURCE_ROOT
from pathlib import Path
from time import monotonic
import numpy as np
import torch
import torch.nn.functional as F

from boundary_refinement.training.fit_native_guide import PER_STRATUM,source_bank
from boundary_refinement.data.cohorts.native_reserve import ROOT,SEEDS
from boundary_refinement.data.provenance import digest,now,read
from boundary_refinement.models.operators.range_kernel import RangeDecoder
from boundary_refinement.training.train_regional_guidance import atomic_write,configure_cuda,torch_save
from boundary_refinement.utils import set_seed


def scores(packed,net=None):
    prior,_,_,donors,distance=packed
    if net is None:
        exponent,bandwidth=1.,.05
    else:
        exponent=.25+1.5*net.spatial.sigmoid()
        bandwidth=.005+.495*net.width.sigmoid()
    valid=prior.isfinite()
    energy=prior.masked_fill(~valid,0)*exponent-.5*distance/bandwidth**2
    plan=energy.masked_fill(~valid,-torch.inf).softmax(-1)
    return torch.bmm(plan[:,None],donors).squeeze(1)


def fit():
    configure_cuda()
    addition=ROOT/"matched-controls.json"
    if not addition.exists():
        assert not (ROOT/"counts").exists()
        atomic_write(addition,dict(parent=digest(ROOT/"protocol.json"),before_target_scoring_utc=now(),
                     reason="Directly test whether the color weighting explains an advantage on identical four donors.",
                     arms=["matched_gaussian_fixed","matched_gaussian_fit"],fit="Fixed sigma .05 and spatial exponent 1, "
                     "or fit bandwidth and exponent on precisely the same source pixels, seeds and 320 updates. "
                     "Compare class_gap to both as additional secondary evidence. The original primary tests remain fixed."))
    start=monotonic()
    bank,source_hash=source_bank(include_distance=True)
    receipts=[]
    for seed in SEEDS:
        set_seed(seed)
        rng=np.random.default_rng(seed+10200)
        generator=torch.Generator(device="cuda").manual_seed(seed+10200)
        net=RangeDecoder("bilinear_gaussian").cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.035,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,320,eta_min=.0035)
        trace=[]
        for step in range(320):
            batch=[]
            for stratum in (0,1):
                group=bank[rng.integers(len(bank))]
                frame=group[rng.integers(len(group))]
                ids=torch.randint(PER_STRATUM,(2048,),generator=generator,device="cuda")+PER_STRATUM*stratum
                batch.append([v[ids] for v in frame])
            *features,y=(torch.cat(v) for v in zip(*batch))
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(scores(features,net),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            trace.append(float(loss.detach()))
        path=ROOT/f"matched_gaussian_fit-{seed}.pth"
        torch_save(path,{k:v.detach().cpu() for k,v in net.state_dict().items()})
        receipts.append(dict(mode="matched_gaussian_fit",seed=seed,steps=320,weights=path.name,sha256=digest(path),loss=trace))
    atomic_write(ROOT/"gaussian-training.json",dict(records=receipts,addition=digest(addition),trainer=digest(__file__),
                 bank_code=digest((SOURCE_ROOT / "boundary_refinement/training/fit_native_guide.py")),operator=digest((SOURCE_ROOT / "boundary_refinement/models/operators/native_guide.py")),
                 initial_training=digest(ROOT/"training.json"),source_inference=source_hash,finished=now(),seconds=monotonic()-start))
    print("Three same-support Gaussian fits complete",flush=True)


if __name__=="__main__":
    fit()

"""Fit a single additional coefficient with every manuscript coefficient frozen."""
from time import monotonic
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scripts import path_decoder_v104 as model
from scripts import train_path_v104 as prior
from scripts import semantic_subpixel_v94 as manuscript
from scripts.reserve_anchor_v105 import OUT
from scripts.weatherproof_release_v103 import read,write

MODES=("endpoint","path","shifted","permuted")
class Anchor(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        self.register_buffer("weights",(3*checkpoint.range_weights.detach().sigmoid()).sum(0).clone())
        self.register_buffer("exponent",(.25+1.5*checkpoint.temperature.detach().sigmoid()).clone())

    def forward(self,features):
        b,d=features[:2];bands=d.new_tensor(model.RANGES)
        range_cost=(-torch.expm1(-.5*d[...,None]/bands.square()))@self.weights
        return b.log()*self.exponent-range_cost

def cues(feat):
    return {"endpoint":-torch.expm1(-.5*feat[1]/.10**2),"path":feat[2],"shifted":feat[3],"permuted":feat[2].roll(1,-1)}

def train():
    prior.source.configure_cuda()
    assert (OUT/"bdd-reserve.json").exists() and (OUT/"wp-reserve.json").exists()
    tic=monotonic();old=manuscript.load();bank,rows,backbone=prior.pool();runs=[]
    for seed in model.SEEDS:
        prior.source.set_seed(seed);rng=np.random.default_rng(seed+10400)
        gen=torch.Generator(device="cuda").manual_seed(seed+10400)
        frozen=Anchor(old["dual_core",seed]).cuda()
        free=nn.Parameter(torch.full((4,),float(np.log(.4/(6-.4))),device="cuda"))
        opt=torch.optim.AdamW([free],lr=.035,weight_decay=0);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,320,eta_min=.0035)
        log=[]
        for i in range(320):
            parts=[]
            for j in (0,1):
                group=bank[rng.integers(len(bank))];frame=group[rng.integers(len(group))]
                ix=j*prior.PER+torch.randint(prior.PER,(2048,),generator=gen,device="cuda")
                parts.append([x[ix] for x in frame])
            *feat,label=(torch.cat(xs) for xs in zip(*parts))
            fixed=frozen(feat);penalty=cues(feat);opt.zero_grad(set_to_none=True);losses=[]
            for j,mode in enumerate(MODES):
                weights=(fixed-6*free[j].sigmoid()*penalty[mode]).softmax(-1)
                scores=(weights[:,None]@feat[-1]).squeeze(1)
                losses.append(F.cross_entropy(scores,label))
            loss=torch.stack(losses);loss.sum().backward()
            assert torch.isfinite(free.grad).all()
            free.grad.clamp_(-5,5);opt.step();sched.step();log.append(loss.detach())
        traces=torch.stack(log).cpu().T.tolist()
        for j,mode in enumerate(MODES):
            path=OUT/f"anchor-{mode}-{seed}.pth";strength=6*free[j].detach().sigmoid()
            prior.source.torch_save(path,dict(range_weights=frozen.weights.cpu(),spatial=frozen.exponent.cpu(),strength=strength.cpu()))
            runs.append(dict(mode=mode,seed=seed,additional_trainable_parameters=1,steps=320,weights=path.name,
                    sha256=prior.source.digest(path),extra_strength=float(strength),losses=traces[j]))
        print("Completed four frozen-manuscript fits",seed,flush=True)
    assert rows==read(prior.OUT/"training.json")["sampling"]
    write(OUT/"training.json",dict(runs=runs,source_frames=len(rows),source_recordings=len(bank),source_bank_matches_v104=True,
        prior_training=prior.source.digest(prior.OUT/"training.json"),source_backbone=backbone,
        original_paper=prior.source.digest(manuscript.ROOT/"completed.json"),trainer=prior.source.digest(__file__),
        protocol=prior.source.digest(OUT/"protocol.json"),bdd_reserve=prior.source.digest(OUT/"bdd-reserve.json"),
        wp_reserve=prior.source.digest(OUT/"wp-reserve.json"),torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),seconds=monotonic()-tic,utc=prior.source.now()))

def load():
    old=manuscript.load();anchors={s:Anchor(old["dual_core",s]).cuda() for s in model.SEEDS};strength={}
    card=read(OUT/"training.json")
    for r in card["runs"]:
        f=OUT/r["weights"];assert prior.source.digest(f)==r["sha256"];state=torch.load(f,map_location="cuda",weights_only=True)
        a=anchors[r["seed"]];assert torch.equal(a.weights,state["range_weights"]) and torch.equal(a.exponent,state["spatial"])
        strength[r["mode"],r["seed"]]=state["strength"]
    return old,anchors,strength

if __name__=="__main__":train()

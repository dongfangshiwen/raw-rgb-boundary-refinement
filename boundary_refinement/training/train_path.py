"""Fit six controlled kernels against the same held-in MUSES fog children."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
import torch.nn.functional as F
from boundary_refinement.training import fit_native_guide as source
from boundary_refinement.models.operators import path_kernel as model
from boundary_refinement.data.cohorts.holdout_path import OUT
from boundary_refinement.data.staging.weatherproof_files import read,write

PER=3072

@torch.no_grad()
def pool():
    author,network,signature=source.author_loader()
    network.eval()
    mean,std=(torch.tensor(x,device="cuda")[None,:,None,None] for x in (author.MEAN,author.STD))
    groups=defaultdict(list); evidence=[]
    for key,group,_,_,truth,rgb in source.paths("fit"):
        padded,(h,w)=source.peer_input(rgb); small=padded[...,:h,:w].contiguous()
        z=F.interpolate(network.predict((small-mean)/std).float(),(h//4,w//4),mode="bilinear",align_corners=False)
        st=model.Stencil(z,small);shape=(4*(h//4),4*(w//4))
        nearest=lambda x:F.interpolate(x[None,None].float(),shape,mode="nearest-exact").flatten()
        labels=nearest(truth).long()
        valid=(labels<19)&st.active
        boundary=nearest(source.boundary_band(truth)).bool()&valid
        gen=torch.Generator(device="cuda").manual_seed(int.from_bytes(sha256(("path104/"+key).encode()).digest()[:4],"big"))
        chosen=[]
        for sel in (valid,boundary):
            eligible=sel.nonzero().flatten()
            if not len(eligible):eligible=valid.nonzero().flatten()
            assert len(eligible)>0
            chosen.append(eligible[torch.randint(len(eligible),(PER,),generator=gen,device="cuda")])
        ids=torch.cat(chosen)
        groups[group].append((*st.features(ids),labels[ids]))
        evidence.append(dict(id=key,recording=group,active_valid=int(valid.sum()),active_boundary=int(boundary.sum()),sample_hash=sha256(ids.cpu().numpy().tobytes()).hexdigest()))
        if len(evidence)%24==0:print("Encoded",len(evidence),"fog frames",flush=True)
    assert len(groups)==9 and len(evidence)==96
    return list(groups.values()),evidence,signature

def train():
    source.configure_cuda()
    assert (OUT/"protocol.json").exists() and (OUT/"reserve.json").exists()
    start=monotonic();bank,rows,signature=pool();runs=[]
    for seed in model.SEEDS:
        source.set_seed(seed);rng=np.random.default_rng(seed+10400)
        generator=torch.Generator(device="cuda").manual_seed(seed+10400)
        nets={k:model.Decoder(k).cuda() for k in model.MODES}
        opt=torch.optim.AdamW([a for net in nets.values() for a in net.parameters()],lr=.035,weight_decay=0)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,320,eta_min=.0035)
        losses={k:[] for k in nets}
        for step in range(320):
            batch=[]
            for part in (0,1):
                g=bank[rng.integers(len(bank))];frame=g[rng.integers(len(g))]
                ix=PER*part+torch.randint(PER,(2048,),generator=generator,device="cuda")
                batch.append([a[ix] for a in frame])
            *features,truth=(torch.cat(t) for t in zip(*batch))
            opt.zero_grad(set_to_none=True)
            ls=[]
            for k,net in nets.items():
                loss=F.cross_entropy(net(features),truth);ls.append(loss)
                losses[k].append(loss.detach())
            sum(ls).backward()
            for net in nets.values():torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            opt.step();sched.step()
        for k,net in nets.items():
            name=f"{k}-{seed}.pth"
            source.torch_save(OUT/name,{a:b.detach().cpu() for a,b in net.state_dict().items()})
            runs.append(dict(mode=k,seed=seed,weights=name,sha256=source.digest(OUT/name),coefficients=net.coefficients(),losses=torch.stack(losses[k]).cpu().tolist(),parameters=sum(x.numel() for x in net.parameters()),steps=320))
        print("Completed six paired source fits",seed,flush=True)
    write(OUT/"training.json",dict(runs=runs,sampling=rows,source_frames=96,source_recordings=9,
        protocol=source.digest(OUT/"protocol.json"),reserve=source.digest(OUT/"reserve.json"),source_manifest=source.digest(source.DATA/"fit.json"),
        trainer=source.digest(__file__),operator=source.digest(model.__file__),source_model=signature,
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,seconds=monotonic()-start,utc=source.now()))

def load():
    card=read(OUT/"training.json");assert card["operator"]==source.digest(model.__file__)
    nets={}
    for r in card["runs"]:
        p=OUT/r["weights"];assert r["sha256"]==source.digest(p)
        net=model.Decoder(r["mode"]);net.load_state_dict(torch.load(p,map_location="cpu",weights_only=True))
        nets[r["mode"],r["seed"]]=net.cuda().eval().requires_grad_(False)
    return nets

if __name__=="__main__":train()

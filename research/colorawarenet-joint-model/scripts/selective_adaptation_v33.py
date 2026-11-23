"""Train stage-matched parameter subsets of the same PGL-Net-T checkpoint."""
import argparse
import math
import time

import torch
from torch import nn
from torch.nn.utils import parametrize

from dehaze_seg.utils import set_seed
from scripts.author_pgl_v25 import load_pgl
from scripts.crop_context_v32 import ROOT as PREVIOUS, batches, crop_forward, originals
from scripts.projective_ledger_v12 import digest, now, read
from scripts.source_replay_v32 import band_parts
from scripts.stage_layer_holdouts_v33 import ROOT
from scripts.stage_lmhaze_v31 import TRAIN_CARD
from scripts.structure_transfer_v32 import PAIRS, SEEDS, SIDE, STEPS, objective, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ARMS = ("full", "affine", "local_matched", "local_lora")
MAIN = "affine"
MASK_SEED = 33043


class TrainCoordinates(nn.Module):
    def __init__(self, weight, ids):
        super().__init__()
        indices = ids.to(weight.device)
        self.register_buffer("indices",indices)
        self.active = nn.Parameter(weight.detach().flatten()[indices].clone())

    def forward(self, original):
        return original.flatten().index_copy(0,self.indices,self.active).view_as(original)


class LocalLoRA(nn.Module):
    def __init__(self, weight, rank):
        super().__init__()
        outputs, inputs = weight.shape[:2]
        if weight.shape[2:] != (1,1):
            raise ValueError("LoRA control uses square channel-mixing convolutions")
        self.down = nn.Parameter(weight.new_empty((rank,inputs)))
        self.up = nn.Parameter(weight.new_zeros((outputs,rank)))
        nn.init.kaiming_uniform_(self.down,a=math.sqrt(5))

    def forward(self, original):
        with torch.autocast("cuda",enabled=False):
            return original+(self.up@self.down).view_as(original)


def layout(model):
    plan = []
    for index, unit in enumerate(model.fusions):
        count = sum(p.numel() for p in unit.parameters())
        decoder = model.stage_num-index-1
        prefix = f"layers.{decoder}."
        local = {}
        points = []
        for key, candidate in model.named_modules():
            if not key.startswith(prefix) or not isinstance(candidate, nn.Conv2d):
                continue
            if ".conv.Wv." in key or ".conv.proj." in key:
                local.update({f"{key}.{n}":p.numel() for n,p in candidate.named_parameters(recurse=False)})
            if key.endswith((".conv.Wv.0",".conv.proj.pw")):
                if candidate.weight.shape[:2] != (unit.dim,unit.dim):
                    raise ValueError("Mismatched LoRA control channel width")
                points.append(key)
        sizes = sorted(local.items())
        total = sum(local.values())
        assigned = {name:count*n//total for name,n in sizes}
        remaining = count-sum(assigned.values())
        by_remainder = sorted(local,key=lambda name:(-(count*local[name]%total),name))
        for name in by_remainder[:remaining]:
            assigned[name] += 1
        rank_count, remainder = divmod(count,2*unit.dim)
        if remainder or not points or total<count:
            raise ValueError("Could not match local capacity at this scale")
        per_layer, bonus = divmod(rank_count,len(points))
        ranks = {name:per_layer+int(i<bonus) for i,name in enumerate(sorted(points))}
        if not 0<per_layer or max(ranks.values())>unit.dim:
            raise ValueError("Infeasible low-rank control")
        plan.append(dict(fusion=f"fusions.{index}",decoder=prefix.rstrip("."),width=unit.dim,
                         budget=count,local_sizes=local,coordinates=assigned,lora_ranks=ranks))
    return plan


def adapt(model,arm,plan):
    model.train().requires_grad_(arm=="full")
    for child in model.modules():
        if isinstance(child,nn.BatchNorm2d):
            child.eval()
    selected = {}
    if arm=="affine":
        model.fusions.requires_grad_(True)
    elif arm=="local_matched":
        rng = torch.Generator().manual_seed(MASK_SEED)
        for scale in plan:
            for name,n in sorted(scale["coordinates"].items()):
                prefix,attr = name.rsplit(".",1)
                module = model.get_submodule(prefix)
                weight = getattr(module,attr)
                ids = torch.randperm(weight.numel(),generator=rng)[:n].sort().values
                parametrize.register_parametrization(module,attr,TrainCoordinates(weight,ids))
                selected[name] = ids
    elif arm=="local_lora":
        for scale in plan:
            for name,rank in scale["lora_ranks"].items():
                module = model.get_submodule(name)
                parametrize.register_parametrization(module,"weight",LocalLoRA(module.weight,rank))
                selected[name+".weight"] = None
    return selected


def materialize(model,selected):
    for key in selected:
        prefix,name = key.rsplit(".",1)
        parametrize.remove_parametrizations(model.get_submodule(prefix),name,leave_parametrized=True)
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def verify_changes(before,after,param_names,selected,arm):
    if after.keys()!=before.keys():
        raise ValueError("An adapter remained in the deployed state")
    changes = 0
    for key,old in before.items():
        if arm=="full":
            revised = key in param_names
        elif arm=="affine":
            revised = key in param_names and key.startswith("fusions.")
        else:
            revised = key in selected
        unequal = (after[key]!=old).flatten()
        if not revised and unequal.any():
            raise ValueError(f"Frozen parameter or buffer changed at {key}")
        if arm=="local_matched" and revised:
            unequal[selected[key]] = False
            if unequal.any():
                raise ValueError(f"A coordinate outside the registered subset changed at {key}")
        if revised:
            changes += int((after[key]!=old).sum())
    return changes


def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    source,lm = originals(),training_scenes()
    example,author = load_pgl()
    stages = layout(example)
    initial = {k:v.detach().cpu().clone() for k,v in example.state_dict().items()}
    param_names = set(dict(example.named_parameters()))
    target_count = sum(s["budget"] for s in stages)
    plan = dict(methods=list(ARMS),main=MAIN,seeds=list(SEEDS),steps=STEPS,side=SIDE,views_per_step=3*PAIRS,
                per_stage=stages,matched_parameters=target_count,mask_seed=MASK_SEED,author=author,
                source_SHA256=digest(__file__),lm_sha=digest(TRAIN_CARD),previous_recipe=digest(PREVIOUS/"protocol.json"),
                hypothesis="At frozen BN and the same source replay, adapting the existing global physics-inspired affine skip filters can preserve more original-domain quality than an equal number of local decoder parameters while adapting to LMHaze.",
                controls="Full training reproduces V32 frozen BN and exactly the same sampler, initialization, source objective and optimizer. The two local controls match the affine count at each decoder scale. The coordinate control chooses a fixed uniform subset of local value-convolution parameters, including depthwise and pointwise values. The local LoRA control uses zero-initialized residuals on local 1x1 channel-mixing weights and merges before inference.",
                evaluation="All three fixed seeds evaluate the final update. Compare old LMHaze and RW2AH populations and the remaining RW2AH capture codes with masked duplicates. One new HD-NH-HAZE subset adds external transfer evidence when staged. Use the author checkpoint as a separate reference. Leave source weights untouched outside each declared subspace, including all BN buffers.")
    protocol = ROOT/"protocol.json"
    if protocol.exists() and read(protocol)!=plan:
        raise ValueError("Selective adaptation protocol changed")
    if not protocol.exists():
        atomic_write(protocol,plan)
    signature = digest(protocol)
    ledger = []
    for seed in SEEDS:
        for arm in ARMS:
            path = ROOT/f"{arm}-{seed}.pth"
            receipt = ROOT/f"{arm}-{seed}.json"
            if receipt.exists():
                known = read(receipt)
                if known["protocol_SHA256"]!=signature or known["checkpoint_SHA256"]!=digest(path):
                    raise ValueError("Frozen model or plan changed")
                ledger.append(known)
                continue
            model,_ = load_pgl()
            set_seed(seed)
            selected = adapt(model,arm,stages)
            before = {k:v.detach().cpu() for k,v in model.state_dict().items()} if not selected else None
            if before is not None and any(not torch.equal(v,initial[k]) for k,v in before.items()):
                raise ValueError("Different starting weights")
            params = [p for p in model.parameters() if p.requires_grad]
            count = sum(p.numel() for p in params)
            if count!=(author["params"] if arm=="full" else target_count):
                raise ValueError("Trainable parameter count changed")
            optim = torch.optim.AdamW(params,lr=.00012,weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim,STEPS,eta_min=.00002)
            started = time.perf_counter()
            for step,(x,context,y,z,full) in enumerate(batches(lm,source,seed),1):
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    prediction = crop_forward(model,x,context,"frozen_bn",full).float()
                adapt_out,source_out = prediction.split((2*PAIRS,PAIRS))
                target_obj,_ = objective(adapt_out,y,None,"point")
                low,high = band_parts(source_out-z)
                total = target_obj+(low.square()+1e-6).sqrt().mean()+.2*high.abs().mean()
                total.backward()
                torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
                optim.step()
                schedule.step()
            if step!=STEPS:
                raise ValueError("Optimization budget changed")
            weights = materialize(model,selected)
            changed = verify_changes(initial,weights,param_names,selected,arm)
            torch_save(path,weights)
            row = dict(arm=arm,seed=seed,updates=step,trainable=count,changed=changed,file=str(path),
                       loss=total.item(),protocol_SHA256=signature,checkpoint_SHA256=digest(path),
                       seconds=round(time.perf_counter()-started,2),finished_utc=now())
            atomic_write(receipt,row)
            print({k:row[k] for k in ("arm","seed","updates","trainable","changed","loss","seconds")},flush=True)
            ledger.append(row)
            del model, params, optim, schedule
    result = dict(runs=ledger,updates=sum(x["updates"] for x in ledger),protocol_SHA256=signature,
                  gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),finished_utc=now())
    atomic_write(ROOT/"completed.json",result)
    print(dict(runs=len(ledger),updates=result["updates"]),flush=True)


def completed():
    path = ROOT/"completed.json"
    value = read(path)
    if value["protocol_SHA256"]!=digest(ROOT/"protocol.json"):
        raise ValueError("Selective adaptation plan changed")
    if {(r["arm"],r["seed"]) for r in value["runs"]}!={(arm,seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("Not all matched runs completed")
    for row in value["runs"]:
        if digest(row["file"])!=row["checkpoint_SHA256"]:
            raise ValueError("Selective adaptation weight changed")
    return value


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    train()

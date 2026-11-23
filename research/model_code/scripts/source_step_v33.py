"""Compare ordinary replay with source-gradient and AdamW-step projection."""
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from dehaze_seg.utils import set_seed, write_csv
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.crop_context_v32 import batches, crop_forward, originals
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_selective_adaptation_v33 import samples
from scripts.evaluate_structure_transfer_v32 import measured
from scripts.projective_ledger_v12 import digest, now, read
from scripts.selective_adaptation_v33 import ROOT as V33, completed, verify_changes
from scripts.source_replay_v32 import band_parts
from scripts.structure_transfer_v32 import PAIRS, SEEDS, STEPS, load_fit, objective, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = V33/"source-step"
METHODS = ("gradient_projection","global_step","branch_step")


def branches(model,names):
    grouped = defaultdict(list)
    for position,key in enumerate(names):
        if key.startswith("fusions."):
            what = "affine"
        elif key.startswith("layers.") and int(key.split(".")[1])>model.half_num:
            what = "local_decoder"
        else:
            what = "other_layers"
        grouped[what].append(position)
    return list(grouped.values())


def vector(items, ids):
    return torch.cat([items[i].reshape(-1) for i in ids])


def projected(move, direction):
    dot = move.dot(direction)
    step = move-(dot.clamp_min(0)/direction.dot(direction).clamp_min(1e-18))*direction
    return step,dot


@torch.no_grad()
def set_vector(parameters, indexes, values, gradients=False):
    for i,value in zip(indexes,values.split([parameters[j].numel() for j in indexes]),strict=True):
        destination = parameters[i].grad if gradients else parameters[i]
        destination.copy_(value.view_as(destination))


def update(model, optimizer, params, groups, batch, method):
    fog,thumb,target,source,full = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        restored = crop_forward(model,fog,thumb,"frozen_bn",full).float()
    pred, replay = restored.split((2*PAIRS,PAIRS))
    dest,_ = objective(pred,target,None,"point")
    low,high = band_parts(replay-source)
    retention = (low.square()+1e-6).sqrt().mean()+.2*high.abs().mean()
    source_grads = torch.autograd.grad(retention,params,retain_graph=True)
    loss = dest+retention
    loss.backward()
    # Adam has already preconditioned and accumulated raw gradients by the time
    # a step is proposed. Keep its moments, then constrain the actual displacement.
    references = [vector(source_grads,ids) for ids in groups]
    if method=="gradient_projection":
        ids, = groups
        gradient,dot = projected(-vector([p.grad for p in params],ids),references[0])
        set_vector(params,ids,-gradient,gradients=True)
    before = [vector(params,ids).detach() for ids in groups]
    torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
    optimizer.step()
    applied, prior = [],[]
    for old,reference,ids in zip(before,references,groups,strict=True):
        with torch.no_grad():
            movement = vector(params,ids)-old
            if method=="gradient_projection":
                prior.append(dot)
            else:
                movement,dot = projected(movement,reference)
                set_vector(params,ids,old+movement)
                prior.append(dot)
            applied.append((vector(params,ids)-old).dot(reference))
    return loss.detach(),torch.stack(prior),torch.stack(applied)


def fit():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    original = completed()
    source,lm = originals(),training_scenes()
    first,author = load_pgl()
    initial = {k:v.detach().cpu().clone() for k,v in first.state_dict().items()}
    names = list(dict(first.named_parameters()))
    roles = branches(first,names)
    grouped_counts = [sum(dict(first.named_parameters())[names[i]].numel() for i in group) for group in roles]
    plan = dict(code_SHA256=digest(__file__),parent_SHA256=digest(V33/"completed.json"),author=author,
                seeds=list(SEEDS),updates=STEPS,methods=list(METHODS),module_counts=grouped_counts,
                claim="Source preservation may depend on the AdamW parameter displacement, whose direction can differ from the raw gradient. Compare A-GEM-style raw-gradient projection, post-step global projection and architecture-group projection on the same full model.",
                settings="Same three fixed seeds, training order, learning rate, loss structure and fixed BN as V33 full. Use current minibatch source gradients. Only project positive first-order source loss increase. Keep Adam's pre-projection moments. Group projection measures affine skip layers, local decoder blocks and remaining parameters independently.",
                assessment="V33 results motivated the follow-up. Holdout from the current fit is guaranteed by the existing manifests. Scores on repeated LMHaze and RW2AH populations are diagnostic. No claim of external safety or mathematical quality bounds.")
    protocol = ROOT/"protocol.json"
    if protocol.exists() and read(protocol)!=plan:
        raise ValueError("Source-step protocol changed")
    if not protocol.exists():
        atomic_write(protocol,plan)
    runs = []
    for seed in SEEDS:
        for method in METHODS:
            path = ROOT/f"{method}-{seed}.pth"
            receipt = ROOT/f"{method}-{seed}.json"
            if receipt.exists():
                row = read(receipt)
                if row["protocol_SHA256"]!=digest(protocol) or row["checkpoint_SHA256"]!=digest(path):
                    raise ValueError("A source-projected model changed")
                runs.append(row)
                continue
            model,_ = load_pgl()
            set_seed(seed)
            model.train().requires_grad_(True)
            for child in model.modules():
                if isinstance(child,nn.BatchNorm2d):
                    child.eval()
            params = list(model.parameters())
            groups = roles if method=="branch_step" else [list(range(len(params)))]
            optimizer = torch.optim.AdamW(params,lr=.00012,weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=.00002)
            observed, applied = [],[]
            for step,batch in enumerate(batches(lm,source,seed),1):
                loss,proposal,actual = update(model,optimizer,params,groups,batch,method)
                schedule.step()
                observed.append(proposal)
                applied.append(actual)
            state = {key:value.detach().cpu() for key,value in model.state_dict().items()}
            verify_changes(initial,state,set(names),{},"full")
            torch_save(path,state)
            row = dict(arm=method,seed=seed,updates=step,loss=loss.item(),file=str(path),checkpoint_SHA256=digest(path),
                       protocol_SHA256=digest(protocol),positive_proposals=int((torch.stack(observed)>1e-8).sum()),
                       positive_applied=int((torch.stack(applied)>1e-8).sum()),groups=len(groups),finished_utc=now())
            atomic_write(receipt,row)
            print({key:row[key] for key in ("arm","seed","updates","positive_proposals","positive_applied")},flush=True)
            runs.append(row)
    atomic_write(ROOT/"completed.json",dict(runs=runs,updates=sum(r["updates"] for r in runs),
                                           protocol_SHA256=digest(protocol),baseline_full=original["protocol_SHA256"],finished_utc=now()))


@torch.inference_mode()
def evaluate():
    configure_cuda()
    card = read(ROOT/"completed.json")
    if card["protocol_SHA256"]!=digest(ROOT/"protocol.json"):
        raise ValueError("Source-step model protocol changed")
    plain = [r for r in completed()["runs"] if r["arm"]=="full"]
    nets = {(r["arm"],r["seed"]):load_fit(load_pgl()[0],r) for r in (*plain,*card["runs"])}
    nets["pgl",0] = load_pgl()[0]
    scores = defaultdict(list)
    same_groups = defaultdict(lambda:defaultdict(list))
    rows = []
    picked = (r for r in samples("development") if r[0]=="lmhaze")
    for i,(d,s,g,x,y) in enumerate((*picked,*samples("fresh_rw")),1):
        a,b = as_float(x),as_float(y)
        per_image = defaultdict(list)
        for (method,seed),net in nets.items():
            stat = measured(pgl_rgb(net,a).clamp(0,1),b)
            rows.append(dict(dataset=d,scene=s,group=g,method=method,seed=seed,**stat))
            per_image[method].append([stat["psnr"],stat["ssim"],stat["edge_mae"]])
        means = {m:np.mean(v,axis=0) for m,v in per_image.items()}
        for method,v in means.items():
            scores[d,method].append(v)
        for m in METHODS:
            same_groups[d,m][g].append(means[m][0]-means["full"][0])
        if i%40==0:
            print(dict(scored=i),flush=True)
    summary = {f"{d}/{m}":dict(zip(("psnr","ssim","edge_mae"),np.mean(v,axis=0).tolist())) for (d,m),v in scores.items()}
    table = ROOT/"test.csv"
    write_csv(table,rows)
    atomic_write(ROOT/"test.json",dict(summary=summary,fit_SHA256=digest(ROOT/"completed.json"),rows=len(rows),
                                     csv_SHA256=digest(table),scores_by_group={f"{d}/{m}":bootstrap({g:np.mean(v) for g,v in group.items()})
                                     for (d,m),group in same_groups.items()},finished_utc=now()))
    print({k:round(v["psnr"],4) for k,v in summary.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("fit","test"))
    fit() if parser.parse_args().action=="fit" else evaluate()

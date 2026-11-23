"""Isolate normalization and image context during matched crop adaptation."""
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dehaze_seg.utils import set_seed, write_csv
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_structure_transfer_v32 import data_split, measured
from scripts.projective_ledger_v12 import digest, now, read
from scripts.source_replay_v32 import ROOT as REPLAY, band_parts, cohorts
from scripts.stage_preview_training_v28 import STAGED
from scripts.structure_transfer_v32 import ROOT as V32, PAIRS, SEEDS, SIDE, STEPS
from scripts.structure_transfer_v32 import load_fit, objective, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = V32/"crop-context"
ARMS = ("frozen_bn","shuffled_context","aligned_context")
MAIN = "aligned_context"


def originals():
    card = read(STAGED)
    plan = read(REPLAY/"protocol.json")
    if digest(card["file"])!=card["SHA256"] or digest(STAGED)!=plan["rw_source"]:
        raise ValueError("Source replay training pixels changed")
    ids = set(plan["rw_fit_ids"])
    data = [torch.stack((r["hazy"],r["clean"])).cuda()
            for r in torch.load(card["file"],map_location="cpu",weights_only=True) if r["id"] in ids]
    if len(data)!=len(ids):
        raise ValueError("Source replay population changed")
    return data


def view(images,rng,full):
    h,w=images.shape[-2:]
    scaled = images.float()/255
    if full or min(h,w)<SIDE:
        patch = F.interpolate(scaled,(SIDE,SIDE),mode="bilinear",align_corners=False,antialias=True)
    else:
        top = int(torch.randint(h-SIDE+1,(),generator=rng))
        left = int(torch.randint(w-SIDE+1,(),generator=rng))
        patch = scaled[...,top:top+SIDE,left:left+SIDE]
    whole = F.interpolate(scaled[:-1],(96,96),mode="bilinear",align_corners=False,antialias=True)
    flip = bool(torch.rand((),generator=rng)<.5)
    return (patch.flip(-1),whole.flip(-1)) if flip else (patch,whole)


def order(n,rng):
    need = STEPS*PAIRS
    return torch.cat([torch.randperm(n,generator=rng) for _ in range((need+n-1)//n)])[:need].split(PAIRS)


def batches(lm,rw,seed):
    r_lm = torch.Generator().manual_seed(seed+8900)
    r_rw = torch.Generator().manual_seed(seed+14450)
    for step,(lm_ids,rw_ids) in enumerate(zip(order(len(lm),r_lm),order(len(rw),r_rw),strict=True)):
        full = step%3==0
        lm_views = []
        for i in lm_ids.tolist():
            row = lm[i]
            a = int(torch.randint(4,(),generator=r_lm))
            b = 4+int(torch.randint(4,(),generator=r_lm))
            lm_views.append(view(torch.stack((row["frames"][a],row["frames"][b],row["clear"])),r_lm,full))
        rw_views = [view(rw[i],r_rw,full) for i in rw_ids.tolist()]
        va,ca = (torch.stack(v).transpose(0,1).contiguous() for v in zip(*lm_views))
        vb,cb = (torch.stack(v).transpose(0,1).contiguous() for v in zip(*rw_views))
        input = torch.cat((*va[:2],vb[0]))
        context = torch.cat((*ca,cb[0]))
        yield input,context,va[2],vb[1],full


def crop_forward(model,patch,full_image,arm,full):
    if full:
        return model(patch)
    # Capture only the actual skip-affine pooling outputs. Their labels and dimensions
    # match the later crop call. The small image pass is shared across all controls.
    modules = [block.avg_pool for block in model.fusions]
    recorded=[]

    def observe(_module,_inputs,result):
        recorded.append(result.detach())

    handles = [layer.register_forward_hook(observe) for layer in modules]
    try:
        with torch.no_grad():
            model(full_image)
    finally:
        for handle in handles:
            handle.remove()
    if arm=="frozen_bn":
        return model(patch)
    if arm=="shuffled_context":
        recorded = [torch.cat([part.roll(1,0) for part in value.split(PAIRS)]) for value in recorded]
    used = iter(recorded)

    def replace(_module,_inputs,result):
        context = next(used)
        if result.shape!=context.shape:
            raise ValueError("Whole-frame context does not match its crop")
        return context

    handles = [layer.register_forward_hook(replace) for layer in modules]
    try:
        prediction = model(patch)
        if next(used,None) is not None:
            raise ValueError("Skip-affine pooling count changed")
        return prediction
    finally:
        for handle in handles:
            handle.remove()


def fit():
    configure_cuda()
    ROOT.mkdir(exist_ok=True,parents=True)
    source,lm = originals(),training_scenes()
    model,author = load_pgl()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prior = digest(REPLAY/"protocol.json")
    plan = dict(source_SHA256=digest(__file__),author=author,prior_replay=prior,seeds=list(SEEDS),arms=list(ARMS),steps=STEPS,
                hypothesis="Author skip-affine blocks summarize each training crop differently from an image. On crop steps use pooled statistics from the same full image through a detached 96x96 pass. This changes training context only. Test at original benchmark geometry with stock inference.",
                controls="Freeze BN running state in every arm, while affine scales and model parameters remain trainable. Identical paired ground-truth objective, views and optimizer. The unknown-information placebo rolls global context between images within each LMHaze intensity or RW²AH subgroup, using the same thumbnail computation. Earlier ordinary replay is reported separately.",
                scope="Design informed by the post-hoc BN swap on validation and the older RW²AH reserve. Held benchmark scores here are repeated diagnostics. No earlier test pixels enter training.")
    card=ROOT/"protocol.json"
    if card.exists() and read(card)!=plan:
        raise ValueError("Crop context protocol differs")
    if not card.exists():
        atomic_write(card,plan)
    rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            path=ROOT/f"{arm}-{seed}.pth"
            receipt=ROOT/f"{arm}-{seed}.json"
            if receipt.exists():
                row=read(receipt)
                if digest(path)!=row["checkpoint_SHA256"] or digest(card)!=row["protocol_SHA256"]:
                    raise ValueError("Stored context run has changed")
                rows.append(row)
                continue
            set_seed(seed)
            model.load_state_dict(initial,strict=True)
            model.train().requires_grad_(True)
            for layer in model.modules():
                if isinstance(layer,nn.BatchNorm2d):
                    layer.eval()
            opt=torch.optim.AdamW(model.parameters(),lr=.00012,weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00002)
            for step,(x,context,y,z,full) in enumerate(batches(lm,source,seed),1):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    adapted=crop_forward(model,x,context,arm,full).float()
                target,source_out=adapted.split((2*PAIRS,PAIRS))
                dest,_=objective(target,y,None,"point")
                low,high=band_parts(source_out-z)
                loss=dest+(low.square()+1e-6).sqrt().mean()+.2*high.abs().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                opt.step()
                schedule.step()
            torch_save(path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            row=dict(arm=arm,seed=seed,updates=step,checkpoint_SHA256=digest(path),file=str(path),
                     protocol_SHA256=digest(card),loss=loss.item(),finished_utc=now())
            atomic_write(receipt,row)
            print(dict(arm=arm,seed=seed,loss=row["loss"],updates=step),flush=True)
            rows.append(row)
    atomic_write(ROOT/"completed.json",dict(runs=rows,steps=sum(r["updates"] for r in rows),protocol_SHA256=digest(card),finished_utc=now()))


@torch.inference_mode()
def test():
    configure_cuda()
    fits=read(ROOT/"completed.json")
    if fits["protocol_SHA256"]!=digest(ROOT/"protocol.json"):
        raise ValueError("The context protocol changed")
    runs=fits["runs"]+[r for r in read(REPLAY/"completed.json")["runs"] if r["arm"]=="replay_gt"]
    models={(r["arm"],r["seed"]):load_fit(load_pgl()[0],r) for r in runs}
    models["pgl",0]=load_pgl()[0]
    rows=[]
    old=defaultdict(dict)
    groups=defaultdict(lambda:defaultdict(list))
    for i,(dataset,scene,group,x,y) in enumerate((*data_split("lmhaze"),*cohorts("confirmation"))):
        a,b=as_float(x),as_float(y)
        for (method,seed),net in models.items():
            values=measured(pgl_rgb(net,a).clamp(0,1),b)
            rows.append(dict(dataset=dataset,scene=scene,group=group,method=method,seed=seed,**values))
            old[dataset,group,scene].setdefault(method,[]).append(values["psnr"])
        if (i+1)%40==0:
            print(dict(scored=i+1),flush=True)
    metrics=defaultdict(list)
    for (dataset,group,_),measurements in old.items():
        for method,values in measurements.items():
            metrics[dataset,method].append(np.mean(values))
        for ref in ("frozen_bn","shuffled_context","replay_gt"):
            groups[dataset,ref][group].append(np.mean(measurements[MAIN])-np.mean(measurements[ref]))
    file=ROOT/"scores.csv"
    write_csv(file,rows)
    summary={f"{data}/{arm}":float(np.mean(v)) for (data,arm),v in metrics.items()}
    paired={f"{d}/{c}":bootstrap({group:np.mean(x) for group,x in v.items()}) for (d,c),v in groups.items()}
    atomic_write(ROOT/"test.json",dict(psnr=summary,intervals=paired,score_SHA256=digest(file),
                                     weights_SHA256=digest(ROOT/"completed.json"),rows=len(rows),finished_utc=now()))
    print(summary,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",choices=("fit","test"))
    fit() if parser.parse_args().step=="fit" else test()

"""Separate ordinary source replay from local, frequency-specific source retention."""
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed, write_csv
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_post_fusion_v30 import samples as v30_samples
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_structure_transfer_v32 import data_split, measured
from scripts.projective_ledger_v12 import digest, now, read
from scripts.stage_lmhaze_v31 import CARD as LMHaze, TRAIN_CARD
from scripts.stage_postfusion_holdout_v30 import camera_code
from scripts.stage_preview_training_v28 import STAGED
from scripts.stage_rw_confirmation_v32 import CARD as FRESH
from scripts.structure_transfer_v32 import ROOT as RGB_ROOT, PAIRS, SEEDS, SIDE, STEPS, completed
from scripts.structure_transfer_v32 import detail, load_fit, objective, scene_stream, training_scenes
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = RGB_ROOT / "source-retention"
ARMS = ("replay_gt","replay_global","replay_local","replay_shifted")
MAIN = "replay_local"
PROTOCOL = ROOT/"protocol.json"


@torch.no_grad()
def source(teacher):
    registered = read(STAGED)
    if digest(registered["file"])!=registered["SHA256"]:
        raise ValueError("Original RW training images changed")
    consumed = []
    for row in torch.load(registered["file"],map_location="cpu",weights_only=True):
        if camera_code(row["id"]) in ("3","4","5"):
            continue
        fog,clear = (row[k].cuda() for k in ("hazy","clean"))
        pred = pgl_rgb(teacher,fog[None].float()/255)[0].clamp(0,1)
        consumed.append((row["id"],torch.stack((fog,clear,(255*pred).round().byte()))))
    if len(consumed)!=86:
        raise ValueError("Unexpected in-domain replay population")
    return consumed


def source_views(data,seed):
    rng = torch.Generator().manual_seed(seed+14450)
    selection = torch.cat([torch.randperm(len(data),generator=rng)
                           for _ in range((STEPS*PAIRS+len(data)-1)//len(data))])[:STEPS*PAIRS]
    for step, indices in enumerate(selection.split(PAIRS)):
        outputs = []
        for i in indices.tolist():
            x = data[i][1]
            h,w = x.shape[-2:]
            if step%3 == 0 or min(h,w)<SIDE:
                value = F.interpolate(x.float()/255,size=(SIDE,SIDE),mode="bilinear",align_corners=False,antialias=True)
            else:
                top = int(torch.randint(h-SIDE+1,(),generator=rng))
                left = int(torch.randint(w-SIDE+1,(),generator=rng))
                value = x[...,top:top+SIDE,left:left+SIDE].float()/255
            if bool(torch.rand((),generator=rng)<.5):
                value = value.flip(-1)
            outputs.append(value)
        yield torch.stack(outputs).transpose(0,1).contiguous().unbind(0)


def band_parts(x):
    high = detail(x)
    return x-high,high


@torch.no_grad()
def source_target(image,clear,teacher,method):
    targets = []
    for raw,gt,frozen,floor in zip(band_parts(image),band_parts(clear),band_parts(teacher),(.03,.008),strict=True):
        if method=="replay_gt":
            targets.append(gt)
            continue

        def energy(x):
            return F.avg_pool2d(x.square().mean(1,keepdim=True),9,1,4,count_include_pad=False).sqrt()

        before = energy(raw-gt)
        after = energy(frozen-gt)
        reliability = ((before-after)/(before+floor)).clamp(0,1)/(1+after/floor)
        if method=="replay_global":
            reliability = reliability.mean((-2,-1),keepdim=True)
        if method=="replay_shifted":
            reliability = reliability.roll((SIDE//2,SIDE//2),(-2,-1))
        targets.append(torch.lerp(gt,frozen,reliability))
    return targets


def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    model,author = load_pgl()
    rw = source(model)
    prefix = {k.split("-",1)[0] for k,_ in rw}
    lm = training_scenes()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    plan = dict(source_SHA256=digest(__file__),backbone=author,lm_manifest=digest(TRAIN_CARD),rw_source=digest(STAGED),
                arms=list(ARMS),main=MAIN,seeds=list(SEEDS),steps=STEPS,rw_fit_ids=[id for id,_ in rw],
                source_prefixes=sorted(prefix),training="Follow the identical LMHaze crop sequence and RGB/detail objective of the matched pointwise study. Add 3 original-domain RW²AH samples at the same crop size each step. The source and target losses each have an independent per-pixel mean and equal coefficients.",
                idea="Construct two source replay targets from the fixed pretrained teacher and the RW clean reference. The mixing strength is per band and locally depends on the teacher reducing error versus the original input. GT-only, spatially uniform equal weight per image, and half-grid rolled weights are the matched controls. Teacher is cached at the evaluation scale and absent at inference.",
                scope="The repeated LMHaze and previous RW reserve outcomes inform the design. Reserve fresh RW²AH capture codes before local model evaluation. The author may have prior exposure to this test archive. Group the inference comparison by filename camera prefix.")
    if PROTOCOL.exists() and read(PROTOCOL)!=plan:
        raise ValueError("Source retention protocol changed")
    if not PROTOCOL.exists():
        atomic_write(PROTOCOL,plan)
    key = digest(PROTOCOL)
    rows = []
    for seed in SEEDS:
        for method in ARMS:
            weight = ROOT/f"{method}-{seed}.pth"
            receipt = ROOT/f"{method}-{seed}.json"
            if receipt.exists():
                row = read(receipt)
                if row["checkpoint_SHA256"]!=digest(weight) or row["protocol_SHA256"]!=key:
                    raise ValueError("Stored source-retention weight changed")
                rows.append(row)
                continue
            set_seed(seed)
            model.load_state_dict(initial,strict=True)
            model.train().requires_grad_(True)
            opt = torch.optim.AdamW(model.parameters(),lr=.00012,weight_decay=.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00002)
            for i,((x_light,x_dense,y),(sx,sy,st)) in enumerate(zip(scene_stream(lm,seed),source_views(rw,seed),strict=True),1):
                opt.zero_grad(set_to_none=True)
                low,high = source_target(sx,sy,st,method)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    output = model(torch.cat((x_light,x_dense,sx))).float()
                target_out,source_out = output.split((2*PAIRS,PAIRS))
                l1,_ = objective(target_out,y,None,"point")
                low_pred,high_pred = band_parts(source_out)
                source_loss = ((low_pred-low).square()+1e-6).sqrt().mean()+.2*(high_pred-high).abs().mean()
                (l1+source_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
                opt.step()
                schedule.step()
            torch_save(weight,{k:v.detach().cpu() for k,v in model.state_dict().items()})
            row = dict(arm=method,seed=seed,updates=i,file=str(weight),checkpoint_SHA256=digest(weight),
                       protocol_SHA256=key,losses=[l1.item(),source_loss.item()],finished_utc=now())
            atomic_write(receipt,row)
            print(dict(arm=method,seed=seed,steps=i,losses=row["losses"]),flush=True)
            rows.append(row)
    atomic_write(ROOT/"completed.json",dict(runs=rows,updates=sum(r["updates"] for r in rows),plan_SHA256=key,finished_utc=now()))


def cohorts(split):
    if split=="confirmation":
        fresh = read(FRESH)
        if digest(fresh["file"])!=fresh["SHA256"]:
            raise ValueError("RW confirmation inputs changed")
        for row in torch.load(fresh["file"],map_location="cpu",weights_only=True):
            id = row["id"]
            yield "rw_confirmation",id,camera_code(id),row["hazy"],row["clean"]
        return
    yield from data_split("lmhaze")
    for dataset,scene,group,x,y,_ in v30_samples("rw_reserve"):
        yield dataset,scene,group,x,y


@torch.inference_mode()
def evaluate(split):
    configure_cuda()
    original = load_pgl()[0]
    runs = read(ROOT/"completed.json")
    if runs["plan_SHA256"]!=digest(PROTOCOL):
        raise ValueError("Replay protocol changed")
    base = [r for r in completed()["runs"] if r["arm"]=="point"]
    models = { (r["arm"],r["seed"]): load_fit(load_pgl()[0],r) for r in (*runs["runs"],*base)}
    models["pgl",0] = original
    ledger, index = [],defaultdict(dict)
    for i,(dataset,scene,group,raw,clear) in enumerate(cohorts(split)):
        x,y = as_float(raw),as_float(clear)
        for (method,seed),net in models.items():
            metric = measured(pgl_rgb(net,x).clamp(0,1),y)
            ledger.append(dict(dataset=dataset,scene=scene,group=group,method=method,seed=seed,**metric))
            index[dataset,group,scene,method][seed] = metric
        if (i+1)%30==0:
            print(dict(split=split,paired=i+1),flush=True)
    summary, differences = defaultdict(list),defaultdict(lambda:defaultdict(list))
    for (dataset,group,scene,name),versions in index.items():
        metrics = np.mean([[v[k] for k in ("psnr","ssim","edge_mae")] for v in versions.values()],axis=0)
        summary[dataset,name].append(metrics)
        if name==MAIN:
            for control in ("replay_gt","replay_global","replay_shifted","pgl"):
                others = index[dataset,group,scene,control]
                difference = metrics[0]-np.mean([r["psnr"] for r in others.values()])
                differences[dataset,control][group].append(difference)
    result = {f"{d}/{m}":dict(zip(("psnr","ssim","edge_mae"),np.mean(v,axis=0).tolist())) for (d,m),v in summary.items()}
    table = ROOT/(split+"_scores.csv")
    write_csv(table,ledger)
    atomic_write(ROOT/(split+".json"),dict(scores=result,table_SHA256=digest(table),policy_SHA256=digest(ROOT/"completed.json"),
                                         input_SHA256=digest(FRESH if split=="confirmation" else LMHaze),
                                         intervals={f"{d}/{m}":bootstrap({g:np.mean(x) for g,x in batches.items()})
                                                    for (d,m),batches in differences.items()},finished_utc=now()))
    print({k:round(v["psnr"],4) for k,v in result.items()},flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",choices=("fit","development","confirmation"))
    args = parser.parse_args()
    if args.step=="fit":
        train()
    else:
        evaluate(args.step)

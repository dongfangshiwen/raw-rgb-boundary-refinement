"""Test reference-free spatial fusion of two frozen, complementary restorers."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.utils import set_seed, write_csv
from boundary_refinement.models.restoration.appearance_refiner import appearance_loss
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.evaluation.photographic.evaluate_author_mct import load_author_reference
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import TESTS, as_float, as_rgb, samples
from boundary_refinement.evaluation.photographic.evaluate_parallel_rgb import bind as bind_rgb
from boundary_refinement.evaluation.paired.evaluate_real_transfer import METRICS, quality
from boundary_refinement.data.staging.prepare_rw2ah import ROOT as V25
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import DATASETS, RealPairs
from boundary_refinement.data.real_haze_data import load_data
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO/"results/expert-fusion-v26"
GATES = ("global_rgb","local_rgb","local_physics","local_projected")
SEEDS = (26003,26017,26021)
H,W = 192,256
BATCH = 6
EPOCHS = 18


def experts():
    bind_rgb()
    mct,mct_info = load_author_reference()
    pgl,pgl_info = load_pgl()
    return (partial(author_rgb,mct),partial(pgl_rgb,pgl)), dict(mct=mct_info,pgl=pgl_info)


def matched_view(images):
    """Resize a list of aligned tensors to a common grid with replicated borders."""
    h,w = images[0].shape[-2:]
    scale = min(H/h,W/w)
    rows,cols = round(h*scale),round(w*scale)
    top,left = (H-rows)//2,(W-cols)//2
    border = (left,W-cols-left,top,H-rows-top)
    valid = F.pad(torch.ones((len(images[0]),1,rows,cols),device=images[0].device),border).bool()
    resized = [F.pad(F.interpolate(v,(rows,cols),mode="bilinear",align_corners=False,antialias=True),border,
                     mode="replicate") for v in images]
    return resized,valid


def physical_cues(raw,mct,pgl):
    batch = len(raw)
    y = raw.mean(1,keepdim=True)
    dy = F.pad((y[...,1:,:]-y[...,:-1,:]).abs(),(0,0,0,1),mode="replicate")
    dx = F.pad((y[...,:,1:]-y[...,:,:-1]).abs(),(0,1,0,0),mode="replicate")
    order = (y-(dx+dy)).flatten(1).topk(max(1,y[0].numel()//50),dim=-1).indices
    ambient = raw.flatten(2).gather(-1,order[:,None].expand(batch,3,-1)).mean(-1)[...,None,None]
    observer = raw-ambient
    components = []
    for estimate in (mct,pgl):
        clear = estimate-ambient
        transmission = ((observer*clear).sum(1,keepdim=True)/clear.square().sum(1,keepdim=True).clamp_min(.0001)).clamp(0,1)
        residual = (observer-transmission*clear).abs().mean(1,keepdim=True)
        components.extend((transmission,residual))
    return torch.cat(components,dim=1)


def features(raw,mct,pgl,physical):
    resized = [F.avg_pool2d(v,4,4) for v in (raw,mct,pgl)]
    x,a,b = resized
    if physical:
        extras = physical_cues(x,a,b)
    else:
        diff_a = a-x
        diff_b = b-x
        extras = torch.cat((diff_a.mean(1,keepdim=True),diff_a.abs().mean(1,keepdim=True),
                            diff_b.mean(1,keepdim=True),diff_b.abs().mean(1,keepdim=True)),dim=1)
    return torch.cat((*resized,(a-b).abs(),extras),dim=1)


class Gate(nn.Module):
    def __init__(self,method):
        super().__init__()
        self.method = method
        self.map = nn.Sequential(nn.Conv2d(16,24,5,padding=2),nn.SiLU(),
                                 nn.Conv2d(24,24,3,padding=2,dilation=2),nn.SiLU(),
                                 nn.Conv2d(24,24,3,padding=3,dilation=3),nn.SiLU(),
                                 nn.Conv2d(24,1,1))
        nn.init.zeros_(self.map[-1].weight)
        nn.init.zeros_(self.map[-1].bias)

    def weights(self,raw,mct,pgl,valid):
        value = self.map(features(raw,mct,pgl,self.method in ("local_physics","local_projected"))).float()
        if self.method=="global_rgb":
            mask = F.avg_pool2d(valid.float(),4,4)
            value = (value*mask).sum((-2,-1),keepdim=True)/mask.sum((-2,-1),keepdim=True).clamp_min(1)
        return value.sigmoid()


def weighted_mix(mct,pgl,weight):
    weight = F.interpolate(weight,mct.shape[-2:],mode="bilinear",align_corners=False)
    return torch.lerp(pgl,mct,weight)


def projected_loss(pred,mct,pgl,ref,valid):
    difference = mct-pgl
    mask = valid.float()
    denominator = F.avg_pool2d(difference.square().sum(1,keepdim=True)*mask,4,4)
    numerator = F.avg_pool2d(((ref-pgl)*difference).sum(1,keepdim=True)*mask,4,4)
    optimal = (numerator/denominator.clamp_min(1e-6)).clamp(0,1)
    return ((pred-optimal).square()*denominator).sum()/denominator.sum().clamp_min(1e-6)


@torch.no_grad()
def training_data(forward):
    manifest = read(V25/"data_manifest.json")
    second = manifest["train"]
    if digest(second["file"])!=second["SHA256"]:
        raise ValueError("RW2AH train cache no longer matches")
    nh = load_data("training")["training"]
    rw = torch.load(second["file"],map_location="cpu",weights_only=True)
    if len(nh)!=45 or len(rw)!=second["count"]:
        raise ValueError("Incorrect registered fusion train population")
    outputs = []
    for row in nh+rw:
        left,top,right,bottom = row["roi"]
        raw,ref = (as_float(row[k][:,top:bottom,left:right]) for k in ("hazy","clean"))
        # Infer without letterbox borders. A 2x view reduces scale disparity to the full-size evaluator.
        whole = F.interpolate(raw,scale_factor=2,mode="bilinear",align_corners=False)
        a,b = (f(whole).clamp(0,1) for f in forward)
        packed,valid = matched_view((whole,a,b,F.interpolate(ref,whole.shape[-2:],mode="bilinear",align_corners=False)))
        outputs.append((*packed,valid))
    return tuple(torch.cat(v) for v in zip(*outputs))


def training_plan(registration):
    file = ROOT/"plan.json"
    design = dict(arms=list(GATES),seeds=list(SEEDS),epochs=EPOCHS,batch=BATCH,paired_training=[45,96],
                  weights=registration,
                  source_SHA256={str(p):digest(p) for p in (Path(__file__),V25/"data_manifest.json",
                       SOURCE_ROOT / "boundary_refinement/baselines/pgl_net.py",SOURCE_ROOT / "boundary_refinement/models/restoration/appearance_refiner.py")},
                  fit="Only the gates train on NH 01–45 and the registered RW2AH subsets. Both external experts are frozen. Shared shuffle and flip, CUDA BF16 and appearance loss. Local projected adds .05 times the weighted patchwise distance to the best convex RGB mixture.",
                  control="Learned global and local use one architecture. Local RGB and physics have 16 input channels and the same trainable weights count. Physics features fit a common bright-low-gradient airlight estimate and candidate transmission. No metadata or clean photo is available at inference.",
                  assessment="Reuse V25's 175 pairs for exploratory comparison. Single experts and fixed arithmetic blend run on exactly the same inputs. A reference-dependent oracle reports potential headroom only. Assess an exploratory, domain-equal average over five test sets.",
                  caution="This fusion was designed after the V25 comparisons. Published experts have different training history and test-domain exposures. These repeated comparisons do not establish independent external validation.")
    if file.exists() and read(file)!=design:
        raise ValueError("Fusion training plan changed")
    if not file.exists():
        atomic_write(file,design)
    return digest(file)


def fit(data,seed,plan):
    artifact = ROOT/f"gate-{seed}.pth"
    receipt = ROOT/f"gate-{seed}.json"
    if receipt.exists():
        row = read(receipt)
        if row["plan_SHA256"]!=plan or digest(artifact)!=row["checkpoint_SHA256"]:
            raise ValueError("Completed gate changed")
        return row
    model = {}
    for method in GATES:
        set_seed(seed)
        model[method] = Gate(method).cuda()
    opt = {k:torch.optim.AdamW(v.parameters(),lr=.001,weight_decay=.01) for k,v in model.items()}
    schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(o,EPOCHS,eta_min=1e-4) for o in opt.values()]
    rng = torch.Generator().manual_seed(seed)
    count = len(data[0])
    started = time.perf_counter()
    steps = 0
    history = []
    for epoch in range(EPOCHS):
        losses = []
        for ids in torch.randperm(count,generator=rng).split(BATCH):
            tiles = [v[ids] for v in data]
            flip = (torch.rand(len(ids),generator=rng)<.5).cuda()[:,None,None,None]
            image,a,b,ref,valid = [torch.where(flip,x.flip(-1),x) for x in tiles]
            each = []
            for name,gate in model.items():
                opt[name].zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    w = gate.weights(image,a,b,valid)
                    y = weighted_mix(a,b,w)
                    loss = appearance_loss(y,ref,valid[:,0])
                    if name=="local_projected":
                        loss = loss + .05*projected_loss(w,a,b,ref,valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(),1,error_if_nonfinite=True)
                opt[name].step()
                each.append(loss.detach())
            losses.append(torch.stack(each))
            steps += 1
        for s in schedules:
            s.step()
        history.append(torch.stack(losses).mean(0).tolist())
        if (epoch+1)%6==0:
            print(dict(seed=seed,epoch=epoch+1,steps=steps,losses=dict(zip(GATES,history[-1]))),flush=True)
    torch_save(artifact,{n:m.state_dict() for n,m in model.items()})
    row = dict(seed=seed,plan_SHA256=plan,checkpoint=str(artifact),checkpoint_SHA256=digest(artifact),steps=steps,
               params={n:sum(x.numel() for x in m.parameters()) for n,m in model.items()},losses=history,
               took_seconds=time.perf_counter()-started,finished_utc=now())
    atomic_write(receipt,row)
    return row


def trained_gates():
    complete = read(ROOT/"completed.json")
    if complete["plan_SHA256"]!=digest(ROOT/"plan.json"):
        raise ValueError("Fusion plan differs from its training receipt")
    gates = {}
    for r in complete["runs"]:
        if digest(r["checkpoint"])!=r["checkpoint_SHA256"]:
            raise ValueError("Fusion checkpoint differs")
        states = torch.load(r["checkpoint"],map_location="cpu",weights_only=True)
        gates[r["seed"]] = {}
        for method in GATES:
            model = Gate(method)
            model.load_state_dict(states[method],strict=True)
            gates[r["seed"]][method] = model.cuda().eval()
    return gates


def infer_gate(gate,raw,a,b):
    (small,first,second),valid = matched_view((raw,a,b))
    weights = gate.weights(small,first,second,valid)
    if weights.shape[-2:]==(1,1):
        native = weights
    else:
        h,w = raw.shape[-2:]
        scale = min(H/h,W/w)
        rows,cols = round(h*scale),round(w*scale)
        top,left = (H-rows)//2,(W-cols)//2
        native = F.interpolate(weights,(H,W),mode="bilinear",align_corners=False)[...,top:top+rows,left:left+cols]
    return weighted_mix(a,b,native),native


def oracle_scalar(a,b,target):
    gap = a-b
    alpha = ((target-b)*gap).sum()/gap.square().sum().clamp_min(1e-9)
    return torch.lerp(b,a,alpha.clamp(0,1))


@torch.inference_mode()
def assess(forward):
    by_seed = trained_gates()
    first_seed = next(iter(by_seed))
    manifest = read(V25/"data_manifest.json")
    rows, images, by_method = [], [], defaultdict(list)
    with RealPairs(DATASETS) as registry:
        for dataset in TESTS:
            count = 0
            for scene,hazy,clear in samples(registry,dataset,manifest):
                x,y = as_float(hazy),as_float(clear)
                mct,pgl = (f(x).clamp(0,1) for f in forward)
                keep = dataset in ("rw2ah","dense") and count==0
                preview = {"hazy":as_rgb(x),"clear":as_rgb(y)} if keep else {}
                data = {"author_mct":mct,"author_pgl":pgl,"equal_blend":.5*(mct+pgl),
                        "scalar_oracle":oracle_scalar(mct,pgl,y)}

                def record(method,pred,seed,weight=None):
                    scores = quality(pred,y)[0]
                    row = dict(dataset=dataset,scene=scene,method=method,seed=seed,
                               alpha_mct=None if weight is None else float(weight.mean()),
                               **dict(zip(METRICS,scores)))
                    rows.append(row)
                    by_method[dataset,method,seed].append(scores)
                    if keep and seed in (-1,first_seed) and method!="scalar_oracle":
                        preview[method] = as_rgb(pred)

                for name,value in data.items():
                    record(name,value,-1)
                for seed,gates in by_seed.items():
                    for name,gate in gates.items():
                        rgb,weight = infer_gate(gate,x,mct,pgl)
                        record(name,rgb,seed,weight)
                        if name=="global_rgb":
                            record("global_hard",mct if float(weight)>=.5 else pgl,seed,weight)
                if keep:
                    images.append((dataset,scene,preview))
                count += 1
            print(dict(dataset=dataset,pairs=count),flush=True)
    if len(rows)!=175*(4+5*len(SEEDS)):
        raise ValueError("Fusion evaluation pair count differs")
    averages = defaultdict(list)
    for (data,method,_), entries in by_method.items():
        averages[data,method].append(np.mean(entries,axis=0))
    summary = {f"{d}/{m}":dict(zip(METRICS,np.mean(ss,axis=0).tolist())) for (d,m),ss in averages.items()}
    meta = ROOT/"paired_metrics.csv"
    write_csv(meta,rows)
    display = ("hazy","author_mct","author_pgl","clear","equal_blend","global_rgb","local_rgb","local_physics","local_projected")
    canvas = Image.new("RGB",(1200,280*3*len(images)),"#f2f4f6")
    draw = ImageDraw.Draw(canvas)
    saved = []
    for i,(dataset,scene,data) in enumerate(images):
        for j,name in enumerate(display):
            image = data[name]
            file = ROOT/f"{dataset}-{scene}-{name}.png"
            image.save(file)
            saved.append(dict(file=str(file),SHA256=digest(file)))
            image.thumbnail((393,248),Image.Resampling.LANCZOS)
            left,top = j%3*400,(i*3+j//3)*280
            canvas.paste(image,(left+(400-image.width)//2,top+22))
            draw.text((left+8,top+4),f"{dataset} {scene} | {name}",fill="#143142")
    canvas.save(ROOT/"comparison.png")
    macro = {name:float(np.mean([summary[f"{d}/{name}"]["psnr"] for d in TESTS]))
             for name in sorted({m for _,m in averages})}
    report = dict(scores=summary,macro_PSNR=macro,table_rows=len(rows),table_SHA256=digest(meta),
                  plan_SHA256=digest(ROOT/"plan.json"),training_SHA256=digest(ROOT/"completed.json"),
                  examples=saved,comparison_SHA256=digest(ROOT/"comparison.png"),finished_utc=now())
    atomic_write(ROOT/"evaluation.json",report)
    print(dict(rows=len(rows),macro_PSNR=macro),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate",action="store_true")
    args = parser.parse_args()
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    forward,registration = experts()
    if args.evaluate:
        assess(forward)
        return
    plan = training_plan(registration)
    data = training_data(forward)
    runs = [fit(data,seed,plan) for seed in SEEDS]
    atomic_write(ROOT/"completed.json",dict(runs=runs,fits=len(runs)*len(GATES),
                 updates=sum(r["steps"] for r in runs)*len(GATES),plan_SHA256=plan,finished_utc=now()))
    del data
    assess(forward)


if __name__=="__main__":
    main()

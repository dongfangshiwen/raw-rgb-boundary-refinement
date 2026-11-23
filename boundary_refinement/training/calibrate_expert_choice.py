"""Calibrate a conservative per-image expert switch with grouped train predictions."""
from boundary_refinement.paths import SOURCE_ROOT
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from boundary_refinement.utils import write_csv
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import TESTS, as_float, as_rgb, samples
from boundary_refinement.evaluation.paired.evaluate_real_transfer import METRICS, quality
from boundary_refinement.models.restoration.expert_fusion import experts, matched_view, physical_cues, training_data
from boundary_refinement.data.staging.prepare_rw2ah import ROOT as V25
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_detail_data import DATASETS, RealPairs
from boundary_refinement.data.real_haze_data import load_data
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO/"results/guarded-expert-v27"
MODES = ("input_sign","pair_sign","lower_bound","support","guarded")


def moments(value,valid):
    area = valid.sum((-2,-1)).clamp_min(1)
    mean = (value*valid).sum((-2,-1))/area
    variance = ((value-mean[...,None,None]).square()*valid).sum((-2,-1))/area
    return torch.cat((mean,variance.clamp_min(1e-8).sqrt()),1)


def descriptor(raw,mct,pgl,valid):
    x,a,b = (F.avg_pool2d(v,4,4) for v in (raw,mct,pgl))
    mask = F.avg_pool2d(valid.float(),4,4)
    weights = raw.new_tensor([.2126,.7152,.0722])[None,:,None,None]

    def content(image):
        gray = (image*weights).sum(1,keepdim=True)
        saturation = image.amax(1,keepdim=True)-image.amin(1,keepdim=True)
        dx = F.pad(gray.diff(dim=-1).abs(),(0,1,0,0),mode="replicate")
        dy = F.pad(gray.diff(dim=-2).abs(),(0,0,0,1),mode="replicate")
        return moments(torch.cat((image,saturation,dx+dy),1),mask)

    base = content(x)
    evidence = torch.cat((base,content(a),content(b),moments(a-x,mask),moments(b-x,mask),
                          moments((a-b).abs(),mask),moments(physical_cues(x,a,b),mask)),1)
    return base,evidence


def ridge(x,gain):
    center = x.mean(0)
    scale = x.std(0,unbiased=False).clamp_min(.003)
    standard = (x-center)/scale
    offset = gain.mean()
    penalty = 24*torch.eye(x.shape[1],device=x.device)
    coef = torch.linalg.solve(standard.T@standard+penalty,standard.T@(gain-offset))
    return dict(center=center,scale=scale,offset=offset,coef=coef)


def regression(model,x):
    return ((x-model["center"])/model["scale"])@model["coef"]+model["offset"]


def fit(forward):
    raw,m,p,ref,valid = training_data(forward)
    a,b = descriptor(raw,m,p,valid)
    error = [((v-ref).square()*valid).flatten(1).sum(1).clamp_min(1e-8) for v in (m,p)]
    gain = 10*(error[0]/error[1]).log10()
    nh = load_data("training")["training"]
    cache = read(V25/"data_manifest.json")["train"]["file"]
    rw = torch.load(cache,map_location="cpu",weights_only=True)
    group = [f"NH-{(int(v['id'])-1)//15}" for v in nh] + ["RW-"+v["id"].split("-")[0] for v in rw]
    models = dict(input=ridge(a,gain),paired=ridge(b,gain))
    normalize = lambda v: (v-models["paired"]["center"])/models["paired"]["scale"]
    donor = gain>.75
    normalized = normalize(b)
    outside, density = torch.empty_like(gain),torch.full_like(gain,float("nan"))
    for label in sorted(set(group)):
        held = torch.tensor([v==label for v in group],device="cuda")
        model = ridge(b[~held],gain[~held])
        outside[held] = regression(model,b[held])
        own, other = held&donor, ~held&donor
        if own.any() and other.any():
            density[own] = torch.cdist(normalized[own],normalized[other]).min(1).values/normalized.shape[1]**.5
    if not torch.isfinite(outside).all() or not torch.isfinite(density[donor]).all():
        raise ValueError("Grouped calibration has insufficient independent donors")
    bound = (outside-gain).quantile(.80).clamp_min(0)
    radius = density[donor].quantile(.85)
    models |= dict(margin=bound,radius=radius,donors=normalized[donor])
    registration = dict(groups={k:group.count(k) for k in sorted(set(group))},better_PGL_train=int(donor.sum()),
                        n=len(group),training_group_oof_MAE_dB=float((outside-gain).abs().mean()),
                        margin_dB=float(bound),support_radius=float(radius),fit_device=str(a.device),
                        choices="Only NH 01–45 and the registered RW2AH training prefix groups. Grouped held-out regressions supply the 80th percentile of advantage overestimation. Radius is the 85th percentile of nearest other-training-group distance among PGL gains >0.75 dB.",
                        variants="Input-only or paired descriptor sign, gain lower estimate only, training support only, and intersection. Default expert is MCT. Support statistics use training data only.")
    torch_save(ROOT/"calibration.pth",models)
    atomic_write(ROOT/"calibration.json",registration)
    print(registration,flush=True)
    return models


def switches(model,raw,mct,pgl):
    (x,a,b),valid = matched_view((raw,mct,pgl))
    raw_features,pair = descriptor(x,a,b,valid)
    score = regression(model["paired"],pair)[0]
    normalized = (pair-model["paired"]["center"])/model["paired"]["scale"]
    gap = torch.cdist(normalized,model["donors"]).min()/normalized.shape[1]**.5
    estimate = bool(score>0)
    lower = bool(score>model["margin"])
    support = bool(gap<=model["radius"])
    return dict(input_sign=bool(regression(model["input"],raw_features)[0]>0),pair_sign=estimate,
                lower_bound=lower,support=estimate and support,guarded=lower and support)


@torch.inference_mode()
def main():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    forward,registration = experts()
    plan_path = ROOT/"plan.json"
    design = dict(source_SHA256={str(p):digest(p) for p in (Path(__file__),V25/"data_manifest.json",
                      SOURCE_ROOT / "boundary_refinement/models/restoration/expert_fusion.py")}, frozen_experts=registration, variants=list(MODES),
                  scope="Exploratory follow-up to V26. The previously inspected five benchmark datasets are reused. Train-only calibration with holdings grouped by NH ID ranges and RW2AH filename prefixes. No dataset ID is an input.")
    if plan_path.exists() and read(plan_path)!=design:
        raise ValueError("The V27 calibration plan changed")
    if not plan_path.exists():
        atomic_write(plan_path,design)
    model = fit(forward)
    rows, previews, picks = [], [], {}
    with RealPairs(DATASETS) as registry:
        for dataset in TESTS:
            choices = dict.fromkeys(MODES,0)
            count = 0
            for scene,hazy,clear in samples(registry,dataset,read(V25/"data_manifest.json")):
                x,y = as_float(hazy),as_float(clear)
                a,b = (f(x).clamp(0,1) for f in forward)
                result = switches(model,x,a,b)
                scores = {"author_mct":quality(a,y)[0],"author_pgl":quality(b,y)[0]}
                for name,s in scores.items():
                    rows.append(dict(dataset=dataset,scene=scene,method=name,used_pgl=int(name=="author_pgl"),
                                     **dict(zip(METRICS,s))))
                for name,selected in result.items():
                    choices[name] += selected
                    rows.append(dict(dataset=dataset,scene=scene,method=name,used_pgl=int(selected),
                                     **dict(zip(METRICS,scores["author_pgl" if selected else "author_mct"]))))
                if count==0 and dataset in ("rw2ah","dense"):
                    previews.append((dataset,scene,[as_rgb(v) for v in (x,a,b,b if result["guarded"] else a,y)]))
                count += 1
            picks[dataset] = dict(n=count,PGL=choices)
            print(dict(dataset=dataset,pairs=count,PGL=choices),flush=True)
    if len(rows)!=175*(len(MODES)+2):
        raise ValueError("Incorrect calibration assessment count")
    csvfile = ROOT/"paired_metrics.csv"
    write_csv(csvfile,rows)
    score = {f"{d}/{m}":float(np.mean([r["psnr"] for r in rows if r["dataset"]==d and r["method"]==m]))
             for d in TESTS for m in (*MODES,"author_mct","author_pgl")}
    macro = {m:float(np.mean([score[f"{d}/{m}"] for d in TESTS])) for m in (*MODES,"author_mct","author_pgl")}
    preview = Image.new("RGB",(1750,len(previews)*268),"#f4f6f7")
    draw = ImageDraw.Draw(preview)
    for i,(d,scene,values) in enumerate(previews):
        for j,(name,value) in enumerate(zip(("Hazy","MCT","PGL-Net-T","Guarded selection","Clear reference"),values)):
            value.thumbnail((346,239),Image.Resampling.LANCZOS)
            x,y = j*350,i*268
            preview.paste(value,(x+(350-value.width)//2,y+27))
            draw.text((x+4,y+5),f"{d} {scene} | {name}",fill="#112738")
    preview.save(ROOT/"comparison.png")
    receipt = dict(scores=score,macro_PSNR=macro,counts=picks,table_SHA256=digest(csvfile),rows=len(rows),
                   calibration_SHA256=digest(ROOT/"calibration.pth"),plan_SHA256=digest(plan_path),
                   comparison_SHA256=digest(ROOT/"comparison.png"),finished_utc=now())
    atomic_write(ROOT/"evaluation.json",receipt)
    print(dict(rows=len(rows),macro_PSNR=macro),flush=True)


if __name__=="__main__":
    main()

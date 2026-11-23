"""Fit a preview-only routing policy to paired, evaluator-size expert gains."""
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from boundary_refinement.utils import write_csv
from boundary_refinement.training.calibrate_expert_choice import descriptor, moments
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.models.restoration.expert_fusion import experts, matched_view
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_detail_data import RealPairs, image_tensor, long_edge
from boundary_refinement.data.staging.stage_preview_training import ROOT, STAGED, stage
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


LEVELS = (128, 256)
HARM = 3.0
PENALTY = 1.0
RESIDUAL_QUANTILE = .80
FEATURES = {
    "raw_mean": ("raw", False),
    "preview_mean": ("preview", False),
    "preview_tail": ("preview", True),
    "scales_mean": ("scales", False),
    "scales_tail": ("scales", True),
    "full_tail": ("full", True),
}


def shrink(image, side):
    h,w = image.shape[-2:]
    scale = side/max(h,w)
    return F.interpolate(image,(max(32,round(h*scale)),max(32,round(w*scale))),
                         mode="bilinear",align_corners=False,antialias=True)


def specialist_outputs(forward,image):
    return tuple(operation(image).clamp(0,1) for operation in forward)


def full_descriptor(image, outputs):
    (x,a,b), valid = matched_view((image,*outputs))
    return descriptor(x,a,b,valid)[1]


def preview_descriptors(forward,image,two=True):
    high = shrink(image,LEVELS[1])
    high_outputs = specialist_outputs(forward,high)
    (x,a,b),valid = matched_view((high,*high_outputs))
    raw,pair = descriptor(x,a,b,valid)
    if not two:
        return {"raw":raw,"preview":pair}
    low = shrink(image,LEVELS[0])
    low_outputs = specialist_outputs(forward,low)
    drifts = []
    for small,large in zip(low_outputs,high_outputs):
        # Compare expert corrections in the same 128-pixel coordinate system.
        large_correction = F.interpolate(large-high,low.shape[-2:],mode="bilinear",align_corners=False,antialias=True)
        drifts.append((large_correction-(small-low)).abs())
    residuals = torch.cat(drifts,1)
    summary = moments(residuals,torch.ones_like(residuals[:, :1]))
    return {"raw":raw,"preview":pair,"scales":torch.cat((pair,summary),1)}


def raw_descriptor(image):
    high = shrink(image,LEVELS[1])
    (view,),valid = matched_view((high,))
    return descriptor(view,view,view,valid)[0]


def fit_ridge(x, y, weights):
    weights = weights / weights.sum()
    center = weights @ x
    scale = ((x-center).square().T@weights).sqrt().clamp_min(.003)
    intercept = weights @ y
    features = (x-center)/scale
    weighted = features*weights[:,None]
    penalty = (24/len(x))*torch.eye(x.shape[1],device=x.device)
    coefficients = torch.linalg.solve(weighted.T@features+penalty,weighted.T@(y-intercept))
    return dict(center=center,scale=scale,intercept=intercept,coefficients=coefficients)


def regress(state, x, aware):
    values = ((x-state["center"])/state["scale"])@state["coefficients"]+state["intercept"]
    score = values[:,0]
    if aware:
        score = score - PENALTY*values[:,1].clamp_min(0)
    return score


def group_weights(groups, exponent):
    counts = torch.bincount(groups)
    return counts[groups].float().pow(-exponent)


def group_quantile(values,groups,q):
    order = values.argsort()
    weights = group_weights(groups,1)[order]
    eligible = (weights.cumsum(0) >= q*weights.sum()).nonzero(as_tuple=True)[0]
    return values[order[eligible[0]]]


def fit_policies(by_feature, gain, group_ids):
    names = sorted(set(group_ids))
    groups = torch.tensor([names.index(g) for g in group_ids],device=gain.device)
    # 3 dB loss roughly doubles MSE relative to the default expert.
    targets = torch.stack((gain,(-gain-HARM).clamp_min(0)),1)
    models, oof = {}, {}
    for name,(key,aware) in FEATURES.items():
        x = by_feature[key]
        prediction = torch.empty_like(gain)
        for group in range(len(names)):
            test = groups == group
            fold = fit_ridge(x[~test],targets[~test],group_weights(groups[~test],.5))
            prediction[test] = regress(fold,x[test],aware)
        utility = gain-PENALTY*targets[:,1] if aware else gain
        margin = group_quantile(prediction-utility,groups,RESIDUAL_QUANTILE).clamp_min(0)
        model = fit_ridge(x,targets,group_weights(groups,.5))
        models[name] = dict(**model,margin=margin)
        oof[name] = prediction
    return models, oof


def choose(models, attributes):
    scores = {name:regress(models[name],attributes[key],aware).item()-models[name]["margin"].item()
              for name,(key,aware) in FEATURES.items() if key in attributes}
    if "preview" in attributes:
        scores["preview_sign"] = scores["preview_mean"]+models["preview_mean"]["margin"].item()
    return scores


@torch.inference_mode()
def restore(image,forward,models,method="scales_tail"):
    key,aware = FEATURES[method]
    if key == "full":
        full = specialist_outputs(forward,image)
        x = full_descriptor(image,full)
        selected = bool(regress(models[method],x,aware)[0] > models[method]["margin"])
        return full[int(selected)], selected
    if key == "raw":
        x = raw_descriptor(image)
    else:
        x = preview_descriptors(forward,image,two=key=="scales")[key]
    selected = bool(regress(models[method],x,aware)[0] > models[method]["margin"])
    return forward[int(selected)](image).clamp(0,1), selected


def training_rgb(card):
    with RealPairs(("nh_train",)) as archive:
        ids = archive.ids("nh_train")
        if ids != [f"{i:02d}" for i in range(1,46)]:
            raise ValueError("Registered NH-HAZE training population changed")
        for key in ids:
            hazy,clear = archive.read("nh_train",key)
            pixels = [image_tensor(long_edge(img,min(1024,max(img.size)))) for img in (hazy,clear)]
            yield "nh_train",key,f"NH-{(int(key)-1)//15}",*pixels
    if digest(card["file"]) != card["SHA256"]:
        raise ValueError("Native training RGB views changed")
    for row in torch.load(card["file"],map_location="cpu",weights_only=True):
        yield "rw2ah_train",row["id"],"RW-"+row["id"].split("-")[0],row["hazy"],row["clean"]


def load_policies():
    reference = read(ROOT/"fit.json")
    if digest(ROOT/"plan.json") != reference["plan_SHA256"] or digest(ROOT/"policies.pth") != reference["weights_SHA256"]:
        raise ValueError("Preview routing records changed")
    return torch.load(ROOT/"policies.pth",map_location="cuda",weights_only=True),reference


@torch.inference_mode()
def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    card = stage()
    forward,authors = experts()
    file = ROOT/"plan.json"
    plan = dict(code_SHA256=digest(__file__),native_train_SHA256=digest(STAGED),
                checkpoint_sources=authors,
                methods={k:list(v) for k,v in FEATURES.items()},preview_long_edges=list(LEVELS),
                penalty=PENALTY,severe_loss_threshold_dB=HARM,one_sided_residual_quantile=RESIDUAL_QUANTILE,
                fit_scope="45 registered NH-HAZE training pairs plus 96 registered RW2AH train pairs. Expert PSNRs at evaluator-size native RGB through 1024. Same two frozen experts for every policy.",
                folds="Three 15-ID NH blocks plus each RW2AH training prefix. Regressions weight samples by inverse square-root group size. Calibration quantiles give each held-out group equal weight.",
                model="Fixed ridge coefficients for evaluator-size PGL-minus-MCT PSNR and excess loss beyond 3 dB. Tail score subtracts nonnegative predicted excess loss. Two preview scales add cross-resolution correction differences.",
                assessment="Evaluate all fitted policies on all previously inspected benchmarks for development, and the intact independently acquired BeDDE set by author site and ROI. Primary report uses fixed full-tail vs preview-tail variants and all controls. A no-margin sign rule measures the role of calibration.")
    if file.exists() and read(file) != plan:
        raise ValueError("Preview routing plan changed")
    if not file.exists():
        atomic_write(file,plan)
    if (ROOT/"fit.json").exists():
        return load_policies()
    aggregate = {n:[] for n,_ in FEATURES.values()}
    labels, gains = [], []
    for dataset,scene,group,hazy,clear in training_rgb(card):
        image, target = as_float(hazy), as_float(clear)
        full = specialist_outputs(forward,image)
        error = torch.stack([(p-target).square().mean() for p in full])
        gains.append(10*(error[0].clamp_min(1e-12)/error[1].clamp_min(1e-12)).log10())
        features = preview_descriptors(forward,image)
        features["full"] = full_descriptor(image,full)
        for key, stat in features.items():
            aggregate[key].append(stat)
        labels.append((dataset,scene,group))
        if len(labels)%30==0:
            print(dict(training_pairs=len(labels)),flush=True)
    if len(labels) != 141:
        raise ValueError("Incorrect preview training pair count")
    gain = torch.stack(gains)
    matrix = {k:torch.cat(v) for k,v in aggregate.items()}
    if not all(torch.isfinite(v).all() for v in (*matrix.values(),gain)):
        raise ValueError("Training features are nonfinite")
    models,predicted = fit_policies(matrix,gain,[p[2] for p in labels])
    weight = ROOT/"policies.pth"
    torch_save(weight,models)
    cv = ROOT/"train_holdout_scores.csv"
    write_csv(cv,[dict(dataset=dataset,scene=scene,group=group,gain_dB=gains[i].item(),
                       **{name:values[i].item() for name,values in predicted.items()})
                  for i,(dataset,scene,group) in enumerate(labels)])
    receipt = dict(weights_SHA256=digest(weight),plan_SHA256=digest(file),crossfold_SHA256=digest(cv),
                   groups=dict(Counter(g for _,_,g in labels)),count=len(labels),
                   train_positive=int((gain>0).sum()),train_severe=int((gain<-HARM).sum()),
                   thresholds={name:model["margin"].item() for name,model in models.items()},
                   columns={name:value.shape[1] for name,value in matrix.items()},
                   torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                   finished_utc=now())
    atomic_write(ROOT/"fit.json",receipt)
    print(receipt,flush=True)
    return models,receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train()


if __name__=="__main__":
    main()

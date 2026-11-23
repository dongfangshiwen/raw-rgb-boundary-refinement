"""Estimate global and spatial fusion weights from the two completed restorations."""
import argparse
from collections import Counter

import torch
import torch.nn.functional as F

from boundary_refinement.training.calibrate_expert_choice import descriptor
from boundary_refinement.training.conditional_fusion import MAIN as PREV_MAIN, actions, bedde_split, city_manifest, features
from boundary_refinement.training.conditional_fusion import load_state as load_previous
from boundary_refinement.training.conditional_fusion import paired_geometry, training_rows
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.models.restoration.expert_fusion import experts, matched_view
from boundary_refinement.training.preview_routing import fit_ridge, group_quantile, group_weights
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.staging.stage_postfusion_holdout import ROOT
from boundary_refinement.data.staging.stage_preview_training import STAGED
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


GRID = 4
MARGIN_QUANTILE = .80
MIN_DISTANCE = .025
MODES = ("post_alpha","post_step","post_lower","window_step","window_lower")
MAIN = "post_step"


def windows(image):
    cells = torch.eye(GRID**2,device=image.device,dtype=image.dtype).reshape(GRID**2,1,GRID,GRID)
    return F.interpolate(cells,size=image.shape[-2:],mode="bilinear",align_corners=False).flatten(1)


def descriptors(image,anchor,challenger,spatial=True):
    views, valid = matched_view((image,anchor,challenger))
    _, full = descriptor(*views,valid)
    if not spatial:
        return full,None
    joined = torch.cat((image,anchor,challenger),1)
    side = tuple(min(160,v) for v in image.shape[-2:])
    small = F.interpolate(joined,size=side,mode="bilinear",align_corners=False,antialias=True)

    def cell_mean(x):
        return F.adaptive_avg_pool2d(x,(GRID,GRID))

    means = cell_mean(small)
    spread = (cell_mean(small.square())-means.square()).clamp_min(0).sqrt()
    _,a,b = small.split(3,dim=1)
    delta = cell_mean((b-a).abs())
    gray = torch.cat([v.mean(1,keepdim=True) for v in small.split(3,dim=1)],1)
    detail = cell_mean((gray-F.avg_pool2d(gray,3,1,1,count_include_pad=False)).abs())
    local = torch.cat((means,spread,delta,detail),1).flatten(2).transpose(1,2)[0]
    return full,torch.cat((full.expand(GRID**2,-1),local),1)


def window_geometry(anchor,challenger,reference,valid,basis):
    residual = reference-anchor
    step = challenger-anchor
    planes = torch.cat((residual.square().mean(1),step.square().mean(1),(residual*step).mean(1)),0).flatten(1).T
    full_distance = ((basis@planes[:,1])/basis.sum(1)).clamp_min(0).sqrt()
    if valid is None:
        weighted = basis
    else:
        weighted = basis*valid.reshape(1,-1).float()
    area = weighted.sum(1)
    terms = (weighted@planes)/area[:,None].clamp_min(1e-6)
    return terms,full_distance,area


def targets(geometry,distance):
    optimum = geometry[...,2]/geometry[...,1].clamp_min(1e-12)
    return torch.stack((optimum.clamp(0,1),optimum*distance),-1)


def predict(model, x):
    return ((x-model["center"])/model["scale"])@model["coefficients"]+model["intercept"]


def train_layer(x,y,weights):
    # Positive steps above one still map to alpha=1. Keep calibration targets in the original units.
    return fit_ridge(x,y.clamp(-1,1),weights)


def local_alpha(step,distance):
    return (step/distance.clamp_min(1e-6)).clamp(0,1)


def field_grid(alpha,shape):
    return F.interpolate(alpha.reshape(1,1,GRID,GRID),size=shape,mode="bilinear",align_corners=False)


def scores_after_call(image,anchor,challenger,state,tile_basis=None,spatial=True):
    global_desc,local_desc = descriptors(image,anchor,challenger,spatial)
    disagreement = (challenger-anchor).square().mean(1).flatten()
    full_dist = disagreement.mean().sqrt()
    estimate = predict(state["linear"],global_desc)[0]
    alpha = estimate[0].clamp(0,1)
    centered = estimate[1]
    lower = centered-state["global_margin"]*full_dist.clamp_min(MIN_DISTANCE)
    weights = dict(post_alpha=alpha,post_step=local_alpha(centered,full_dist),post_lower=local_alpha(lower,full_dist))
    detail = dict(lower_step=lower)
    if not spatial:
        return weights,detail
    if tile_basis is None:
        tile_basis = windows(image)
    dist = ((tile_basis@disagreement)/tile_basis.sum(1)).clamp_min(0).sqrt()
    tile_step = predict(state["local"],local_desc)[:,1]
    local_lower = tile_step-state["joint_margin"]*dist.clamp_min(MIN_DISTANCE)
    weights.update(window_step=field_grid(local_alpha(tile_step,dist),image.shape[-2:]),
                   window_lower=field_grid(local_alpha(local_lower,dist),image.shape[-2:]))
    detail["tile_lower_steps"] = local_lower
    return weights,detail


@torch.inference_mode()
def restore(image,forward,previous,state,method=MAIN):
    anchor = forward[0](image).clamp(0,1)
    route,amount,_ = actions(previous,features(image,anchor))[PREV_MAIN]
    if route.item()<=previous["thresholds"][PREV_MAIN] or amount.item()<=0:
        return anchor,dict(pgl=False,mean_alpha=0.)
    challenger = forward[1](image).clamp(0,1)
    weights,_ = scores_after_call(image,anchor,challenger,state,spatial=method.startswith("window_"))
    alpha = weights[method]
    return torch.lerp(anchor,challenger,alpha),dict(pgl=True,mean_alpha=alpha.mean().item())


def load():
    protocol = ROOT/"protocol.json"
    weight = ROOT/"weights.pth"
    info = read(ROOT/"fit.json")
    if digest(protocol)!=info["protocol_SHA256"] or digest(weight)!=info["weights_SHA256"]:
        raise ValueError("Post-fusion policy does not match the frozen receipt")
    previous,old = load_previous()
    if old["weights_SHA256"]!=info["prior_gate_SHA256"]:
        raise ValueError("The policy's frozen invocation decision changed")
    data = torch.load(weight,map_location="cuda",weights_only=True)
    return previous,data,info


@torch.inference_mode()
def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    _,prior = load_previous()
    cities = city_manifest()
    split = bedde_split(cities)
    previous = read(ROOT.parent/"conditional-fusion-v29/protocol.json")
    if previous["city_roles"]!=split:
        raise ValueError("V29 and V30 training groups differ")
    forward,authors = experts()
    design = dict(parent=prior["weights_SHA256"],code_SHA256=digest(__file__),role_policy="Exact V29 fit and calibration split.",
                  methods=list(MODES),main=MAIN,grid=[GRID,GRID],margin_quantile=MARGIN_QUANTILE,
                  images="All label supervision uses the decoded paired RGB at the evaluator's geometry and the original BeDDE masks.",
                  design="Fit bounded alpha and signed projected step by CUDA ridge. Predict globally or in overlapping bilinear windows. Calibrate overprediction after normalizing by full-image (or per-window) disagreement. For spatial upper errors use the largest valid-window overprediction for each image.",
                  inference="Frozen V29 anchor gate, same selected frames for every post-fusion weight. PGL is invoked at most once after positive gate decision.",
                  cases="Keep old cohorts diagnostic. Test never-scored REVIDE Test frames and reserved RW2AH capture codes. Use 30% per-cohort budget with ordered unlabeled gate scores, then separate online fixed-threshold evaluation.",
                  authors=authors)
    card = ROOT/"protocol.json"
    if card.exists() and read(card)!=design:
        raise ValueError("The post-fusion protocol changed")
    if not card.exists():
        atomic_write(card,design)
    if (ROOT/"fit.json").exists():
        return load()
    labels, xg,xl,yg,yl,dg,dl,area = ([] for _ in range(8))
    for dataset,scene,group,hazy,clear,roi,role in training_rows(read(STAGED),cities,split):
        source,target = as_float(hazy),as_float(clear)
        mct,pgl = (operation(source).clamp(0,1) for operation in forward)
        whole,tile = descriptors(source,mct,pgl)
        valid = roi[None,None].cuda() if roi is not None else None
        global_stats = paired_geometry(mct,pgl,target,valid)
        stats,distance,areas = window_geometry(mct,pgl,target,valid,windows(source))
        length = global_stats[3].sqrt()
        xg.append(whole[0])
        xl.append(tile)
        yg.append(targets(global_stats[:3],length))
        yl.append(targets(stats,distance))
        dg.append(length)
        dl.append(distance)
        area.append(areas/areas.sum())
        labels.append((dataset,scene,group,role))
        if len(labels)%60==0:
            print(dict(training_and_calibration_pairs=len(labels)),flush=True)
    xg,xl,yg,yl,dg,dl,area = (torch.stack(v) for v in (xg,xl,yg,yl,dg,dl,area))
    if not all(torch.isfinite(v).all() for v in (xg,xl,yg,yl,dg,dl,area)):
        raise ValueError("Nonfinite post-fusion training feature")
    indices = {v:i for i,v in enumerate(sorted({r[2] for r in labels}))}
    groups = torch.tensor([indices[r[2]] for r in labels],device="cuda")
    fit = torch.tensor([r[3]=="fit" for r in labels],device="cuda")
    count = Counter(r[3] for r in labels)
    if count!={"fit":247,"calibrate":70}:
        raise ValueError("Previously registered split membership differs")
    group_weight = group_weights(groups[fit],.5)
    linear = train_layer(xg[fit],yg[fit],group_weight)
    valid_tiles = (area[fit]>0).flatten()
    local = train_layer(xl[fit].flatten(0,1)[valid_tiles],yl[fit].flatten(0,1)[valid_tiles],
                        (group_weight[:,None]*area[fit]).flatten()[valid_tiles])
    residual = (predict(linear,xg[~fit])[:,1]-yg[~fit,1])/dg[~fit].clamp_min(MIN_DISTANCE)
    tile_errors = (predict(local,xl[~fit])[:,:,1]-yl[~fit,:,1])/dl[~fit].clamp_min(MIN_DISTANCE)
    maximum = torch.where(area[~fit]>0,tile_errors,-torch.inf).amax(1)
    q = group_quantile(residual,groups[~fit],MARGIN_QUANTILE).clamp_min(0)
    joint = group_quantile(maximum,groups[~fit],MARGIN_QUANTILE).clamp_min(0)
    target_file = ROOT/"weights.pth"
    fitted = dict(linear=linear,local=local,global_margin=q,joint_margin=joint)
    torch_save(target_file,fitted)
    receipt = dict(protocol_SHA256=digest(card),weights_SHA256=digest(target_file),prior_gate_SHA256=prior["weights_SHA256"],
                   fit_pairs=count["fit"],calibration_pairs=count["calibrate"],fit_windows=int(valid_tiles.sum()),
                   margin=q.item(),joint_margin=joint.item(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
                   finished_utc=now())
    atomic_write(ROOT/"fit.json",receipt)
    print(receipt,flush=True)
    return fitted,receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train()


if __name__=="__main__":
    main()

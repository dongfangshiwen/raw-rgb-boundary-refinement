"""Fit and execute conditional continuous fusion with an anchor-error loss cap."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps
from scipy.io import loadmat
import torch

from dehaze_seg.utils import write_csv
from scripts.calibrate_expert_choice_v27 import descriptor
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.expert_fusion_v26 import experts, matched_view
from scripts.preview_routing_v28 import fit_ridge, group_quantile, group_weights, raw_descriptor, training_rgb
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_detail_data_v24 import image_tensor
from scripts.stage_bedde_v28 import RECEIPT as BEDDE
from scripts.stage_fusion_holdouts_v29 import ROOT
from scripts.stage_preview_training_v28 import STAGED
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


LOSS_DB = 3.0
RADIUS = 10**(LOSS_DB/20)-1
CALL_SHARE = .30
ANCHOR_QUANTILE = .90
MAIN = "geometry_capped"
METHODS = {
    "raw_hard": ("raw", "hard"),
    "raw_blend": ("raw", "blend"),
    "anchor_blend": ("anchor", "blend"),
    "raw_geometry_capped": ("raw", "geometry_capped"),
    "geometry_half": ("anchor", "geometry_half"),
    "geometry_open": ("anchor", "geometry_open"),
    MAIN: ("anchor", "geometry_capped"),
}


def city_manifest():
    source = read(BEDDE)
    origin = Path(source["directory"])
    result = defaultdict(lambda: defaultdict(dict))
    for name in source["members"]:
        path = PurePosixPath(name)
        file = origin/path
        if digest(file)!=source["image_hashes"][name]:
            raise ValueError("BeDDE source file changed")
        result[path.parts[-3].lower()][path.parts[-2].lower()][path.stem.lower()] = file
    if len(result)!=23 or sum(len(x["fog"]) for x in result.values())!=208:
        raise ValueError("The registered BeDDE population changed")
    return result


def bedde_split(cities):
    ordered = sorted(cities, key=lambda name: hashlib.sha256(("conditional-fusion-v29/"+name).encode()).digest())
    return {city: "calibrate" if i<5 else "dev" if i<10 else "fit" for i,city in enumerate(ordered)}


def city_pairs(cities):
    for city, roles in sorted(cities.items()):
        if len(roles["gt"])!=1:
            raise ValueError("BeDDE city needs exactly one reference")
        with Image.open(next(iter(roles["gt"].values()))) as opened:
            target = ImageOps.exif_transpose(opened).convert("RGB")
        scale = min(1024,max(target.size))/max(target.size)
        size = tuple(round(v*scale) for v in target.size)
        clean = image_tensor(target.resize(size,Image.Resampling.BILINEAR))
        for scene,path in sorted(roles["fog"].items()):
            source = roles["mask"].get(scene+"_mask")
            if source is None:
                raise ValueError("Missing BeDDE valid region for "+scene)
            mask = np.squeeze(loadmat(source,variable_names=["mask"])["mask"])
            if mask.shape != (target.height,target.width):
                raise ValueError("BeDDE ROI doesn't match reference coordinates")
            roi = torch.from_numpy(np.array(Image.fromarray(mask>0).resize(size,Image.Resampling.NEAREST)))
            if not roi.any():
                raise ValueError("BeDDE ROI contains no pixels")
            with Image.open(path) as opened:
                hazy = image_tensor(ImageOps.exif_transpose(opened).convert("RGB").resize(size,Image.Resampling.BILINEAR))
            yield "bedde",scene,city,hazy,clean,roi


def features(image, anchor):
    # The raw-only control keeps the previous input-only representation.
    raw = raw_descriptor(image)
    (x,a), valid = matched_view((image,anchor))
    _, paired = descriptor(x,a,a,valid)
    anchor_features = torch.cat((raw,paired[:,10:20],paired[:,30:36],paired[:,-8:-4]),1)
    return {"raw":raw,"anchor":anchor_features}


def paired_geometry(anchor, challenger, clean, mask):
    residual = clean-anchor
    direction = challenger-anchor

    def average(value):
        return value.mean() if mask is None else (value*mask).sum()/(3*mask.sum())

    err = average(residual.square()).clamp_min(1e-12)
    energy = average(direction.square())
    projection = average(residual*direction)
    full_energy = direction.square().mean()
    return torch.stack((err,energy,projection,full_energy))


def relative_psnr(stats, alpha):
    base,energy,projection,_ = stats.unbind(-1)
    mse = (base-2*projection*alpha+energy*alpha.square()).clamp_min(1e-12)
    return 10*(base/mse).log10()


def supervision(stats):
    base, energy, projection, full = stats.unbind(-1)
    sigma = base.sqrt()
    distance = energy.clamp_min(1e-12).sqrt()
    alpha = (projection/energy.clamp_min(1e-12)).clamp(0,1)
    agreement = (projection/(distance*sigma)).clamp(-.99,.99)
    scale_for_cap = sigma*full.clamp_min(1e-12).sqrt()/distance
    return torch.stack((relative_psnr(stats,torch.ones_like(alpha)),relative_psnr(stats,alpha),
                        alpha,sigma.log(),distance.log(),agreement.atanh(),scale_for_cap.log()),-1)


def estimate(model,x):
    return ((x-model["center"])/model["scale"])@model["coefficients"]+model["intercept"]


def prior_action(pred,kind):
    if kind=="hard":
        return pred[:,0],torch.ones_like(pred[:,0])
    if kind=="blend":
        return pred[:,1],pred[:,2].clamp(0,1)
    ratio = (pred[:,4]-pred[:,3]).clamp(-10,10).exp()
    agreement = pred[:,5].tanh()
    alpha = (agreement/ratio).clamp(0,1)
    argument = alpha*ratio
    error = ((1-agreement*argument).square()+(1-agreement.square())*argument.square()).clamp_min(1e-8)
    score = -10*error.log10()
    return score,torch.full_like(alpha,.5) if kind=="geometry_half" else alpha


def actions(state,inputs):
    predictions = {k:estimate(state["models"][k],v) for k,v in inputs.items()}
    result = {}
    for method,(key,kind) in METHODS.items():
        row = predictions[key]
        score,alpha = prior_action(row,kind)
        lower = (row[:,6]-state["scale_margin"][key]).clamp(-20,3).exp()
        result[method] = (score,alpha,lower)
    return result


def cap_alpha(alpha,lower,distance):
    return torch.minimum(alpha,RADIUS*lower/distance.clamp_min(1e-6))


@torch.inference_mode()
def restore(image,forward,state,method=MAIN):
    key,kind = METHODS[method]
    cutoff = state["thresholds"][method]
    if kind=="hard":
        row = estimate(state["models"]["raw"],raw_descriptor(image))
        score = row[0,0].item()
        selected = score>cutoff
        return forward[int(selected)](image).clamp(0,1),dict(pgl=selected,alpha=float(selected),score=score)

    anchor = forward[0](image).clamp(0,1)
    x = raw_descriptor(image) if key=="raw" else features(image,anchor)[key]
    row = estimate(state["models"][key],x)
    score,alpha = prior_action(row,kind)
    if score.item()<=cutoff or alpha.item()<=0:
        return anchor,dict(pgl=False,alpha=0.,score=score.item())
    alternative = forward[1](image).clamp(0,1)
    if kind=="geometry_capped":
        lower = (row[:,6]-state["scale_margin"][key]).clamp(-20,3).exp()
        alpha = cap_alpha(alpha,lower,(alternative-anchor).square().mean().sqrt())
    return torch.lerp(anchor,alternative,alpha.item()),dict(pgl=True,alpha=alpha.item(),score=score.item())


def training_rows(registration,cities,split):
    for dataset,scene,group,hazy,clear in training_rgb(registration):
        role = "calibrate" if group in ("NH-2","RW-3","RW-4","RW-5") else "fit"
        yield dataset,scene,group,hazy,clear,None,role
    for dataset,scene,city,hazy,clear,valid in city_pairs(cities):
        if split[city]!="dev":
            yield dataset,scene,"BeDDE/"+city,hazy,clear,valid,split[city]


def calibration(matrix,stats,labels):
    group_ids = {name:i for i,name in enumerate(sorted({v[2] for v in labels}))}
    groups = torch.tensor([group_ids[v[2]] for v in labels],device=stats.device)
    fit_mask = torch.tensor([v[3]=="fit" for v in labels],device=stats.device)
    cal_groups = groups[~fit_mask]
    y = supervision(stats)
    models = {k:fit_ridge(v[fit_mask],y[fit_mask],group_weights(groups[fit_mask],.5)) for k,v in matrix.items()}
    predictions = {k:estimate(x,matrix[k][~fit_mask]) for k,x in models.items()}
    margins = {k:group_quantile(v[:,6]-y[~fit_mask,6],cal_groups,ANCHOR_QUANTILE).clamp_min(0)
               for k,v in predictions.items()}
    thresholds = {m:group_quantile(prior_action(predictions[k],kind)[0],cal_groups,1-CALL_SHARE).clamp_min(0).item()
                  for m,(k,kind) in METHODS.items()}
    return dict(models=models,scale_margin=margins,thresholds=thresholds)


def load_state():
    card = read(ROOT/"fit.json")
    if digest(ROOT/"protocol.json")!=card["protocol_SHA256"] or digest(ROOT/"weights.pth")!=card["weights_SHA256"]:
        raise ValueError("Conditional fusion state differs from the frozen fit")
    return torch.load(ROOT/"weights.pth",map_location="cuda",weights_only=True),card


@torch.inference_mode()
def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    forward,authors = experts()
    registered = read(STAGED)
    cities = city_manifest()
    split = bedde_split(cities)
    protocol = dict(author_models=authors,methods={k:list(v) for k,v in METHODS.items()},
                    prior_training_SHA256=digest(STAGED),bedde_SHA256=digest(BEDDE),city_roles=split,
                    roi="Only BeDDE author valid masks enter paired targets. The inference descriptor receives RGB only.",
                    training="V28 training populations and BeDDE fit cities. Coefficients estimated by CUDA weighted ridge.",
                    calibration="Separate BeDDE cities, NH block 2, and RW filename prefixes 3, 4, 5. Each calibration group has equal quantile weight. No fresh-site images or metrics are used.",
                    evaluation="Fixed online thresholds dominate the prospective evaluation. Report a separate matched 30% PGL call budget using score order without reference values. RW-Haze released pairs and the unused NTIRE24 complement are held out.",
                    losses="Fit PSNR improvement, projected alpha and first- and second-order relative error terms. Cap correction using observed full-image disagreement and an empirically calibrated lower estimate of effective anchor error.",
                    nominal_share=CALL_SHARE,harm_dB=LOSS_DB,lower_anchor_quantile=ANCHOR_QUANTILE,
                    code_SHA256=digest(__file__))
    file = ROOT/"protocol.json"
    if file.exists() and read(file)!=protocol:
        raise ValueError("Conditional fusion protocol changed")
    if not file.exists():
        atomic_write(file,protocol)
    if (ROOT/"fit.json").exists():
        return load_state()
    rows,stats,labels = defaultdict(list),[],[]
    for dataset,scene,group,hazy,clear,roi,role in training_rows(registered,cities,split):
        image,target = as_float(hazy),as_float(clear)
        a,b = (operation(image).clamp(0,1) for operation in forward)
        for key,value in features(image,a).items():
            rows[key].append(value)
        terms = paired_geometry(a,b,target,roi[None,None].cuda() if roi is not None else None)
        stats.append(terms)
        labels.append((dataset,scene,group,role))
        if len(labels)%60==0:
            print(dict(pairs_processed=len(labels)),flush=True)
    matrix = {k:torch.cat(v) for k,v in rows.items()}
    geometry = torch.stack(stats)
    if not all(torch.isfinite(z).all() for z in (geometry,*matrix.values())):
        raise ValueError("A training feature or blend geometry is nonfinite")
    model = calibration(matrix,geometry,labels)
    target_file = ROOT/"weights.pth"
    torch_save(target_file,model)
    pairs_file = ROOT/"fit_pairs.csv"
    write_csv(pairs_file,[dict(dataset=d,scene=s,group=g,role=r,anchor_mse=float(geometry[i,0]),
                             disagreement=float(geometry[i,1]),projection=float(geometry[i,2]))
                         for i,(d,s,g,r) in enumerate(labels)])
    counts = Counter(f"{r}/{d}" for d,s,g,r in labels)
    receipt = dict(weights_SHA256=digest(target_file),protocol_SHA256=digest(file),pair_roles=dict(counts),
                   fit_pairs_SHA256=digest(pairs_file),thresholds=model["thresholds"],
                   scale_margin={k:v.item() for k,v in model["scale_margin"].items()},
                   cuda=torch.version.cuda,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                   finished_utc=now())
    atomic_write(ROOT/"fit.json",receipt)
    print(receipt,flush=True)
    return model,receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train()


if __name__=="__main__":
    main()

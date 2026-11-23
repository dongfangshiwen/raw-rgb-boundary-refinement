"""Compare all frozen adaptive subspaces on matched old and new RGB pairs."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import torch

from boundary_refinement.utils import write_csv
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float, as_rgb
from boundary_refinement.evaluation.photographic.evaluate_post_fusion import samples as v30_samples
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.photographic.evaluate_structure_transfer import data_split, measured
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.training.selective_adaptation import ARMS, MAIN, completed
from boundary_refinement.training.source_replay import cohorts
from boundary_refinement.data.staging.stage_layer_holdouts import ROOT
from boundary_refinement.data.staging.stage_postfusion_holdout import camera_code
from boundary_refinement.training.structure_transfer import SEEDS, load_fit
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


SETS = ("development", "fresh_rw", "transfer", "fresh_hd")
TRANSFER = {"revide","rwhaze","ntire24_complement","nhhaze2"}
NAMES = ("Hazy input","PGL-Net-T author","Full fine-tuning","Affine only","Local coordinates","Local LoRA","Clear")


def samples(split):
    if split=="development":
        yield from data_split("lmhaze")
        for data,id,group,x,y,_ in v30_samples("rw_reserve"):
            yield data,id,group,x,y
        yield from cohorts("confirmation")
        return
    if split=="transfer":
        for data,id,group,x,y in data_split("transfer"):
            if data in TRANSFER:
                yield data,id,group,x,y
        return
    flag = "rw" if split=="fresh_rw" else "hd"
    card = read(ROOT/(flag+"_data.json"))
    if digest(card["file"]) != card["SHA256"]:
        raise ValueError("Fresh holdout RGB changed")
    for row in torch.load(card["file"],map_location="cpu",weights_only=True):
        identifier = row["id"]
        group = camera_code(identifier) if flag=="rw" else identifier
        yield ("rw_new" if flag=="rw" else "hd_nh_haze"),identifier,group,row["hazy"],row["clean"]


def render(path,examples):
    w,h = 255,222
    grid = Image.new("RGB",(w*len(NAMES),h*len(examples)),"#edf3ef")
    annotate = ImageDraw.Draw(grid)
    for i,(title,views) in enumerate(examples):
        annotate.text((6,i*h+4),title,fill="#142a22")
        for j,(label,image) in enumerate(zip(NAMES,views,strict=True)):
            annotate.text((j*w+6,i*h+21),label,fill="#142a22")
            image.thumbnail((w-8,180),Image.Resampling.LANCZOS)
            grid.paste(image,(j*w+(w-image.width)//2,i*h+36))
    grid.save(path)


def aggregate(rows):
    per_seed = defaultdict(list)
    per_image = defaultdict(dict)
    for row in rows:
        d,m,s,g,key=(row[k] for k in ("dataset","method","seed","group","scene"))
        per_seed[d,m,s].append(row)
        per_image[d,g,key].setdefault(m,[]).append(row)
    metrics = ("psnr","ssim","edge_mae")
    by_seed = {f"{d}/{m}/{s}":{k:float(np.mean([r[k] for r in values])) for k in metrics}
               for (d,m,s),values in per_seed.items()}
    scores = defaultdict(list)
    severe = defaultdict(list)
    groups = defaultdict(lambda:defaultdict(list))
    pairs = defaultdict(lambda:defaultdict(list))
    for (dataset,group,id),values in per_image.items():
        computed = {m:np.mean([[r[k] for k in metrics] for r in rr],axis=0) for m,rr in values.items()}
        for method,stat in computed.items():
            scores[dataset,method].append(stat)
            base = computed["pgl"][0]
            severe[dataset,method].append(stat[0]-base)
            groups[dataset,method][group].append(stat[0])
            if method!=MAIN:
                pairs[dataset,method][group].append(computed[MAIN][0]-stat[0])
    summary = {}
    for (dataset,method),parts in scores.items():
        effects = severe[dataset,method]
        mean = np.mean(parts,axis=0)
        summary[f"{dataset}/{method}"] = dict(zip(metrics,mean.tolist()),n=len(parts),
                                  loss_gt_3db=sum(bool(v< -3) for v in effects),
                                  gain_lower_decile=float(np.percentile(effects,10)),
                                  group_psnr={g:float(np.mean(v)) for g,v in groups[dataset,method].items()})
    intervals = {f"{d}/{m}":bootstrap({g:np.mean(v) for g,v in group.items()})
                 for (d,m),group in pairs.items()}
    return summary,by_seed,intervals


@torch.inference_mode()
def evaluate(split):
    configure_cuda()
    record = completed()
    nets = {(r["arm"],r["seed"]):load_fit(load_pgl()[0],r) for r in record["runs"]}
    pre,author = experts()
    observations, demos, seen = [], [], defaultdict(int)
    for i,(data,id,group,hazy,clear) in enumerate(samples(split),1):
        x,y = as_float(hazy),as_float(clear)
        prediction = {("pgl",0):pre[1](x).clamp(0,1),("mct",0):pre[0](x).clamp(0,1)}
        prediction.update({k:pgl_rgb(net,x).clamp(0,1) for k,net in nets.items()})
        observations.extend(dict(dataset=data,method=method,seed=seed,scene=id,group=group,**measured(im,y))
                            for (method,seed),im in prediction.items())
        seen[data] += 1
        if (data in ("lmhaze","rw_new","hd_nh_haze")) and seen[data] in (1,10):
            demos.append((f"{data} {id} | seed {SEEDS[0]}",
                         [as_rgb(v) for v in (x,prediction["pgl",0],*(prediction[m,SEEDS[0]] for m in ARMS),y)]))
        if i%30==0:
            print(dict(split=split,scored=i),flush=True)
    summary,by_seed,intervals = aggregate(observations)
    table = ROOT/(split+".csv")
    write_csv(table,observations)
    picture = ROOT/(split+"_predictions.png")
    if demos:
        render(picture,demos)
    report = dict(split=split,counts=dict(seen),summary=summary,by_seed=by_seed,intervals=intervals,
                  model_registry=author,fit_SHA256=digest(ROOT/"completed.json"),
                  image_SHA256=digest(picture) if demos else None,csv_SHA256=digest(table),
                  code_SHA256=digest(__file__),finished_utc=now())
    atomic_write(ROOT/(split+".json"),report)
    print(dict(counts=dict(seen),psnr={k:round(v["psnr"],4) for k,v in summary.items()}),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=SETS)
    evaluate(parser.parse_args().split)

"""Compare fixed-step restoration ablations on held scenes and external fog data."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import torch

from dehaze_seg.utils import write_csv
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.evaluate_curve_transfer_v25 import as_float, as_rgb
from scripts.evaluate_post_fusion_v30 import samples as v30
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.expert_fusion_v26 import experts
from scripts.projective_ledger_v12 import digest, now, read
from scripts.stage_lmhaze_v31 import CARD as LMHaze, TRAIN_CARD, samples
from scripts.stage_nhhaze2_v32 import CARD as NH2
from scripts.structure_transfer_v32 import ARMS, MAIN, ROOT, SEEDS, completed, edge_visibility, load_fit
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


EXTERNAL = {"rwhaze","ntire24_complement"}
METHOD_NAMES = ("Hazy input", "PGL-Net-T", "Pointwise", "Uniform transfer", "Local transfer", "Local ground truth", "Clear")


def data_split(split):
    if split in ("calibrate","lmhaze"):
        role = "calibrate" if split=="calibrate" else "test"
        for _,scene,group,x,y,_,_ in samples({role}):
            yield split,scene,group,x,y
        return
    for source in ("revide","rw_reserve","prior"):
        for dataset,scene,group,x,y,roi in v30(source):
            if source!="prior" or dataset in EXTERNAL:
                if roi is not None:
                    raise ValueError("The restoration check expects paired full-image targets")
                yield dataset,scene,group,x,y
    if split=="transfer":
        archive = read(NH2)
        if digest(archive["file"])!=archive["SHA256"]:
            raise ValueError("Supplementary NH-HAZE2 inputs changed")
        for row in torch.load(archive["file"],map_location="cpu",weights_only=True):
            yield "nhhaze2",row["id"],row["id"],row["hazy"],row["clean"]


def measured(out, ref):
    delta = out-ref
    mse = delta.square().mean().clamp_min(1e-12)
    gx = (delta[...,1:]-delta[...,:-1]).abs().mean()
    gy = (delta[...,1:,:]-delta[...,:-1,:]).abs().mean()
    return dict(psnr=(-10*mse.log10()).item(),ssim=gaussian_ssim(out,ref)[0].item(),
                edge_mae=(.5*(gx+gy)).item()*255)


def aggregate(rows):
    groups = defaultdict(list)
    means = defaultdict(list)
    for r in rows:
        for subset in ("all","dense_half") if r["dense_half"] else ("all",):
            groups[r["dataset"],subset,r["method"],r["seed"]].append(r)
            means[r["dataset"],subset,r["method"],r["group"],r["scene"]].append(r)
    psnr = {"/".join(map(str,key)): {"n":len(data), **{m:float(np.mean([r[m] for r in data]))
                                               for m in ("psnr","ssim","edge_mae")}} for key,data in groups.items()}
    shared = defaultdict(list)
    for (data,subset,method,group,_),items in means.items():
        shared[data,subset,method].append(np.mean([[r[m] for m in ("psnr","ssim","edge_mae")] for r in items],axis=0))
    summary = {f"{data}/{subset}/{method}":dict(zip(("psnr","ssim","edge_mae"),np.mean(values,axis=0).tolist()))
               for (data,subset,method),values in shared.items()}
    paired = defaultdict(lambda:defaultdict(list))
    for (data,subset,method,group,scene),items in means.items():
        if method != MAIN:
            continue
        value = np.mean([[r[m] for m in ("psnr","ssim","edge_mae")] for r in items],axis=0)
        for control in ("point","uniform_pair","local_reference","pgl","mct"):
            ref = np.mean([[r[m] for m in ("psnr","ssim","edge_mae")] for r in means[data,subset,control,group,scene]],axis=0)
            paired[data,subset,control][group].append(value-ref)
    intervals = {f"{d}/{scope}/{arm}":{metric:bootstrap({g:np.mean(z,axis=0)[i] for g,z in by_group.items()})
                                      for i,metric in enumerate(("psnr","ssim","edge_mae"))}
                 for (d,scope,arm),by_group in paired.items()}
    return summary,psnr,intervals


def render(path, examples):
    width,row_height = 265,218
    grid = Image.new("RGB",(width*len(METHOD_NAMES),row_height*len(examples)),"#ecf3ef")
    draw = ImageDraw.Draw(grid)
    for row,(caption,images) in enumerate(examples):
        draw.text((6,row*row_height+5),caption,fill="#132620")
        for col,(title,image) in enumerate(zip(METHOD_NAMES,images,strict=True)):
            image.thumbnail((width-8,176),Image.Resampling.LANCZOS)
            draw.text((col*width+6,row*row_height+21),title,fill="#132620")
            grid.paste(image,(col*width+(width-image.width)//2,row*row_height+34))
    grid.save(path)


@torch.inference_mode()
def evaluate(split):
    configure_cuda()
    training = completed()
    forward,authors = experts()
    nets = {key:load_fit(load_pgl()[0],r) for r in training["runs"] for key in [(r["arm"],r["seed"]) ]}
    rows, difficulty, demos = [],defaultdict(dict),[]
    for i,(dataset,scene,group,hazy,clear) in enumerate(data_split(split)):
        x,y = as_float(hazy),as_float(clear)
        visible = edge_visibility(hazy[None].cuda(),clear.cuda())[0].item()
        difficulty[dataset,group][scene] = visible
        pgl = forward[1](x).clamp(0,1)
        predictions = {("pgl",0):pgl,("mct",0):forward[0](x).clamp(0,1)}
        predictions.update({key:pgl_rgb(net,x).clamp(0,1) for key,net in nets.items()})
        rows.extend(dict(dataset=dataset,scene=scene,group=group,method=arm,seed=seed,visibility=visible,
                         **measured(pred,y)) for (arm,seed),pred in predictions.items())
        if split=="lmhaze" and i in (5,65,115):
            images = [as_rgb(v) for v in (x,pgl,*(predictions[m,SEEDS[0]] for m in ARMS),y)]
            demos.append((f"LMHaze {scene} | input edge visibility {visible:.2f} | seed {SEEDS[0]}",images))
        if (i+1)%20==0:
            print(dict(split=split,images=i+1),flush=True)
    dense = {(d,g,k) for (d,g),values in difficulty.items() if len(values)>1
             for k in sorted(values,key=values.get)[:len(values)//2]}
    for row in rows:
        row["dense_half"] = (row["dataset"],row["group"],row["scene"]) in dense
    summary,by_seed,intervals = aggregate(rows)
    path = ROOT/f"{split}_scores.csv"
    write_csv(path,rows)
    pic = ROOT/"held_predictions.png"
    if demos:
        render(pic,demos)
    report = dict(split=split,summary=summary,by_seed=by_seed,intervals=intervals,
                  table_SHA256=digest(path),weights_manifest_SHA256=digest(ROOT/"completed.json"),
                  sources={str(p):digest(p) for p in (TRAIN_CARD,*( (LMHaze,) if split=="lmhaze" else () ),
                                                     *( (NH2,) if split=="transfer" else () ))},
                  evaluator_SHA256=digest(__file__),external_models=authors,picture_SHA256=digest(pic) if demos else None,
                  finished_utc=now())
    atomic_write(ROOT/f"{split}.json",report)
    for key,value in summary.items():
        if "/all/" in key or split in ("calibrate","lmhaze"):
            print(key,{k:round(v,4) for k,v in value.items()},flush=True)
    return report


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("calibrate","lmhaze","external","transfer"))
    evaluate(parser.parse_args().split)

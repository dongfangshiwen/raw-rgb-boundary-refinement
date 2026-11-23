"""Native paired SOTS/HSTS restoration screen after the complete V8 weight lock."""
from boundary_refinement.paths import SOURCE_ROOT
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from boundary_refinement.evaluation.paired.evaluate_parallel_joint import ROOT,REPO,read,write,digest,training_lock


def bind():
    global torch,F,np,Image,load_model,ciede2000_rgb,gaussian_ssim_valid
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from boundary_refinement.models.joint.parallel_joint import load_model
    from boundary_refinement.metrics.color import ciede2000_rgb
    from boundary_refinement.evaluation.paired import evaluate_parallel_joint as scoring
    scoring.bind();gaussian_ssim_valid=scoring.gaussian_ssim_valid


def pair(record):
    result=[]
    for role in ("hazy","clear"):
        path=record[role]["path"]
        if digest(path)!=record[role]["file_sha256"]:raise ValueError("Native paired image changed")
        with Image.open(path) as image:
            image=image.convert("RGB")
            if role=="clear" and record["alignment"]=="crop_gt_10px_each_edge":
                image=image.crop((10,10,image.width-10,image.height-10))
            result.append(torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].cuda().float()/255)
    if result[0].shape!=result[1].shape:raise ValueError("Native RGB pair is misaligned")
    return result


def prediction(model,image):
    h,w=image.shape[-2:];right,bottom=-w%16,-h%16
    padded=F.pad(image,(0,right,0,bottom),mode="reflect") if right or bottom else image
    restored=model(padded)[0] if model is not None else padded
    return restored[:,:,:h,:w].float().clamp(0,1)


def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",choices=["v8","partial"],default="v8");args=parser.parse_args()
    lock_function=training_lock
    if args.study=="partial":
        from boundary_refinement.evaluation.paired.evaluate_partial_sharing import training_lock as partial_lock,ROOT as partial_root
        lock_function=partial_lock;ROOT=partial_root
    _,registration,lock=lock_function()
    manifest_path=registration["rgb_benchmarks"]["manifest"]
    if digest(manifest_path)!=registration["rgb_benchmarks"]["manifest_sha256"]:raise ValueError("RGB benchmark registration changed")
    manifest=read(manifest_path)
    jobs=[dict(method="hazy_input",seed=None,checkpoint=None,sha256=None)]
    jobs += [r for r in lock["jobs"] if r["method"] in registration["rgb_benchmarks"]["methods"]]
    implementation=dict(source_sha256=digest(__file__),weight_lock_sha256=digest(ROOT/"training_lock.json"),
        scoring_sha256=digest(SOURCE_ROOT / "boundary_refinement/evaluation/paired/evaluate_parallel_joint.py"),color_sha256=digest(SOURCE_ROOT / "boundary_refinement/metrics/color.py"))
    path=ROOT/"rgb_evaluation_implementation.json"
    if path.exists() and read(path)!=implementation:raise ValueError("Native RGB evaluator changed")
    if not path.exists():
        write(path,implementation)
        snapshot=ROOT/"source_snapshot/boundary_refinement/evaluation/photographic/evaluate_parallel_rgb.py";snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    bind()
    if args.study=="partial":
        global load_model
        from boundary_refinement.models.joint.partial_sharing import load_model
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    summaries=[];predictions=0
    for job in jobs:
        path=ROOT/f"rgb-evaluation/{job['method']}/seed_{job['seed']}.json"
        if path.exists():
            payload=read(path)
            if payload["checkpoint_sha256"]!=job["sha256"]:raise ValueError("RGB saved identity changed")
        else:
            model=None
            if job["checkpoint"]:model,_=load_model(job["checkpoint"])
            rows=[]
            with torch.inference_mode():
                for i,record in enumerate(manifest["pairs"]):
                    image,clear=pair(record);out=prediction(model,image)
                    rows.append(dict(id=record["id"],dataset=record["dataset"],
                        psnr=float(-10*(out-clear).square().mean().clamp_min(1e-10).log10()),
                        ssim_gaussian11_valid=gaussian_ssim_valid(out,clear),l1=float((out-clear).abs().mean()),
                        ciede2000=ciede2000_rgb(out,clear)))
                    if i%100==0:
                        write(ROOT/"rgb_evaluation_status.json",dict(method=job["method"],seed=job["seed"],images=i+1,total_images=len(manifest["pairs"]),updated_utc=datetime.now(timezone.utc).isoformat()))
            groups=[]
            for dataset in manifest["datasets"]:
                chosen=[r for r in rows if r["dataset"]==dataset]
                groups.append(dict(dataset=dataset,count=len(chosen),**{k:float(np.mean([r[k] for r in chosen])) for k in ("psnr","ssim_gaussian11_valid","l1","ciede2000")}))
            payload=dict(method=job["method"],seed=job["seed"],checkpoint_sha256=job["sha256"],images=rows,summary=groups)
            if job["checkpoint"] and digest(job["checkpoint"])!=job["sha256"]:raise ValueError("RGB weights changed")
            write(path,payload);del model;torch.cuda.empty_cache()
        summaries.extend(dict(method=job["method"],seed=job["seed"],**r) for r in payload["summary"])
        predictions+=len(payload["images"])
        print(json.dumps(dict(method=job["method"],seed=job["seed"],predictions=predictions)),flush=True)
    write(ROOT/"rgb_evaluation_summary.json",dict(rows=summaries))
    write(ROOT/"rgb_evaluation_completed.json",dict(models=len(jobs)-1,raw_input_reference=True,paired_images=len(manifest["pairs"]),predictions=predictions,
        source_weights_unchanged=True,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=="__main__":main()

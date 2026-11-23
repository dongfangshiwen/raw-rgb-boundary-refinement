"""Locked evaluation of V8: no target selection, RGB GT never supplied to model."""
from boundary_refinement.paths import MODEL_DATA, SOURCE_ROOT, WORKSPACE
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

REPO=MODEL_DATA;ROOT=REPO/"results/parallel-joint-v8"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def digest(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def training_lock():
    plan=read(ROOT/"training_plan.json");registered=read(ROOT/"evaluation_registration.json")
    if digest(ROOT/"training_plan.json")!=registered["training_plan_sha256"]:raise ValueError("Training registration changed")
    for path,sha in plan["source_sha256"].items():
        if digest(REPO/path)!=sha:raise ValueError("Frozen V8 source changed")
    if digest(plan["label_manifest"])!=plan["label_manifest_sha256"]:raise ValueError("Label registration changed")
    for row in read(plan["label_manifest"])["records"]:
        for path,sha in ((row["original_mask"],row["original_sha256"]),(row["draft_mask"],row["draft_sha256"]),(row["previous_draft_mask"],row["previous_draft_sha256"])):
            if digest(path)!=sha:raise ValueError("Original/registered draft label changed")
    jobs=[]
    for method in plan["methods"]:
        for seed in plan["seeds"]:
            folder=ROOT/f"models/{method}/seed_{seed}"
            record=read(folder/"completed.json")
            if record["plan_sha256"]!=plan["plan_sha256"] or record["epochs"]!=plan["epochs"]:raise ValueError("Training job incomplete")
            if record["successful_steps"]+record["skipped_steps"]!=record["expected_steps"]:raise ValueError("Optimizer-step audit failed")
            if digest(folder/"best.pth")!=record["checkpoint_sha256"]:raise ValueError("Selected weight changed")
            if method!="seg_only" and record["initialization"]["rgb_head"]==record["final_rgb_head_sha256"]:raise ValueError("RGB branch never learned")
            jobs.append(dict(method=method,seed=seed,checkpoint=str((folder/"best.pth").resolve()),sha256=record["checkpoint_sha256"],
                epochs=record["epochs"],successful_steps=record["successful_steps"],skipped_steps=record["skipped_steps"]))
    lock=dict(jobs=jobs,training_jobs=len(jobs),training_epochs=sum(r["epochs"] for r in jobs),
        successful_steps=sum(r["successful_steps"] for r in jobs),amp_skips=sum(r["skipped_steps"] for r in jobs),
        evaluation_registration_sha256=digest(ROOT/"evaluation_registration.json"),training_plan_sha256=digest(ROOT/"training_plan.json"))
    path=ROOT/"training_lock.json"
    if path.exists():
        previous=read(path);previous.pop("created_utc",None)
        if previous!=lock:raise ValueError("Post-training lock changed")
    else:write(path,dict(created_utc=now(),**lock))
    return plan,registered,lock


def bind():
    global np,Image,torch,F,load_model,confusion_matrix,segmentation_metrics,ciede2000_rgb
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from boundary_refinement.models.joint.parallel_joint import load_model
    from boundary_refinement.metrics.segmentation import confusion_matrix,segmentation_metrics
    from boundary_refinement.metrics.color import ciede2000_rgb


def gaussian_ssim_valid(first,second):
    coordinates=torch.arange(-5,6,device=first.device,dtype=first.dtype)
    kernel=torch.exp(-coordinates.square()/(2*1.5**2));kernel=kernel/kernel.sum()
    window=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,11,11)
    def conv(value):return F.conv2d(value,window,groups=3)
    a,b=conv(first),conv(second)
    va,vb=conv(first*first)-a.square(),conv(second*second)-b.square()
    covariance=conv(first*second)-a*b
    return float((((2*a*b+.01**2)*(2*covariance+.03**2))/((a.square()+b.square()+.01**2)*(va+vb+.03**2))).mean())


def image_tensor(path,size=(384,384)):
    with Image.open(path) as im:
        im=im.convert("RGB")
        if size:im=im.resize(size,Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)


def examples(plan,registered):
    own={"val":[],"test":[]};lookup={}
    for row in read(plan["label_manifest"])["records"]:
        if row["partition"] not in own:continue
        if row["partition"]=="test" and row["id"] not in registered["canonical_own_test_ids"]:continue
        clear=image_tensor(row["clear"])
        with Image.open(row["draft_mask"]) as im:target=torch.from_numpy(np.asarray(im.resize((384,384),Image.Resampling.NEAREST)).copy())
        for condition in ("clear","hazy"):
            sample=dict(id=row["id"],condition=condition,image=clear if condition=="clear" else image_tensor(row[condition]),target=target,reference=clear)
            own[row["partition"]].append(sample);lookup[row["id"],condition]=row[condition]
    negatives=[]
    for row in registered["negative_crops"]:
        x,y,r,b=row["box"]
        for condition in ("clear","hazy"):
            # Keep the exact earlier512 crop coordinates before384 resize.
            with Image.open(lookup[row["id"],condition]) as image:
                image=image.convert("RGB").resize((512,512),Image.Resampling.BILINEAR).crop((x,y,r,b)).resize((384,384),Image.Resampling.BILINEAR)
                tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)
            negatives.append(dict(id=f"{row['id']}:{row['index']}",condition=condition,image=tensor,native_size=(384,384)))
    original=read(REPO/"results/architecture-v3b/architecture_plan.json")
    external_manifest=Path(original["previous"])/"external-crosswalk191/manifest.json"
    old_eval=read(REPO/"results/structure-visibility-v6/evaluation_plan.json")
    if digest(external_manifest)!=old_eval["external_manifest_sha256"]:raise ValueError("Crosswalk manifest changed")
    manifest=read(external_manifest);folder=Path(manifest["root"]);external=[]
    for row in manifest["records"]:
        ip,mp=folder/row["image"],folder/row["mask"]
        if digest(ip)!=row["image_sha256"] or digest(mp)!=row["mask_sha256"]:raise ValueError("Crosswalk image changed")
        with Image.open(mp) as image:target=torch.from_numpy((np.asarray(image)>127).astype(np.uint8))
        external.append(dict(id=row["id"],image=image_tensor(ip),target=target,sequence=row["sequence"]))
    negative_root=(WORKSPACE / "benchmark_data" / 'cdset-explicit-negatives-v1')
    if digest(negative_root/"manifest.json")!=old_eval["negative_manifest_sha256"] or digest(negative_root/"label_audit.json")!=old_eval["negative_label_audit_sha256"]:
        raise ValueError("Negative-case audit changed")
    audit=read(negative_root/"label_audit.json");cdset=[]
    for row in read(negative_root/"manifest.json")["records"]:
        if row["name"] not in audit["eligible_ids"]:continue
        if digest(row["path"])!=row["sha256"]:raise ValueError("Negative image changed")
        with Image.open(row["path"]) as image:size=(image.height,image.width)
        cdset.append(dict(id=row["name"],image=image_tensor(row["path"]),native_size=size,sequence=row["filename_prefix"]))
    return own,negatives,external,cdset


def controlled_stress(image,condition):
    if condition=="uniform_t030":
        t=torch.full_like(image[:,:1],.3);a=image.new_tensor([.88,.92,.96])[None,:,None,None]
    elif condition=="smooth_field":
        y,x=torch.meshgrid(torch.linspace(-1,1,image.shape[-2],device=image.device),torch.linspace(-1,1,image.shape[-1],device=image.device),indexing="ij")
        t=(.25+.55*(.5+.25*torch.sin(2*torch.pi*x)+.25*y))[None,None].expand(len(image),1,-1,-1)
        a=image.new_tensor([.86,.90,.94])[None,:,None,None]
    else:raise ValueError("Unknown controlled stress")
    return t*image+(1-t)*a,t,a.expand(len(image),3,1,1)


def correlation(sums):
    n,sx,sy,sxx,syy,sxy=sums
    denom=max(0,n*sxx-sx*sx)*max(0,n*syy-sy*sy)
    return (n*sxy-sx*sy)/(denom**.5) if denom>1e-16 else None


def pooled_summary(rows,partition,condition):
    summary=dict(partition=partition,condition=condition,count=len(rows))
    if "confusion_matrix" in rows[0]:
        matrix=torch.tensor([r["confusion_matrix"] for r in rows],dtype=torch.int64).sum(0)
        metrics=segmentation_metrics(matrix)
        summary.update(miou=metrics["miou"],foreground_iou=metrics["class_iou"][1],confusion_matrix=matrix.tolist(),
            known_fraction=int(matrix.sum())/sum(r["label_pixels"] for r in rows))
    else:
        summary.update(pixel_fpr=sum(r["foreground_pixels"] for r in rows)/sum(r["pixels"] for r in rows),
            image_alarm_fraction=sum(r["foreground_pixels"]>0 for r in rows)/len(rows),
            foreground_pixels=sum(r["foreground_pixels"] for r in rows),pixels=sum(r["pixels"] for r in rows))
    for key in ("rgb_psnr","rgb_ssim","rgb_l1","rgb_ciede2000","rgb_opponent_mae","error_prediction_mae","synthetic_t_mae","synthetic_a_mae"):
        if key in rows[0]:summary[key]=sum(r[key] for r in rows)/len(rows)
    if "error_sums" in rows[0]:
        summary["error_correlation"]=correlation([sum(r["error_sums"][i] for r in rows) for i in range(6)])
        bins=np.asarray([r["semantic_calibration_bins"] for r in rows]).sum(0)
        summary["semantic_calibration_bins"]=bins.tolist()
        summary["known_semantic_ece"]=sum(abs(b[1]-b[2]) for b in bins)/max(1,bins[:,0].sum())
    return summary


def score(model,samples,partition,condition,source=False):
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(samples),4):
            batch=samples[start:start+4]
            image=torch.stack([r["image"] for r in batch]).cuda().float()/255
            truth_t=truth_a=None
            if condition in ("uniform_t030","smooth_field"):image,truth_t,truth_a=controlled_stress(image,condition)
            if condition=="uniform_haze_t055":image=.55*image+.45
            restored,logits,aux=model(image)
            for i,example in enumerate(batch):
                row=dict(id=example["id"])
                native=example["target"].shape if "target" in example else example["native_size"]
                native_logits=F.interpolate(logits[i:i+1],size=native,mode="bilinear",align_corners=False)
                prediction=native_logits.argmax(1)
                if "target" in example:
                    cm=confusion_matrix(prediction,example["target"][None].cuda(),2)
                    row.update(confusion_matrix=cm.tolist(),label_pixels=example["target"].numel())
                else:
                    row.update(foreground_pixels=int((prediction==1).sum()),pixels=prediction.numel())
                if source:
                    reference=example["reference"][None].cuda().float()/255;output=restored[i:i+1].float()
                    mse=(output-reference).square().mean().clamp_min(1e-10)
                    opponent=(output[:,[0,2]]-output[:,1:2])-(reference[:,[0,2]]-reference[:,1:2])
                    row.update(rgb_psnr=float(-10*mse.log10()),rgb_ssim=gaussian_ssim_valid(output,reference),
                        rgb_l1=float((output-reference).abs().mean()),rgb_opponent_mae=float(opponent.abs().mean()),rgb_ciede2000=ciede2000_rgb(output,reference))
                    error=F.adaptive_avg_pool2d((output-reference).abs().mean(1,keepdim=True),aux["error"].shape[-2:]).flatten().double()
                    forecast=aux["error"][i].flatten().double()
                    row["error_prediction_mae"]=float((error-forecast).abs().mean())
                    row["error_sums"]=[error.numel(),float(forecast.sum()),float(error.sum()),float(forecast.square().sum()),float(error.square().sum()),float((forecast*error).sum())]
                    prob=aux["semantic"][i:i+1].float().softmax(1);confidence,category=prob.max(1)
                    coarse=F.interpolate(example["target"][None,None].cuda().float(),prob.shape[-2:],mode="nearest")[:,0].long()
                    valid=coarse!=255;confidence=confidence[valid];correct=(category[valid]==coarse[valid]).float()
                    bins=[]
                    for b in range(10):
                        keep=(confidence>=b/10)&(confidence<(b+1)/10 if b<9 else confidence<=1)
                        bins.append([int(keep.sum()),float(confidence[keep].sum()),float(correct[keep].sum())])
                    row["semantic_calibration_bins"]=bins
                    if truth_t is not None:
                        t=F.interpolate(truth_t[i:i+1],aux["transmission"].shape[-2:],mode="bilinear",align_corners=False)
                        row["synthetic_t_mae"]=float((aux["transmission"][i:i+1]-t).abs().mean())
                        row["synthetic_a_mae"]=float((aux["airlight"][i]-truth_a[i,:,0,0]).abs().mean())
                rows.append(row)
    return dict(summary=pooled_summary(rows,partition,condition),images=rows)


def profile(model):
    counts=[];hooks=[]
    def count(module,args,output):
        if isinstance(module,torch.nn.Conv2d):counts.append(output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1])
        elif isinstance(module,torch.nn.Linear):counts.append(output.numel()*module.in_features)
    for module in model.modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(module.register_forward_hook(count))
    image=torch.rand(1,3,384,384,device="cuda")
    with torch.inference_mode():model(image)
    for hook in hooks:hook.remove()
    with torch.inference_mode():
        for _ in range(10):model(image)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(50):model(image)
        last.record();torch.cuda.synchronize()
    return dict(parameters=sum(p.numel() for p in model.parameters()),conv_linear_MACs=sum(counts),
        cuda_mean_ms=first.elapsed_time(last)/50,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        scope="384FP32batch1,RGB+S+auxiliaryfullforward,10warmup50iterations,Conv/LinearMACsonly;excludesIO/resize")


def warmup():
    model,_=load_model(ROOT/"models/full/seed_42/best.pth")
    with torch.inference_mode():
        for n in (1,2,3,4):model(torch.rand(n,3,384,384,device="cuda"))
    del model;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--lock-only",action="store_true")
    args=parser.parse_args();plan,registered,lock=training_lock()
    if args.lock_only:
        print(json.dumps(dict(jobs=lock["training_jobs"],epochs=lock["training_epochs"],steps=lock["successful_steps"],skips=lock["amp_skips"])));return
    source=Path(__file__).resolve();code_path=ROOT/"evaluation_implementation.json"
    implementation=dict(code_sha256=digest(source),registration_sha256=digest(ROOT/"evaluation_registration.json"),
        helper_sha256={p:digest(SOURCE_ROOT/p) for p in ("boundary_refinement/metrics/color.py","boundary_refinement/metrics/segmentation.py")})
    if code_path.exists() and read(code_path)!=implementation:raise ValueError("Evaluation implementation changed")
    if not code_path.exists():
        write(code_path,implementation);target=ROOT/"source_snapshot/boundary_refinement/evaluation/paired/evaluate_parallel_joint.py";target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    bind();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    warmup();own,negatives,external,cdset=examples(plan,registered)
    summary=[];profiles=[];total=0
    for job in lock["jobs"]:
        method,seed=job["method"],job["seed"];path=ROOT/f"evaluation/{method}/seed_{seed}.json"
        if path.exists():
            payload=read(path)
            if payload["checkpoint_sha256"]!=job["sha256"]:raise ValueError("Saved result weight identity differs")
        else:
            model,packet=load_model(job["checkpoint"]);conditions=[]
            for condition in ("clear","hazy","uniform_t030","smooth_field"):
                samples=[r for r in own["val"] if r["condition"]==(condition if condition in ("clear","hazy") else "clear")]
                conditions.append(score(model,samples,"source_val_partial",condition,True))
            source={r["summary"]["condition"]:r["summary"] for r in conditions}
            for condition in ("clear","hazy"):
                if abs(source[condition]["miou"]-packet["selection"][condition+"_miou"])>.0002:raise ValueError("Source scorer disagrees with registered selection")
                conditions.append(score(model,[r for r in negatives if r["condition"]==condition],"val_absent_crops",condition))
                conditions.append(score(model,[r for r in own["test"] if r["condition"]==condition],"own_test15_partial_exploratory",condition))
            for condition in ("original","uniform_haze_t055"):
                conditions.append(score(model,external,"crosswalk191_exploratory",condition))
                conditions.append(score(model,cdset,"cdset5_exploratory_absence",condition))
            payload=dict(method=method,seed=seed,checkpoint_sha256=job["sha256"],conditions=conditions,profile=profile(model) if seed==42 else None,finished_utc=now())
            if digest(job["checkpoint"])!=job["sha256"]:raise ValueError("Weights changed during inference")
            write(path,payload);del model;torch.cuda.empty_cache()
        for condition in payload["conditions"]:
            summary.append(dict(method=method,seed=seed,**condition["summary"]));total+=len(condition["images"])
        if payload["profile"]:profiles.append(dict(method=method,seed=seed,**payload["profile"]))
        write(ROOT/"evaluation_status.json",dict(models=len(summary)//12,total_models=60,last_method=method,last_seed=seed,updated_utc=now()))
        print(json.dumps(dict(method=method,seed=seed,models=len(summary)//12)),flush=True)
    write(ROOT/"evaluation_summary.json",dict(rows=summary));write(ROOT/"profiles.json",profiles)
    write(ROOT/"evaluation_completed.json",dict(models=60,summary_rows=len(summary),predictions=total,finished_utc=now()))


if __name__=="__main__":main()

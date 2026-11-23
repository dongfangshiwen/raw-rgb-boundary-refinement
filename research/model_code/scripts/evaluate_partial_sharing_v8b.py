"""Use the identical registered V8 scoring functions for two sharing cuts."""
from datetime import datetime,timezone
import json
from pathlib import Path

from scripts import evaluate_parallel_joint_v8 as original

REPO=original.REPO;ROOT=REPO/"results/partial-sharing-v8b";PARENT=original.ROOT
read=original.read;write=original.write;digest=original.digest


def training_lock():
    parent_plan,registration,parent_lock=original.training_lock()
    plan=read(ROOT/"training_plan.json")
    if digest(PARENT/"training_plan.json")!=plan["parent_training_plan_sha256"]:raise ValueError("Parent plan changed")
    for path,sha in plan["source_sha256"].items():
        if digest(REPO/path)!=sha:raise ValueError("Registered sharing code changed")
    jobs=[]
    for method in plan["methods"]:
        for seed in plan["seeds"]:
            folder=ROOT/f"models/{method}/seed_{seed}";job=read(folder/"completed.json")
            if job["epochs"]!=12 or job["plan_sha256"]!=plan["plan_sha256"]:raise ValueError("Supplement incomplete")
            if job["successful_steps"]+job["skipped_steps"]!=432:raise ValueError("Supplement step audit failed")
            if digest(folder/"best.pth")!=job["checkpoint_sha256"]:raise ValueError("Supplement selected weight changed")
            if job["initialization"]!=read(PARENT/f"initialization/seed_{seed}.json"):raise ValueError("Supplement initial heads differ")
            jobs.append(dict(method=method,seed=seed,checkpoint=str((folder/"best.pth").resolve()),sha256=job["checkpoint_sha256"],
                epochs=12,successful_steps=job["successful_steps"],skipped_steps=job["skipped_steps"]))
    lock=dict(jobs=jobs,training_jobs=6,training_epochs=72,successful_steps=sum(r["successful_steps"] for r in jobs),
        amp_skips=sum(r["skipped_steps"] for r in jobs),parent_60weight_lock_sha256=digest(PARENT/"training_lock.json"),
        source_plan_sha256=digest(ROOT/"training_plan.json"))
    output=ROOT/"training_lock.json"
    if output.exists():
        before=read(output);before.pop("created_utc",None)
        if before!=lock:raise ValueError("Supplement weight lock changed")
    else:write(output,dict(created_utc=datetime.now(timezone.utc).isoformat(),**lock))
    # Source/target scoring and benchmark manifest definitions remain identical.
    registration=dict(registration,rgb_benchmarks=dict(registration["rgb_benchmarks"],methods=list(plan["methods"])))
    return parent_plan,registration,lock


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--lock-only",action="store_true");args=parser.parse_args()
    plan,registration,lock=training_lock()
    if args.lock_only:print(json.dumps(dict(jobs=6,epochs=72,successful_steps=lock["successful_steps"],skips=lock["amp_skips"])));return
    implementation=dict(source_sha256=digest(__file__),scoring_sha256=digest(REPO/"scripts/evaluate_parallel_joint_v8.py"),
        registration_sha256=digest(PARENT/"evaluation_registration.json"),model_sha256=digest(REPO/"scripts/partial_sharing_v8b.py"))
    path=ROOT/"evaluation_implementation.json"
    if path.exists() and read(path)!=implementation:raise ValueError("Supplement evaluator changed")
    if not path.exists():write(path,implementation)
    original.bind();torch=original.torch
    from scripts.partial_sharing_v8b import load_model
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_=load_model(lock["jobs"][0]["checkpoint"])
    with torch.inference_mode():
        for n in (1,2,3,4):model(torch.rand(n,3,384,384,device="cuda"))
    del model;torch.cuda.empty_cache()
    own,negatives,external,cdset=original.examples(plan,registration);summaries=[];total=0;profiles=[]
    for job in lock["jobs"]:
        path=ROOT/f"evaluation/{job['method']}/seed_{job['seed']}.json"
        if path.exists():
            payload=read(path)
            if payload["checkpoint_sha256"]!=job["sha256"]:raise ValueError("Cached supplement weight identity changed")
        else:
            model,packet=load_model(job["checkpoint"]);conditions=[]
            for condition in ("clear","hazy","uniform_t030","smooth_field"):
                samples=[r for r in own["val"] if r["condition"]==(condition if condition in ("clear","hazy") else "clear")]
                conditions.append(original.score(model,samples,"source_val_partial",condition,True))
            for scored in conditions[:2]:
                condition=scored["summary"]["condition"]
                if abs(scored["summary"]["miou"]-packet["selection"][condition+"_miou"])>.0002:raise ValueError("Supplement selection scoring differs")
            for condition in ("clear","hazy"):
                conditions.append(original.score(model,[r for r in negatives if r["condition"]==condition],"val_absent_crops",condition))
                conditions.append(original.score(model,[r for r in own["test"] if r["condition"]==condition],"own_test15_partial_exploratory",condition))
            for condition in ("original","uniform_haze_t055"):
                conditions.append(original.score(model,external,"crosswalk191_exploratory",condition))
                conditions.append(original.score(model,cdset,"cdset5_exploratory_absence",condition))
            payload=dict(method=job["method"],seed=job["seed"],checkpoint_sha256=job["sha256"],conditions=conditions,
                profile=original.profile(model) if job["seed"]==42 else None,finished_utc=datetime.now(timezone.utc).isoformat())
            write(path,payload);del model;torch.cuda.empty_cache()
        for condition in payload["conditions"]:
            summaries.append(dict(method=job["method"],seed=job["seed"],**condition["summary"]));total+=len(condition["images"])
        if payload["profile"]:profiles.append(dict(method=job["method"],seed=job["seed"],**payload["profile"]))
        print(json.dumps(dict(method=job["method"],seed=job["seed"])),flush=True)
    write(ROOT/"evaluation_summary.json",dict(rows=summaries));write(ROOT/"profiles.json",profiles)
    write(ROOT/"evaluation_completed.json",dict(models=6,summary_rows=len(summaries),predictions=total,finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=="__main__":main()

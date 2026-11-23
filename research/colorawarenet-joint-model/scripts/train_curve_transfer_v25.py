"""Run data, mapping and supervision ablations with three verified V24 starts."""
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import time

import torch

from dehaze_seg.utils import set_seed
from scripts.appearance_refiner_v22 import appearance_loss
from scripts.prepare_rw2ah_v25 import ROOT
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import ROOT as V24
from scripts.real_haze_data_v22 import load_data
from scripts.tone_curves_v25 import CurveRestoration, KNOTS, affine_rgb, edge_loss, positive_diagonal, sample_curves
from scripts.train_real_detail_v24 import start_model
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


EPOCHS = 8
BATCH = 4
EDGE_WEIGHT = .20
METHODS = ("direct_nh", "direct_mixed", "affine_field", "curves", "monotone", "uniform_edges", "visible_edges")
CURVE_METHODS = METHODS[3:]


def starts():
    upstream = read(V24/"completed.json")
    initial = {original["video"]: (original, final) for original, final in read(V24/"plan.json")["starts"]}
    if not upstream["all_complete"] or len(initial)!=3 or len(upstream["runs"])!=3:
        raise ValueError("The experiment requires all three V24 initialization fits")
    return [(run, *initial[run["video"]]) for run in upstream["runs"]]


def initial_model(run, original, final):
    if digest(run["checkpoint"])!=run["checkpoint_SHA256"]:
        raise ValueError("V24 initialization checksum differs")
    saved = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    model = start_model(original, final)
    model.load_state_dict(saved["models"]["mixed_scale"], strict=True)
    return model


def build(network, method):
    return CurveRestoration(network, monotone=method=="monotone") if method in CURVE_METHODS else network


def predict(network, method, image):
    if method=="affine_field":
        return affine_rgb(network, image)
    if method in CURVE_METHODS:
        return network(image)
    return author_rgb(network, image)


def training_frames():
    manifest = read(ROOT/"data_manifest.json")
    if not manifest["all_passed"] or digest(manifest["train"]["file"])!=manifest["train"]["SHA256"]:
        raise ValueError("RW2AH training manifest or archive failed")
    first = load_data("training")["training"]
    second = torch.load(manifest["train"]["file"], map_location="cpu", weights_only=True)
    if [v["id"] for v in first]!=[f"{i:02d}" for i in range(1,46)] or len(second)!=manifest["train"]["count"]:
        raise ValueError("Unexpected original RGB training list")
    if set(v["id"].split("-")[0] for v in second) & set(manifest["test"]["prefixes"]):
        raise ValueError("RW2AH held prefix is in the train cache")
    source = [*first,*second]
    pixels = [torch.stack([v[role] for v in source]).to("cuda") for role in ("hazy","clean")]
    roi = torch.zeros_like(pixels[0][:,0], dtype=torch.bool)
    for i,v in enumerate(source):
        left, top, right, bottom = v["roi"]
        roi[i,top:bottom,left:right] = True
    return (*pixels,roi), len(first)


def batches(frames, old_count, generator):
    all_count = len(frames[0])
    common = torch.randperm(all_count, generator=generator)
    draws = torch.cat([torch.randperm(old_count,generator=generator) for _ in range((all_count+old_count-1)//old_count)])
    flips = torch.rand(all_count, generator=generator)<.5
    for mix, nh, flip in zip(common.split(BATCH), draws[:all_count].split(BATCH), flips.split(BATCH)):
        outputs = []
        for indices in (nh,mix):
            flip_gpu = flip.cuda()
            parts = [v[indices].contiguous() for v in frames]
            outputs.append((*(torch.where(flip_gpu[:,None,None,None],x.flip(-1),x).float()/255 for x in parts[:2]),
                            torch.where(flip_gpu[:,None,None],parts[2].flip(-1),parts[2])))
        yield outputs


def register(entries, old_count, size):
    checked = [ROOT/"data_manifest.json", V24/"completed.json", V24/"plan.json",
               *(REPO/"scripts"/p for p in ("train_curve_transfer_v25.py","tone_curves_v25.py",
                                           "prepare_rw2ah_v25.py","appearance_refiner_v22.py","camvid_data_v15.py"))]
    plan = dict(methods=list(METHODS), starts=[r for r,_,_ in entries], batch=BATCH, epochs=EPOCHS,
                nh_train=old_count, rw2ah_train=size-old_count, steps_per_fit=EPOCHS*((size+BATCH-1)//BATCH),
                matched_data="Each epoch draws each mixed image once. The NH-only model uses balanced shuffled repetitions of the same 45 pairs and step count. Mixed-data arms use identical paired views and shared flips.",
                controls="Direct goes through the author's original full-image decoder. Affine and curve heads use identical half-size localization, capped at 256 pixels. Curves versus affine isolates the richer mapping head; monotone and edge arms use the same curve parameter count. Uniform and visibility-weighted edges use the same weight.",
                training="CUDA BF16 with FP32 grid sampling, 256x192 letterboxed pairs, appearance_loss, AdamW 0.0002 to 0.00002 cosine, wd 0.01, norm clip 1.0. Final epoch only.",
                edge_weight=EDGE_WEIGHT, prior_exposure="All initial checkpoints have seen NH train and CamVid, with no learned RW2AH exposure. All test decisions are frozen before training.",
                assessment="Image-level paired PSNR, SSIM, gradient MAE and channel bias on held RW2AH and the four earlier paired benchmarks. Author PGL and MCT are published references with different training. The three held RW2AH filename prefixes are the test strata, with no other semantic labels claimed.",
                source_SHA256={str(p):digest(p) for p in checked})
    path = ROOT/"plan.json"
    if path.exists() and read(path)!=plan:
        raise ValueError("V25 registered plan changed")
    if not path.exists():
        atomic_write(path,plan)
    return digest(path)


def update(network, optimizer, method, batch):
    image, target, valid = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        restored = predict(network,method,image)
        loss = appearance_loss(restored,target,valid)
        if method in ("uniform_edges","visible_edges"):
            loss = loss + EDGE_WEIGHT*edge_loss(image,restored,target,valid,method=="visible_edges")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def opt(network):
    return torch.optim.AdamW(network.parameters(),lr=2e-4,weight_decay=.01)


def preflight(parent, frames, old_count):
    base = initial_model(*parent)
    nh, mix = next(batches(frames,old_count,torch.Generator().manual_seed(2500)))
    models = {k:build(deepcopy(base),k) for k in METHODS}
    with torch.no_grad():
        affine_error = (predict(models["affine_field"],"affine_field",mix[0])-models["curves"](mix[0])).abs().max().item()
        raw = torch.randn(2,3,3,KNOTS,5,7,device="cuda")
        projected = positive_diagonal(raw)
        diagonal = torch.stack([projected[:,c,c] for c in range(3)],1)
        if affine_error>4e-6 or diagonal.diff(dim=2).min() < -1e-6:
            raise ValueError(f"Affine initialization or monotone increments failed {affine_error}")
        mean_error = max((raw[:,c,c].mean(1)-projected[:,c,c].mean(1)).abs().max().item() for c in range(3))
        if mean_error>1e-6 or any(not torch.equal(raw[:,i,j],projected[:,i,j]) for i in range(3) for j in range(3) if i!=j):
            raise ValueError("Constraint changed cross-channel terms or own-channel average")
        from scripts.evaluate_parallel_rgb_v8 import bind
        bind()
        from scripts.evaluate_author_mct_v8 import load_author_reference
        mct,_ = load_author_reference()
        example = mix[0][:2,:,:,::3].contiguous()*2-1
        err_mct = (sample_curves(example,raw)-mct.mapping(example,raw.flatten(1,3))).abs().max().item()
        if err_mct>4e-6:
            raise ValueError(f"Author MCT mapping contract differs {err_mct}")
    losses = {k:update(model,opt(model),k,nh if k=="direct_nh" else mix).item() for k,model in models.items()}
    record = dict(passed=True,affine_max_error=affine_error,MCT_author_max_error=err_mct,
                  monotone_mean_max_error=mean_error,discarded_updates=len(losses),losses=losses,
                  torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),finished_utc=now())
    atomic_write(ROOT/"smoke.json",record)
    print(record,flush=True)


def fit(parent, frames, old_count, plan):
    prior = parent[0]
    video = prior["video"]
    folder = ROOT/video
    folder.mkdir(exist_ok=True)
    log, checkpoint = folder/"training.json", folder/"trained.pth"
    if log.exists():
        record = read(log)
        if digest(checkpoint)!=record["checkpoint_SHA256"] or record["plan_SHA256"]!=plan:
            raise ValueError("Completed V25 checkpoint changed")
        return record
    seed = 25_000_000 + sum(video.encode())
    set_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    base = initial_model(*parent)
    models = {n:build(deepcopy(base),n).train() for n in METHODS}
    optimizers = {n:opt(model) for n,model in models.items()}
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o,EPOCHS,eta_min=2e-5) for o in optimizers.values()]
    counts = {n:sum(p.numel() for p in model.parameters()) for n,model in models.items()}
    steps = 0
    started = time.perf_counter()
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,EPOCHS+1):
        totals = []
        for nh,mix in batches(frames,old_count,rng):
            totals.append(torch.stack([update(model,optimizers[n],n,nh if n=="direct_nh" else mix)
                                       for n,model in models.items()]))
            steps += 1
        for sched in schedulers:
            sched.step()
        means = torch.stack(totals).mean(0).tolist()
        row = dict(video=video,epoch=epoch,steps_per_fit=steps,losses=dict(zip(METHODS,means)),
                   seconds=round(time.perf_counter()-started,2),peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,3),
                   updated_utc=now())
        atomic_write(ROOT/"progress.json",row)
        history.append(row)
        print(row,flush=True)
    if steps!=EPOCHS*((len(frames[0])+BATCH-1)//BATCH):
        raise ValueError("V25 matched update budget differed")
    torch_save(checkpoint,dict(format="curve-transfer-v25",video=video,plan_SHA256=plan,
                               models={n:model.state_dict() for n,model in models.items()}))
    atomic_write(folder/"curves.json",history)
    record = dict(video=video,seed=seed,checkpoint=str(checkpoint),checkpoint_SHA256=digest(checkpoint),
                  source_SHA256=prior["checkpoint_SHA256"],plan_SHA256=plan,steps_per_fit=steps,
                  parameter_counts=counts,finished_utc=now())
    atomic_write(log,record)
    return record


def load_fits():
    complete = read(ROOT/"completed.json")
    parent_table = {v[0]["video"]:v for v in starts()}
    if not complete["all_complete"] or digest(ROOT/"plan.json")!=complete["plan_SHA256"]:
        raise ValueError("V25 training is incomplete or changed")
    models = {}
    for row in complete["runs"]:
        path = Path(row["checkpoint"])
        if digest(path)!=row["checkpoint_SHA256"]:
            raise ValueError("V25 evaluation checkpoint changed")
        saved = torch.load(path,map_location="cpu",weights_only=True)
        base = initial_model(*parent_table[row["video"]])
        models[row["video"]] = {}
        for method in METHODS:
            fitted = build(deepcopy(base),method)
            fitted.load_state_dict(saved["models"][method],strict=True)
            models[row["video"]][method] = fitted.eval()
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    opts = parser.parse_args()
    configure_cuda()
    frames, old_count = training_frames()
    entries = starts()
    if opts.smoke:
        preflight(entries[0],frames,old_count)
        return
    if not read(ROOT/"smoke.json")["passed"]:
        raise ValueError("V25 GPU preflight required")
    plan = register(entries,old_count,len(frames[0]))
    runs = []
    for parent in entries:
        runs.append(fit(parent,frames,old_count,plan))
        gc.collect()
        torch.cuda.empty_cache()
    complete = dict(all_complete=True,runs=runs,fits=len(runs)*len(METHODS),
                    updates=sum(r["steps_per_fit"] for r in runs)*len(METHODS),plan_SHA256=plan,finished_utc=now())
    atomic_write(ROOT/"completed.json",complete)
    print({k:v for k,v in complete.items() if k!="runs"},flush=True)


if __name__=="__main__":
    main()

"""Preregistered matched CUDA screening of a trainable non-serial RGB/S model."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dehaze_seg.data.splits import file_digest
from dehaze_seg.metrics import confusion_matrix,segmentation_metrics
from dehaze_seg.utils import set_seed,runtime_metadata
from scripts.parallel_joint_v8 import METHODS,FACTORIAL,ParallelJoint,joint_loss

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/"results/parallel-joint-v8"
LABELS=REPO/"protocols/crosswalk-partial-background-v8/manifest.json"
SIZE=384;EPOCHS=12;SEEDS=(42,43,44)


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def parameter_digest(module):
    digest=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_plan():
    parent=read(REPO/"results/vision-baseline-v1/training_plan.json")
    sources=[Path(__file__).resolve(),REPO/"scripts/parallel_joint_v8.py",REPO/"scripts/prepare_parallel_data_v8.py",
        REPO/"scripts/vision_baseline_v1.py",REPO/"scripts/partial_joint_v7.py"]
    value=dict(methods={k:asdict(v) for k,v in METHODS.items()},factorial=list(FACTORIAL),seeds=list(SEEDS),epochs=EPOCHS,
        resize=[SIZE,SIZE],scene_batch=4,views_per_scene=["clear","original_hazy","controlled_synthetic"],updates_per_epoch=36,
        label_manifest=str(LABELS.resolve()),label_manifest_sha256=file_digest(LABELS),
        split_file=parent["split_file"],split_sha256=file_digest(parent["split_file"]),pretrained=parent["pretrained"],
        source_sha256={p.relative_to(REPO).as_posix():file_digest(p) for p in sources},
        architecture="Official torchvision ImageNet MobileNetV3-Large shared encoder and DeepLabV3 semantic head, plus trainable RGB FPN decoder. Inference consumes RGB only. RGB never feeds S; reciprocal feature messages are simultaneous before decoders. Independent control duplicates encoder with identical initialization.",
        factors={"A":"Bounded normalized-feature residual for semantics, mean/logstd statistics bypass for RGB. Not a proof of uniquely identifiable content/weather factors; raw-adapter and no-stats controls required.",
            "B":"Detached calibrated reconstruction-error and coarse-semantic-entropy reliability gates for reciprocal pre-decoding features. Reliability prediction can be wrong; unweighted, learnedtasklossgate(without errorcalibration), spatially shuffled and one-way controls required.",
            "C":"Same clear/hazy/synthetic examples in every method; normalized paired semantic features plus controlled synthetic t/A auxiliary supervision. Only synthetic encoded-RGB intervention parameters are known, never real transmission/depth. Feature-only and weather-only controls separate terms."},
        augmentation="Each144trainingIDs once/epoch;4scenes perstep, aligned independenthorizontalflips acrossviews. Equal3views in allmethods. Mean transmission stratified6bins [.4,.95]; 4x4 zero-mean bounded shape with amplitude .5min(mu-.25,.95-mu); globalRGBairlight U(.75,1). Exactsynthetic operator I=tJ+(1-t)A in encodedRGB, not asserted radiometrically exactreal fog.",
        loss="Joint CE+macroDice on knownpixels +.1coarseCE/Dice + RGBL1+.1opponentMAE+.05spatialgradientL1 +.05detached-error calibrationMAE; Cadds10normalized-featureMSE +.1synthetic t/A MAE. Single-task controls drop other task losses, not examples.",
        optimization="AlltrainablefromfreshImageNetencoder; sameinitialsemanticandRGBheads, AdamW1e-4 wd1e-5 clip1 AMP, cosineLR to1e-6 over12epochs. No persistentoptimizer/laststates. This12epoch study is developmental screening, not a converged author-recipe reproduction.",
        selection="Epoch0included. Joint: prioritize sourcehazyPSNR >=rawhazy+.2dB, then max min(clear,hazy) knownmIoU, tie hazyPSNR and earlier. If noRGBfeasibleepoch, markfailedcriterion. RGBonly max sourcehazyPSNR; Segonly max sourceknownworst. No stress/negative/test/external selection.",
        hypotheses={"A":"factor vsjoint: knownworst>=+.5pp andhazyPSNR>=-.1dB, >=2positiveSseeds; vsrawadapter knownworst>=+.25pp; statsbypass vsnostats improvesPSNR>=+.1dB orcolorerror>=2percent;fullleaveout directionpositive",
            "B":"full vsfactor_intervention andfull_ungated: knownworst>=+.5pp,hazyPSNR>=-.1dB,>=2positiveSseeds; compareoneway/shuffled explicitly; predictedRGBerror positivelycorrelatedwithactualerror. No novelty inferred from gate alone.",
            "C":"full vsfactor_exchange: meanfixed-source-stressknownmIoU>=+.5pp,hazyPSNR>=-.1dB,>=2positiveseeds; comparefeature/weatheronly; synthetic t/A diagnostics notrealdepthclaim",
            "overall":"full vsjoint must improveknownworst>=+.5pp andhazyPSNR>=+.2dB, RGB CIEDE2000 notincrease>2percent, >=2positiveseeds forboth. Practicalgates are developmental, not significanceacrosslocations or noveltyproof."},
        post_lock_evaluation="All60selectedweights locked BEFORE new stress/absence/external inference. Sourceclear/hazy, uniformt.30 coloredveil andnonuniformt[.25,.8], nativeCrosswalk191original/t.55veil andCDSet5absent, owncanonical15 partialmasks. PairedSOTS/HSTS RGB only forsource-registeredjoint/full/rgb_only x3seeds plusrawinput; nativeRGBinference pad16; nosegtruth assumed. Prior targets inspected, all exploratory.",
        limitations="185frames possiblyoneintersection; originalall-visiblemaskincomplete. V8 recoversauditedtop20percent backgroundbut44.03percentpixelsstillunknown. AIbackgroundreview isnotcompletedhumangroundtruth. Differentlabel/resolution/viewbudget thanV7, so do not attribute crossround differences toarchitecture.",
        disk="uint8RAM sourceRGBcache only; selectedfullnetworkweights withhash andcompactlogs; nooptimizerorper-epochweights. Prior core/data/results unchanged.",
        prior_art=[
            dict(name="Dehaze-Seg",url="https://github.com/aresdrw/dehaze-seg",scope="OfficialREADMEandprimarypreprintabstractread; nofullofficialcheckpoint reproduction inthis environment"),
            dict(name="ALLNetCVPR2026",url="https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_ALLNet_Multi-task_Dense_Prediction_for_Degraded_Images_CVPR_2026_paper.pdf",scope="Primaryindexedabstractread, directfulltext403; restoration/densepredictionbidirectionalcollaborationalreadyexists"),
            dict(name="DomainSeparationNetworks",url="https://arxiv.org/abs/1608.06019",scope="Shared/private latentfactorization isestablished"),
            dict(name="IBN-Net",url="https://arxiv.org/abs/1807.09441",scope="Appearance-invariantnormalizationis established"),
            dict(name="RobustNet",url="https://arxiv.org/abs/2103.15597",scope="Style/contentwhiteningforsemanticgeneralizationis established"),
            dict(name="FIFO",url="https://arxiv.org/abs/2204.01587",scope="Fog-invariantfeaturesare established"),
            dict(name="PCGrad",url="https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf",scope="Gradientconflicthandlingis established, notclaimedhere asinnovation"),
        ])
    value["plan_sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    output=ROOT/"training_plan.json"
    if output.exists():
        previous=read(output);previous.pop("created_utc",None)
        if previous!=value:raise ValueError("Registered V8 plan/code changed; create a new version")
    else:
        write(output,dict(created_utc=now(),**value))
        for source in sources:
            target=ROOT/"source_snapshot"/source.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    return value


def prepare_data():
    records=read(LABELS)["records"];data={}
    for part in ("train","val"):
        rows=[r for r in records if r["partition"]==part];rgb=[];masks=[]
        for row in rows:
            pair=[]
            for role in ("clear","hazy"):
                with Image.open(row[role]) as image:
                    pair.append(torch.from_numpy(np.asarray(image.convert("RGB").resize((SIZE,SIZE),Image.Resampling.BILINEAR)).copy()).permute(2,0,1))
            if file_digest(row["draft_mask"])!=row["draft_sha256"]:raise ValueError("Used label changed")
            with Image.open(row["draft_mask"]) as image:
                masks.append(torch.from_numpy(np.asarray(image.resize((SIZE,SIZE),Image.Resampling.NEAREST)).copy()))
            rgb.append(torch.stack(pair))
        data[part]=dict(raw=torch.stack(rgb),mask=torch.stack(masks),ids=[r["id"] for r in rows])
    return data


def synthesize(clear,random_generator,strata):
    n=len(clear);device=clear.device
    mu=(.4+.55*(strata.float()+torch.rand(n,generator=random_generator))/6).to(device)[:,None,None,None]
    noise=torch.rand((n,1,4,4),generator=random_generator).to(device)*2-1
    noise=noise-noise.mean((2,3),keepdim=True)
    noise=noise/noise.abs().amax((2,3),keepdim=True).clamp_min(1e-6)
    amplitude=.5*torch.minimum(mu-.25,.95-mu)
    transmission=mu+amplitude*F.interpolate(noise,clear.shape[-2:],mode="bilinear",align_corners=False)
    airlight=(.75+.25*torch.rand((n,3,1,1),generator=random_generator)).to(device)
    return transmission*clear+(1-transmission)*airlight,transmission,airlight


def make_batch(values,indices,generator,strata):
    raw=values["raw"][indices].cuda().float()/255
    mask=values["mask"][indices].cuda().long()
    flips=(torch.rand(len(indices),generator=generator)>.5).to(raw.device)
    raw=torch.where(flips[:,None,None,None,None],raw.flip(-1),raw)
    mask=torch.where(flips[:,None,None],mask.flip(-1),mask)
    clear,hazy=raw[:,0],raw[:,1]
    synthetic,t,a=synthesize(clear,generator,strata)
    return torch.cat((clear,hazy,synthetic)),clear.repeat(3,1,1,1),mask.repeat(3,1,1),t,a


@torch.inference_mode()
def validate(model,values):
    model.eval();result={}
    for style,condition in enumerate(("clear","hazy")):
        cm=torch.zeros(2,2,dtype=torch.int64);psnrs=[];l1=[];opponent=[];input_psnrs=[]
        for start in range(0,len(values["ids"]),4):
            raw=values["raw"][start:start+4].cuda().float()/255
            image=raw[:,style];target=values["mask"][start:start+4].cuda().long();reference=raw[:,0]
            restored,logits,_=model(image)
            # The repository metric excludes targets outside [0,num_classes), including255.
            cm+=confusion_matrix(logits.argmax(1),target,2)
            mse=(restored-reference).square().mean((1,2,3));psnrs.extend((-10*mse.clamp_min(1e-10).log10()).tolist())
            input_psnrs.extend((-10*(image-reference).square().mean((1,2,3)).clamp_min(1e-10).log10()).tolist())
            l1.extend((restored-reference).abs().mean((1,2,3)).tolist())
            op=(restored[:,[0,2]]-restored[:,1:2])-(reference[:,[0,2]]-reference[:,1:2])
            opponent.extend(op.abs().mean((1,2,3)).tolist())
        metrics=segmentation_metrics(cm)
        result[condition+"_miou"]=float(metrics["miou"])
        result[condition+"_foreground_iou"]=metrics["class_iou"][1]
        result[condition+"_confusion_matrix"]=cm.cpu().tolist()
        result[condition+"_rgb_psnr"]=float(np.mean(psnrs));result[condition+"_rgb_l1"]=float(np.mean(l1))
        result[condition+"_rgb_opponent_mae"]=float(np.mean(opponent));result[condition+"_input_psnr"]=float(np.mean(input_psnrs))
    result["worst_miou"]=min(result["clear_miou"],result["hazy_miou"])
    result["rgb_feasible"]=result["hazy_rgb_psnr"]>=result["hazy_input_psnr"]+.2
    result["known_pixels_only"]=True
    return result


def selection_key(stats,task):
    if task=="rgb":return (1,stats["hazy_rgb_psnr"],-stats["hazy_rgb_l1"])
    if task=="seg":return (0,stats["worst_miou"],stats["hazy_rgb_psnr"])
    return (int(stats["rgb_feasible"]),stats["worst_miou"],stats["hazy_rgb_psnr"])


def warmup():
    # Before RAM cache allocation: avoids Windows CUDA compiler commit spikes.
    model=ParallelJoint(42,"full",pretrained=True).cuda().train()
    image=torch.rand(12,3,SIZE,SIZE,device="cuda");target=torch.randint(0,2,(12,SIZE,SIZE),device="cuda")
    with torch.autocast("cuda",dtype=torch.float16):
        loss,_=joint_loss(model,image,image,target,4,torch.full((4,1,SIZE,SIZE),.6,device="cuda"),torch.full((4,3,1,1),.9,device="cuda"))
    loss.backward();torch.cuda.synchronize()
    del model,image,target,loss
    torch.cuda.empty_cache()


def run(plan,data,method,seed):
    folder=ROOT/f"models/{method}/seed_{seed}"
    if (folder/"completed.json").exists():
        previous=read(folder/"completed.json")
        if previous["plan_sha256"]!=plan["plan_sha256"] or file_digest(folder/"best.pth")!=previous["checkpoint_sha256"]:
            raise ValueError("Completed run changed")
        return
    if folder.exists() and any(folder.iterdir()):raise ValueError("Preserve an incomplete V8 run; do not silently restart")
    set_seed(seed);model=ParallelJoint(seed,method,True).cuda()
    initial={"encoder":parameter_digest(model.encoder),"semantic_head":parameter_digest(model.segmenter),"rgb_head":parameter_digest(model.rgb_refine)}
    initial_file=ROOT/f"initialization/seed_{seed}.json"
    if initial_file.exists() and read(initial_file)!=initial:raise ValueError("Matched initializations differ")
    if not initial_file.exists():write(initial_file,initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda");successful=0;skipped=0
    stats=validate(model,data["val"]);best=selection_key(stats,model.variant.task);selected_epoch=0
    folder.mkdir(parents=True,exist_ok=True)
    def save(epoch,stats):
        torch.save(dict(format="parallel-joint-v8",method=method,seed=seed,epoch=epoch,plan_sha256=plan["plan_sha256"],
            model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},selection=stats,
            research_only=True,complete_all_visible_labels=False,trained_rgb_branch=model.variant.task!="seg"),folder/"best.pth")
    save(0,stats);rows=[dict(epoch=0,**stats)]
    start=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train();generator=torch.Generator().manual_seed(seed*1_000_000+epoch)
        order=torch.randperm(144,generator=generator);strata=torch.randperm(144,generator=generator)%6
        losses=[]
        for step in range(36):
            indices=order[step*4:step*4+4]
            image,reference,mask,t,a=make_batch(data["train"],indices,generator,strata[step*4:step*4+4])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):loss,detail=joint_loss(model,image,reference,mask,4,t,a)
            if not torch.isfinite(loss):raise ValueError("Nonfinite V8 loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale:skipped+=1
            else:successful+=1
            losses.append(detail)
        scheduler.step();stats=validate(model,data["val"]);key=selection_key(stats,model.variant.task)
        if key>best:
            best=key;selected_epoch=epoch;save(epoch,stats)
        row=dict(epoch=epoch,learning_rate=optimizer.param_groups[0]["lr"],successful_steps=successful,skipped_steps=skipped,
            loss={k:float(np.mean([v[k] for v in losses])) for k in losses[0]},**stats)
        rows.append(row);write(folder/"metrics.json",rows)
        write(ROOT/"training_status.json",dict(status="training",method=method,seed=seed,epoch=epoch,epochs=EPOCHS,updated_utc=now()))
        print(json.dumps(dict(method=method,seed=seed,epoch=epoch,known_worst=round(stats["worst_miou"],5),hazy_psnr=round(stats["hazy_rgb_psnr"],3),best_epoch=selected_epoch)),flush=True)
    selected=torch.load(folder/"best.pth",map_location="cpu",weights_only=True)
    done=dict(method=method,seed=seed,epochs=EPOCHS,plan_sha256=plan["plan_sha256"],expected_steps=EPOCHS*36,
        successful_steps=successful,skipped_steps=skipped,selected_epoch=selected_epoch,selection=selected["selection"],
        checkpoint_sha256=file_digest(folder/"best.pth"),initialization=initial,final_rgb_head_sha256=parameter_digest(model.rgb_refine),
        final_encoder_sha256=parameter_digest(model.encoder),elapsed_seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()),finished_utc=now())
    write(folder/"completed.json",done)
    del selected,model,optimizer,scheduler,scaler
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan-only",action="store_true")
    args=parser.parse_args();plan=make_plan()
    if args.plan_only:
        print(json.dumps(dict(methods=len(METHODS),jobs=len(METHODS)*3,epochs=len(METHODS)*3*EPOCHS,plan_sha256=plan["plan_sha256"])));return
    if not torch.cuda.is_available():raise RuntimeError("CUDA PyTorch required")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    write(ROOT/"runtime.json",runtime_metadata(torch.device("cuda")));warmup();data=prepare_data()
    if len(data["train"]["ids"])!=144 or len(data["val"]["ids"])!=25:raise ValueError("Expected fixed144/25source split")
    for seed in SEEDS:
        for method in METHODS:run(plan,data,method,seed)
    write(ROOT/"training_status.json",dict(status="complete",jobs=len(METHODS)*3,epochs=len(METHODS)*3*EPOCHS,finished_utc=now()))


if __name__=="__main__":main()

"""Test real-fog semantic adaptation with supervision for local restoration utility."""
import argparse
from collections import defaultdict
from hashlib import sha256
import re

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.metrics.segmentation import confusion_matrix, segmentation_metrics
from boundary_refinement.utils import set_seed
from boundary_refinement.data.cohorts.audit_real_fog import city_train_ids
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import DRIVING, ZURICH, load_pairs, load_segmenter, resize_image, verified_image
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_haze_data import overlap_check
from boundary_refinement.data.staging.stage_fusion_holdouts import witness
from boundary_refinement.training.structure_transfer import ROOT as RGB_ROOT, SEEDS, completed, load_fit
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = RGB_ROOT / "parallel-semantics"
ARMS = ("raw_adapter","correction_adapter","utility_gate","shifted_gate")
MAIN = "utility_gate"
STEPS = 192
WINDOW = 96
BATCH = 4
CHECKPOINT = ROOT / "weights.pth"
PROTOCOL = ROOT / "protocol.json"


def capture_group(key):
    timestamp = re.match(r"(public|pedestrian)_(\d{8})_",key)
    if timestamp:
        return "_".join(timestamp.groups())
    sequence = re.match(r"web_(\d{6})\d+",key)
    return "web_"+sequence[1] if sequence else key


def verified_split():
    all_rows = read(DRIVING)["rows"]
    coarse = [v for v in all_rows if v["annotation_kind"]=="gtCoarse"]
    fine = [v for v in all_rows if v["annotation_kind"]=="gtFine"]
    if (len(coarse),len(fine)) != (68,33):
        raise ValueError("Unexpected Foggy Driving annotation split")
    held_groups = {capture_group(v["id"]) for v in fine}
    dates = [r["id"] for r in coarse if capture_group(r["id"]) in held_groups]
    candidates = [r for r in coarse if r["id"] not in dates]
    native_held = []
    for domain in ("foggy_zurich_testv2","foggy_driving_fine"):
        for name,_,rgb,_ in load_pairs(domain):
            native_held.append(witness(domain+"/"+name,Image.fromarray(rgb),sha256(rgb.tobytes()).hexdigest()))
    train_refs = []
    for row in candidates:
        rgb = verified_image(row["rgb"],row["rgb_SHA256"],False)
        train_refs.append(witness(row["id"],Image.fromarray(rgb),sha256(rgb.tobytes()).hexdigest()))
    flags,nearest = overlap_check(native_held,train_refs)
    excluded = {r["id"] for r in flags}
    available = [r for r in candidates if r["id"] not in excluded]
    groups = sorted({capture_group(r["id"]) for r in available},
                    key=lambda k:sha256(("parallel-utility-v32/"+k).encode()).digest())
    reserved = set(groups[:6])
    fit = [r for r in available if capture_group(r["id"]) not in reserved]
    val = [r for r in available if capture_group(r["id"]) in reserved]
    return fit,val,dict(same_capture_exclusions=dates,overlap_flags=flags,nearest=nearest,
                        fit_ids=[r["id"] for r in fit],validation_ids=[r["id"] for r in val],
                        scope="Local research split of Foggy Driving gtCoarse. Exclude pedestrian capture dates represented in gtFine and flagged perceptual duplicates with both tests. Reserve six coarse capture groups before fitting. The full 101-image conventional all-test protocol differs.")


def backbones():
    fits = completed()
    record = next(r for r in fits["runs"] if r["arm"]=="point" and r["seed"]==SEEDS[0])
    return (*load_segmenter(),load_fit(load_pgl()[0],record),record)


def vocabulary(segmenter, mean, std, x):
    return segmenter(pixel_values=(x-mean)/std).logits.float()


def proxy_maps(raw, restored, base):
    size = base.shape[-2:]
    small = F.adaptive_avg_pool2d(raw,size)

    def maps(x):
        return torch.cat(((3*x).clamp(-2,2),(3*x.abs()).clamp_max(2)),dim=1)

    raw_delta = small-F.avg_pool2d(small,5,1,2,count_include_pad=False)
    correction = F.adaptive_avg_pool2d(restored-raw,size)
    confidence = (base.softmax(1)*base.log_softmax(1)).sum(1,keepdim=True)
    data = torch.cat(((base-base.mean(1,keepdim=True)).div(6).clamp(-2,2),2*small-1,confidence),1)
    return data,maps(raw_delta),maps(correction)


def adapter_input(proxies, arm):
    common,raw,correction = proxies
    if arm=="raw_adapter":
        return torch.cat((common,raw),1)
    if arm=="shifted_gate":
        displacement = tuple(s//2 for s in correction.shape[-2:])
        correction = correction.roll(displacement,(-2,-1))
    return torch.cat((common,correction),1)


class UtilityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Sequential(nn.Conv2d(29,24,3,padding=1),nn.SiLU(),
                                  nn.Conv2d(24,24,3,padding=1),nn.SiLU())
        self.project = nn.Conv2d(24,19,1)
        self.gate = nn.Conv2d(24,1,1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        features = self.core(x)
        logit = self.gate(features)
        correction = 2*self.project(features).tanh()*logit.sigmoid()
        return correction,logit


@torch.no_grad()
def cache_one(row, segmenter, mean, std, restorer):
    rgb = verified_image(row["rgb"],row["rgb_SHA256"],False)
    target = city_train_ids(verified_image(row["annotation"],row["mask_SHA256"],True))
    raw,_ = resize_image(rgb)
    restored = pgl_rgb(restorer,raw).clamp(0,1)
    before = vocabulary(segmenter,mean,std,raw)
    after = vocabulary(segmenter,mean,std,restored)
    truth = torch.from_numpy(target.astype(np.int64))[None,None].cuda()
    y = F.interpolate(truth.float(),before.shape[-2:],mode="nearest-exact").long()
    ids = y.clamp(0,18)
    utility = after.log_softmax(1).gather(1,ids)-before.log_softmax(1).gather(1,ids)
    common,raw_proxy,correction = proxy_maps(raw,restored,before)
    return dict(base=before, data=(common,raw_proxy,correction), target=y,utility=utility)


def strip(value, top, left, h,w, pad):
    return F.pad(value[...,top:top+h,left:left+w],pad,value=0)


def batches(data, seed, arm):
    rng = torch.Generator().manual_seed(seed+1243)
    for _ in range(STEPS):
        chosen = torch.randint(len(data),(BATCH,),generator=rng).tolist()
        output = []
        for i in chosen:
            source = data[i]
            height,width = source["base"].shape[-2:]
            h,w = min(WINDOW,height),min(WINDOW,width)
            top = int(torch.randint(height-h+1,(),generator=rng))
            left = int(torch.randint(width-w+1,(),generator=rng))
            crop = (top,left,h,w,(0,WINDOW-w,0,WINDOW-h))
            # Preserve void on padded or coarsely unlabelled pixels.
            target = strip((source["target"]!=255).float(),*crop)>0
            labels = strip(source["target"],*crop).masked_fill(~target,255)
            values = (source["input"][arm],source["base"],source["utility"])
            output.append((*[strip(x,*crop) for x in values],labels))
        yield [torch.cat(z) for z in zip(*output)]


def fit_network(data,arm,seed):
    set_seed(seed)
    model = UtilityAdapter().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    for x,base,gain,labels in batches(data,seed,arm):
        optimizer.zero_grad(set_to_none=True)
        delta,logit = model(x)
        valid = labels!=255
        n = valid.sum().clamp_min(1)
        loss = F.cross_entropy(base+delta,labels[:,0],ignore_index=255,reduction="sum")/n+.01*delta.square().mean()
        if arm in (MAIN,"shifted_gate"):
            target = (.5+gain/.8).clamp(0,1)
            importance = gain.abs().clamp(.15,2)*valid
            utility = F.binary_cross_entropy_with_logits(logit,target,reduction="none")
            loss = loss+.35*(utility*importance).sum()/importance.sum().clamp_min(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2,error_if_nonfinite=True)
        optimizer.step()
    print(dict(arm=arm,seed=seed,steps=STEPS,loss=loss.item()),flush=True)
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def load_adapter(weights):
    model = UtilityAdapter().cuda()
    model.load_state_dict(weights,strict=True)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def summed_scores(matrix):
    return {key:segmentation_metrics(torch.stack(values).sum(0).cpu()) for key,values in matrix.items()}


def train():
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    fit,val,split = verified_split()
    seg,mean,std,card,restorer,rgb = backbones()
    plan = dict(source_SHA256=digest(__file__),dataset_manifest=digest(DRIVING),zurich_manifest=digest(ZURICH),
                data=split,segmentation=card,rgb_weight=rgb,arms=list(ARMS),seeds=list(SEEDS),steps=STEPS,
                mechanism="A common low-resolution residual adapter consumes the semantic logits of raw RGB and aligned RGB corrections already computed by the other branch. The main gate is supervised by the actual improvement in coarse-label log likelihood when the frozen segmenter is fed the restored image. The second segmentation pass is only used to construct supervision.",
                controls="A raw-only matched adapter, an aligned correction adapter with the same capacity and no utility target, and a spatially shifted correction placebo with the same utility target. All arms have fixed RGB outputs, segmentation backbone, patch locations, update counts and inference sizes.",
                score="Select no checkpoints by testing. Three seeds use their final weights. Test all 40 Foggy Zurich v2 and 33 fine Foggy Driving images. Treat reuse of the earlier test cohorts as a diagnostic comparison, independent of the current fitting data.")
    if PROTOCOL.exists() and read(PROTOCOL)!=plan:
        raise ValueError("Parallel study protocol changed")
    if not PROTOCOL.exists():
        atomic_write(PROTOCOL,plan)
    if (ROOT/"fit.json").exists():
        return load()
    data = []
    for i,row in enumerate(fit):
        item = cache_one(row,seg,mean,std,restorer)
        item["input"] = {arm: adapter_input(item["data"],arm) for arm in ARMS}
        data.append(item)
        if (i+1)%15==0:
            print(dict(coarse_frames=i+1),flush=True)
    weights = {f"{arm}/{seed}":fit_network(data,arm,seed) for seed in SEEDS for arm in ARMS}
    torch_save(CHECKPOINT,weights)
    record = dict(weights_SHA256=digest(CHECKPOINT),protocol_SHA256=digest(PROTOCOL),fit=len(fit),calibrate=len(val),
                  fit_group_count=len({capture_group(r["id"]) for r in fit}),updated=STEPS*len(weights),
                  parameter_count=sum(x.numel() for x in weights[next(iter(weights))].values()),
                  finished_utc=now(),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))
    atomic_write(ROOT/"fit.json",record)
    print(record,flush=True)
    return record


def load():
    record = read(ROOT/"fit.json")
    if record["protocol_SHA256"]!=digest(PROTOCOL) or record["weights_SHA256"]!=digest(CHECKPOINT):
        raise ValueError("Frozen parallel checkpoint differs")
    states = torch.load(CHECKPOINT,map_location="cpu",weights_only=True)
    return {tuple(key.split("/")):load_adapter(v) for key,v in states.items()},record


def score_here(prediction, truth):
    return confusion_matrix(F.interpolate(prediction,truth.shape[-2:],mode="bilinear",align_corners=False).argmax(1),truth,19)


@torch.inference_mode()
def evaluate():
    configure_cuda()
    models,fit = load()
    seg,mean,std,card,restorer,_ = backbones()
    cases = list((d,scene,group,rgb,y) for d in ("foggy_zurich_testv2","foggy_driving_fine")
                 for scene,group,rgb,y in load_pairs(d))
    validation = set(read(PROTOCOL)["data"]["validation_ids"])
    for row in read(DRIVING)["rows"]:
        if row["id"] in validation:
            rgb = verified_image(row["rgb"],row["rgb_SHA256"],False)
            label = city_train_ids(verified_image(row["annotation"],row["mask_SHA256"],True))
            cases.append(("coarse_validation",row["id"],capture_group(row["id"]),rgb,label))
    all_records, matrix = [],defaultdict(list)
    for i,(dataset,scene,group,rgb,label) in enumerate(cases):
        raw,_ = resize_image(rgb)
        target = torch.from_numpy(label.astype(np.int64))[None].cuda()
        before = vocabulary(seg,mean,std,raw)
        restored = pgl_rgb(restorer,raw).clamp(0,1)
        proxies = proxy_maps(raw,restored,before)
        predictions = {("raw", "0"):before,("restored", "0"):vocabulary(seg,mean,std,restored)}
        predictions.update({(arm,seed):before+net(adapter_input(proxies,arm))[0] for (arm,seed),net in models.items()})
        expected = torch.bincount(target.flatten(),minlength=256)[:19].cpu()
        for (arm,seed),pred in predictions.items():
            score = score_here(pred,target)
            if not torch.equal(score.sum(1),expected):
                raise ValueError("A parallel classifier changed the evaluation label population")
            key = f"{dataset}/{arm}/{seed}"
            matrix[key].append(score)
            all_records.append(dict(dataset=dataset,scene=scene,group=group,method=arm,seed=seed,confusion=score.tolist()))
        if (i+1)%20==0:
            print(dict(semantic_test=i+1),flush=True)
    summary = summed_scores(matrix)
    means = defaultdict(list)
    for key,value in summary.items():
        means[key.rsplit("/",1)[0]].append(100*value["miou"])
    table = ROOT/"confusions.json"
    atomic_write(table,all_records)
    record = dict(fit_SHA256=fit["weights_SHA256"],scores_by_seed=summary,miou_percent={k:float(np.mean(v)) for k,v in means.items()},
                  rows=len(all_records),table_SHA256=digest(table),segmentation=card["model"],finished_utc=now())
    atomic_write(ROOT/"test.json",record)
    print(record["miou_percent"],flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","test"))
    if parser.parse_args().stage=="fit":
        train()
    else:
        evaluate()

"""Transfer the matched tonal correction to a frozen, stronger MCAF-Net anchor."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
from statistics import mean

import torch

from boundary_refinement.utils import set_seed
from boundary_refinement.models.restoration.complementary_rgb import (
    ColorGeometry, ColorHead, MODES, adjustment, field_moments, objective, projection, tonal_fields
)
from boundary_refinement.training.decision_overlap import MANIFEST, rr_records
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_complementary_rgb import HELD, NEW, additional_samples, fresh_pairs
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.exact_overlap import OFFSETS, neighbor_axis
from boundary_refinement.data.cohorts.fresh_real import CARD, pairs
from boundary_refinement.baselines.mcaf_attention import memory_efficient
from boundary_refinement.baselines.mcaf_reference import ROOT as ANCHOR, trained as mcaf
from boundary_refinement.training.overlap_quadratic import SEEDS, geometry, references
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT=REPO/"results/strong-anchor-color-v62"
STEPS=320
STRENGTHS=(0.,.25,.5,.75,1.)
ANCHOR_SEED=SEEDS[0]
EXPECTED=dict(rrshid=304,rrshid_extra=135,rrshid_fresh=180)


def split():
    cohorts=defaultdict(list)
    for row in read(MANIFEST)["pairs"]:
        if row["role"]=="val":
            cohorts[row["level"]].append(row["id"])
    fit, val = [], []
    for level, examples in sorted(cohorts.items()):
        if len(examples)!=24:
            raise ValueError("The registered RRSHID validation stratum changed")
        examples.sort(key=lambda key:sha256(f"v62/{key}".encode()).digest())
        fit.extend(examples[:16])
        val.extend(examples[16:])
    if len(set(fit))!=48 or len(set(val))!=24 or set(fit)&set(val):
        raise ValueError("The post-MCAF splits are incomplete")
    return dict(fit=fit, validation=val)


def registration():
    return dict(source_SHA256=digest(__file__), anchor_SHA256=digest(ANCHOR/"completed.json"),
                data_SHA256=digest(MANIFEST), prior_test_SHA256={p.name:digest(p) for p in (HELD,NEW,CARD)},
                cases=split(),steps=STEPS,modes=list(MODES),seeds=list(SEEDS),stages=list(STRENGTHS),
                design="Use frozen MCAF-Net seed 32003, trained only on the prior 240 images. "
                       "Fit the same global, local and projection heads on 48 different photograph pairs from "
                       "the previously reserved validation archive; use another 24 only to choose a scalar amplitude. "
                       "Evaluate all arms on the previously fixed RRSHID cohorts. A strong reference transfer check.")


@torch.no_grad()
def context(x,restorers,baseline):
    original=geometry(x,restorers)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        core=baseline(x)[0].float().clamp(0.,1.)
    fields=torch.cat((original.base-core,original.base+original.deltas[1:2]-core,x-core,tonal_fields(core)))
    product=field_moments(fields)
    p=original.partition
    matrix=p.average(product).permute(2,3,0,1)[None]
    t,old_rms,new_rms=projection(matrix)
    cue=torch.cat((original.features,2.*p.average(core)-1.,3.*new_rms.permute(0,3,1,2)),dim=1)
    return ColorGeometry(core,fields,p,cue,t,old_rms,new_rms,p.mass/p.mass.sum()),product


@torch.no_grad()
def training_pairs(recipe, restorer, baseline, smoke):
    examples={}
    for _,i,_,fog,clear in rr_records("val"):
        if i not in recipe["cases"]["fit"]:
            continue
        x,y=as_float(fog),as_float(clear)
        result,product=context(x,restorer,baseline)
        p=result.partition
        r={i:neighbor_axis(p.rows,i) for i in (-1,0,1)}
        c={i:neighbor_axis(p.cols,i) for i in (-1,0,1)}
        metric=torch.stack([((r[i].T@product@c[j])/p.mass).permute(2,3,0,1)[None] for i,j in OFFSETS])
        difference=y-result.anchor
        z=p.average((result.fields*difference).mean(1)).permute(1,2,0)[None]
        examples[i]=(result.features,result.transform,result.raw_rms,result.projected_rms,result.mass,metric,z,
                     difference.square().mean().clamp_min(.0025))
        if smoke and len(examples)==6:
            break
    if len(examples)!=(6 if smoke else 48):
        raise ValueError("The MCAF adapter trained on an incomplete image split")
    return list(examples.values())


def anchor():
    return memory_efficient(mcaf()[ANCHOR_SEED])


def fit(smoke=False):
    configure_cuda()
    recipe=registration()
    lock=ROOT/"protocol.json"
    if not smoke:
        if lock.exists() and read(lock)!=recipe:
            raise ValueError("Existing strong-anchor runs use a different input")
        if not lock.exists():
            atomic_write(lock,recipe)
    pool=training_pairs(recipe,references(),anchor(),smoke)
    rows=[]
    for seed in SEEDS[:1] if smoke else SEEDS:
        for mode in MODES:
            path=ROOT/f"{mode}-{seed}.pth"
            log=path.with_suffix(".json")
            if not smoke and log.exists():
                item=read(log)
                if item["weights_SHA256"]!=digest(path) or item["protocol_SHA256"]!=digest(lock):
                    raise ValueError("Strong-anchor previous adapter changed")
                rows.append(item)
                continue
            set_seed(seed)
            rng=torch.Generator().manual_seed(seed+62225)
            model=ColorHead().cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00007)
            for step in range(2 if smoke else STEPS):
                opt.zero_grad(set_to_none=True)
                items=torch.randint(len(pool),(3,),generator=rng).tolist()
                loss=torch.stack([objective(model,pool[i],mode) for i in items]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),2.5,error_if_nonfinite=True)
                opt.step()
                schedule.step()
            item=dict(seed=seed,mode=mode,steps=step+1,loss=float(loss.detach()))
            if not smoke:
                torch_save(path,{name:param.detach().cpu() for name,param in model.state_dict().items()})
                item.update(weights_SHA256=digest(path),protocol_SHA256=digest(lock))
                atomic_write(log,item)
                rows.append(item)
            print(item,flush=True)
    if not smoke:
        atomic_write(ROOT/"completed.json",dict(runs=rows,updates=sum(v["steps"] for v in rows),
            protocol_SHA256=digest(lock),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),finished_utc=now()))


def trained():
    note=ROOT/"protocol.json"
    ledger=read(ROOT/"completed.json")
    if not matches_registered(read(note), registration()) or ledger["protocol_SHA256"]!=digest(note):
        raise ValueError("The strong-anchor adapter provenance changed")
    head={}
    for r in ledger["runs"]:
        i=r["mode"],r["seed"]
        weight=ROOT/f"{i[0]}-{i[1]}.pth"
        if r["weights_SHA256"]!=digest(weight):
            raise ValueError("A strong-anchor weight lacks its expected digest")
        model=ColorHead().cuda()
        model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True),strict=True)
        head[i]=model.eval().requires_grad_(False)
    if set(head)!={(m,s) for m in MODES for s in SEEDS}:
        raise ValueError("Missing matched strong-anchor models")
    return head


@torch.inference_mode()
def calibrate(recipe, restorer, backbone, bank):
    values=defaultdict(list)
    verified=set()
    for d,i,_,x,y in rr_records("val"):
        if i not in recipe["cases"]["validation"]:
            continue
        q, _ = context(as_float(x),restorer,backbone)
        reference=as_float(y)
        verified.add(i)
        for (mode,_),model in bank.items():
            change=adjustment(q,model,mode)-q.anchor
            for s in STRENGTHS:
                values[mode,s].append(area_psnr((q.anchor+s*change).clamp(0.,1.),reference,None))
    if verified!=set(recipe["cases"]["validation"]):
        raise ValueError("The strong anchor amplitude validation used different images")
    probability={k:mean(v) for k,v in values.items()}
    choices={m:max(STRENGTHS,key=lambda s:(probability[m,s],-s)) for m in MODES}
    return dict(amplitude=choices, scores={"/".join(map(str,k)):v for k,v in probability.items()}, fitted_validation=sorted(verified))


def official():
    for row,fog,clear in pairs():
        if row["dataset"]=="rrshid":
            yield row["dataset"],row["id"],row["group"],fog,clear,None,row["level"]


@torch.inference_mode()
def evaluate():
    configure_cuda()
    rgb, m, h=references(),anchor(),trained()
    setup=dict(source_SHA256=digest(__file__),data=EXPECTED,frozen_SHA256=digest(ROOT/"completed.json"),
               recipe_SHA256=digest(ROOT/"protocol.json"),selection="24 preprocessing validation images only")
    lock=ROOT/"evaluation_protocol.json"
    if lock.exists() and read(lock)!=setup:
        raise ValueError("The MCAF follow-up protocol changed")
    if not lock.exists():
        atomic_write(lock,setup)
    scale=calibrate(registration(),rgb,m,h)
    atomic_write(ROOT/"calibration.json",scale)
    seen, rows, panels=Counter(),[],[]
    summary=defaultdict(list)
    by_group=defaultdict(lambda:defaultdict(list))
    for d,i,g,x,y,_,weather in chain(official(),additional_samples(read(HELD)),fresh_pairs(read(NEW))):
        fog,truth=as_float(x),as_float(y)
        q,_=context(fog,rgb,m)
        bank={("mcaf_seed0",0):q.anchor}
        for (mode,seed),head in h.items():
            step=adjustment(q,head,mode)-q.anchor
            bank[mode+"_full",seed]=(q.anchor+step).clamp(0.,1.)
            bank[mode,seed]=(q.anchor+scale["amplitude"][mode]*step).clamp(0.,1.)
        scores=defaultdict(list)
        for (method,seed),pic in bank.items():
            metric=area_psnr(pic,truth,None)
            scores[method].append(metric)
            rows.append(dict(dataset=d,id=i,group=g,condition=weather,method=method,seed=seed,psnr=metric))
        for mode,values in scores.items():
            summary[d,mode].append(mean(values))
            for ref in ("mcaf_seed0","global","local"):
                if mode!=ref:
                    by_group[d,mode,ref][g].append(mean(values)-mean(scores[ref]))
        if not seen[d]:
            panels.append((f"{d} / {i}",[to_rgb(p) for p in (fog,q.anchor,*[bank[k,SEEDS[0]] for k in MODES],truth)],
                ("Hazy input","MCAF anchor","Global adapter","Local adapter","Projection adapter","Clear reference")))
        seen[d]+=1
        if seen[d]==EXPECTED[d]:
            print(d,{m:round(mean(r),4) for (group,m),r in summary.items() if group==d},flush=True)
    if seen!=EXPECTED:
        raise ValueError("The three RRSHID evaluation cohorts changed")
    out, image=ROOT/"individual.json",ROOT/"predictions.png"
    atomic_write(out,rows)
    draw(image,panels)
    atomic_write(ROOT/"test.json",dict(counts=dict(seen),rows=len(rows),psnr={"/".join(k):mean(v) for k,v in summary.items()},
        intervals={"/".join(k):bootstrap({name:mean(v) for name,v in d.items()}) for k,d in by_group.items()},
        calibration=scale["amplitude"],table_SHA256=digest(out),picture_SHA256=digest(image),protocol_SHA256=digest(lock),
        finished_utc=now()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","fit","test"))
    code=p.parse_args().stage
    if code=="test":
        evaluate()
    else:
        fit(code=="smoke")

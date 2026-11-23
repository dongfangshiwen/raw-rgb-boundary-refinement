"""Isolate the intensity window with a same-capacity affine RGB control."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import chain
from statistics import mean

import torch

from dehaze_seg.utils import set_seed
from scripts.author_pgl_v25 import pgl_rgb
from scripts.complementary_rgb_v59 import (
    ROOT as WINDOWED, ColorHead, adjustment, code, context, field_moments, models as parent_models
)
from scripts.cross_scale_rgb_v44 import resize
from scripts.decision_overlap_v55 import MANIFEST, exact, rr_records
from scripts.edge_projection_v47 import draw
from scripts.evaluate_complementary_rgb_v59 import NEW, fresh_pairs
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_orthogonal_risk_v52 import reference_models
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_shared_haze_field_v40 import to_rgb
from scripts.exact_overlap_v54 import OFFSETS, neighbor_axis
from scripts.hrsd_test_v63 import MANIFEST as HRSD, pairs, stage as hrsd
from scripts.overlap_quadratic_v53 import SEEDS, references
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.region_restore_v48 import view_size
from scripts.relative_utility_v56 import fitted
from scripts.satehaze_test_v59 import MANIFEST as SATE, samples as sate_pairs, stage as sate
from scripts.single_anchor_student_v50 import area_psnr
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save
from scripts.veil_response_v45 import TRAIN_CARD, training_pairs


ROOT=REPO/"results/tonal-window-v63"
STEPS=480
TESTS=dict(rrshid_fresh=180,satehaze=135,hrsd_dhid=40,hrsd_lhid=40)


def unwindowed(anchor):
    luminosity=(anchor*anchor.new_tensor((.25,.5,.25))[None,:,None,None]).sum(1,keepdim=True)
    flat=torch.ones_like(luminosity)
    value=torch.cat((flat,2.*luminosity-1.,flat,flat),dim=0)
    rgb=anchor.new_tensor(((1.,1.,1.),(1.,1.,1.),(1.,-1.,0.),(-.5,-.5,1.)))
    return value*rgb[:,:,None,None]


def altered(item):
    fields=unwindowed(item.anchor)
    p=item.partition
    product=field_moments(fields)
    rms=p.average(product.diagonal(dim1=0,dim2=1).permute(2,0,1)).permute(1,2,0)[None].clamp_min(0.).sqrt()
    return replace(item,fields=torch.cat((item.fields[:3],fields)),raw_rms=rms),product


@torch.no_grad()
def data(restorers, prior, smoke):
    bank=defaultdict(list)
    counts=Counter()
    expected=dict(lm=128,rw=86,nh=36,rr=240)
    for d,_,_,x,y in chain(training_pairs(),rr_records("train")):
        if smoke and counts[d]==3:
            continue
        patch=torch.cat((as_float(x),as_float(y)))
        patch=resize(patch,view_size(patch,576),antialias=True)
        for item in (patch,) if d=="rr" else (patch,patch.pow(1.8)):
            image,truth=item.split(1)
            geom,_=context(image,restorers,prior)
            a,product=altered(geom)
            p=a.partition
            rows={i:neighbor_axis(p.rows,i) for i in (-1,0,1)}
            cols={i:neighbor_axis(p.cols,i) for i in (-1,0,1)}
            matrix=torch.stack([((rows[i].T@product@cols[j])/p.mass).permute(2,3,0,1)[None] for i,j in OFFSETS])
            residual=truth-a.anchor
            target=p.average((a.fields[3:]*residual).mean(1)).permute(1,2,0)[None]
            bank[d].append((a.features,a.transform,a.raw_rms,a.projected_rms,a.mass,matrix,target,
                           residual.square().mean().clamp_min(.0025)))
        counts[d]+=1
        if counts[d]==expected[d]:
            print(dict(cached=d,photos=counts[d]),flush=True)
    if counts!=(dict.fromkeys(expected,3) if smoke else expected):
        raise ValueError("The affine control does not use the matched V59 cohort")
    return tuple(bank.values())


def objective(net, item):
    feats,t,rms1,rms2,mass,metric,z,error=item
    direction,amplitude=code(net,feats,t,rms1,rms2,mass,"local")
    return (exact(metric,direction[...,3:],z)*mass).sum()/error+.35*amplitude.square().mean()


def plan():
    return dict(code_SHA256=digest(__file__), original_fit_SHA256=digest(WINDOWED/"completed.json"),
        fit_data={p.name:digest(p) for p in (TRAIN_CARD,MANIFEST)}, seeds=list(SEEDS),steps=STEPS,
        hypothesis="Compare nonlinear brightness window 4L(1-L) with unit-window affine RGB at equal feature input, "
                   "same initial heads, response limits, fit draws, loss and schedule. Record same-weight removal of "
                   "the brightness window as a second check. SateHaze has already been used for diagnostic scoring. "
                   "The HRSD test subsets are new optical synthetic haze and kept out of fits.")


def fit(smoke=False):
    configure_cuda()
    lock=ROOT/"protocol.json"
    design=plan()
    if not smoke:
        if lock.exists() and read(lock)!=design:
            raise ValueError("The nonlinear color ablation recipe changed")
        if not lock.exists():
            atomic_write(lock,design)
    groups=data(references(),fitted(),smoke)
    runs=[]
    for seed in SEEDS[:1] if smoke else SEEDS:
        path=ROOT/f"affine-{seed}.pth"
        note=path.with_suffix(".json")
        if not smoke and note.exists():
            row=read(note)
            if row["weights_SHA256"]!=digest(path) or row["protocol_SHA256"]!=digest(lock):
                raise ValueError("A fixed affine ablation run changed")
            runs.append(row)
            continue
        set_seed(seed)
        rng=torch.Generator().manual_seed(seed+59387)
        net=ColorHead().cuda()
        opt=torch.optim.AdamW(net.parameters(),lr=.0011,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00009)
        for step in range(2 if smoke else STEPS):
            opt.zero_grad(set_to_none=True)
            value=torch.stack([objective(net,q[int(torch.randint(len(q),(),generator=rng))]) for q in groups]).mean()
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.5,error_if_nonfinite=True)
            opt.step()
            schedule.step()
        row=dict(seed=seed,steps=step+1,loss=float(value.detach()))
        if not smoke:
            torch_save(path,{k:v.detach().cpu() for k,v in net.state_dict().items()})
            row.update(weights_SHA256=digest(path),protocol_SHA256=digest(lock))
            atomic_write(note,row)
            runs.append(row)
        print(row,flush=True)
    if not smoke:
        atomic_write(ROOT/"completed.json",dict(runs=runs,updates=sum(r["steps"] for r in runs),protocol_SHA256=digest(lock),
            torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),finished_utc=now()))


def trained():
    lock=ROOT/"protocol.json"
    summary=read(ROOT/"completed.json")
    if read(lock)!=plan() or digest(lock)!=summary["protocol_SHA256"]:
        raise ValueError("Affine fitting source no longer matches the checkpoint")
    found={}
    for r in summary["runs"]:
        filename=ROOT/f"affine-{r['seed']}.pth"
        if r["weights_SHA256"]!=digest(filename):
            raise ValueError("An affine checkpoint changed")
        net=ColorHead().cuda()
        net.load_state_dict(torch.load(filename,map_location="cpu",weights_only=True),strict=True)
        found[r["seed"]]=net.eval().requires_grad_(False)
    if set(found)!=set(SEEDS):
        raise ValueError("The same-capacity affine control seed set is incomplete")
    return found


@torch.inference_mode()
def test():
    configure_cuda()
    public, more = sate(), hrsd()
    fit, before, src, restorers=trained(),parent_models(),fitted(),references()
    pgl,rs,author=reference_models()
    design=dict(code_SHA256=digest(__file__),cohorts={p.name:digest(p) for p in (NEW,SATE,HRSD)},
        source_SHA256=digest(ROOT/"completed.json"),counts=TESTS,
        diagnostic_roi="Dark and bright pixels of the known V56 anchor: luma <=0.1 or >=0.9. "
                       "No ground truth is used to define the ROI.")
    lock=ROOT/"evaluation_protocol.json"
    if lock.exists() and read(lock)!=design:
        raise ValueError("The native-grid window evaluation changed")
    if not lock.exists():
        atomic_write(lock,design)
    count, rows, panels=Counter(),[],[]
    stats=defaultdict(list)
    groups=defaultdict(lambda:defaultdict(list))
    for d,i,g,x,y,_,condition in chain(fresh_pairs(read(NEW)),sate_pairs(public),pairs(more)):
        frame,truth=as_float(x),as_float(y)
        old,mct=context(frame,restorers,src)
        a,_=altered(old)
        l=(old.anchor*old.anchor.new_tensor((.25,.5,.25))[None,:,None,None]).sum(1)[0]
        region=(l<=.1)|(l>=.9)
        extremes=int(region.sum())
        available=extremes>=256
        outputs={("mct",0):mct,("base",0):old.anchor,("pgl_author",0):pgl_rgb(pgl,frame).clamp(0.,1.),
                 ("dehazeformer_rs",0):author_rgb(rs,frame/(frame.amax()+1e-8)).clamp(0.,1.)}
        for seed in SEEDS:
            outputs["windowed",seed]=adjustment(old,before["local",seed],"local")
            outputs["windowed_global",seed]=adjustment(old,before["global",seed],"global")
            outputs["affine",seed]=adjustment(a,fit[seed],"local")
            outputs["frozen_remove_window",seed]=adjustment(a,before["local",seed],"local")
        summary=defaultdict(list)
        for (m,seed),raw_rgb in outputs.items():
            output=raw_rgb.clamp(0.,1.)
            psnr=area_psnr(output,truth,None)
            ssim=float(gaussian_ssim(output,truth)[0]) if seed in (0,SEEDS[0]) else None
            roi=area_psnr(output,truth,region) if available else None
            outside=float(((raw_rgb<0)|(raw_rgb>1)).float().mean())
            r=dict(dataset=d,id=i,group=g,condition=condition,method=m,seed=seed,psnr=psnr,
                   ssim=ssim,extreme_psnr=roi,extreme_pixels=extremes,clipped_channel_fraction=outside)
            rows.append(r)
            stats[d,m].append(r)
            summary[m].append(psnr)
        for control in ("affine","windowed_global","frozen_remove_window","base","pgl_author","dehazeformer_rs","mct"):
            groups[d,control][g].append(mean(summary["windowed"])-mean(summary[control]))
        if not count[d]:
            choices=(frame,old.anchor,mct,outputs["affine",SEEDS[0]],outputs["windowed",SEEDS[0]],truth)
            panels.append((f"{d} / {i}",[to_rgb(p.clamp(0.,1.)) for p in choices],
                ("Hazy photo","V56","Published MCT","Local affine","Local brightness window","Clear reference")))
        count[d]+=1
        if count[d]==TESTS[d]:
            print(d,{m:round(mean(r["psnr"] for r in rows),4) for (source,m),rows in stats.items() if source==d},flush=True)
    if count!=TESTS:
        raise ValueError("HRSD or SateHaze image lists changed")
    table,img=ROOT/"individual.json",ROOT/"predictions.png"
    atomic_write(table,rows)
    draw(img,panels)
    atomic_write(ROOT/"test.json",dict(counts=dict(count),rows=len(rows),author=author,
        means={"/".join(k):{metric:mean(r[metric] for r in v if r[metric] is not None)
            for metric in ("psnr","ssim","extreme_psnr","clipped_channel_fraction") if any(r[metric] is not None for r in v)}
            for k,v in stats.items()},
        paired={"/".join(k):bootstrap({group:mean(values) for group,values in v.items()}) for k,v in groups.items()},
        protocol_SHA256=digest(lock),table_SHA256=digest(table),panel_SHA256=digest(img),finished_utc=now()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","smoke","test"))
    cmd=parser.parse_args().stage
    if cmd=="test":
        test()
    else:
        fit(cmd=="smoke")

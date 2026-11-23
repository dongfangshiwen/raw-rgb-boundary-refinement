"""Compare weight estimators on the same frames selected by the frozen V29 gate."""
import argparse
from collections import defaultdict
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from dehaze_seg.utils import write_csv
from scripts.conditional_fusion_v29 import CALL_SHARE, LOSS_DB, MAIN as V29_MAIN, actions, cap_alpha, features
from scripts.conditional_fusion_v29 import paired_geometry
from scripts.evaluate_conditional_fusion_v29 import evaluation_pairs
from scripts.evaluate_curve_transfer_v25 import as_float, as_rgb
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.expert_fusion_v26 import experts
from scripts.post_fusion_v30 import MAIN, MODES, load, scores_after_call, targets, window_geometry, windows
from scripts.projective_ledger_v12 import digest, now, read
from scripts.stage_postfusion_holdout_v30 import ROOT, RW_CARD, camera_code
from scripts.stage_revide_v30 import CARD as REVIDE
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


OLD = ("development","rwhaze","ntire24_complement")
FRESH = ("rw_reserve","revide")
CONTROLS = ("selected_hard","selected_half","v29_shrink","v29_cap")
METHODS = (*CONTROLS,*MODES)
V29 = ROOT.parent/"conditional-fusion-v29"


def samples(cohort):
    if cohort=="prior":
        for label in OLD:
            yield from evaluation_pairs(label)
        return
    card = read(REVIDE if cohort=="revide" else RW_CARD)
    if digest(card["file"])!=card["SHA256"]:
        raise ValueError("Fresh paired evaluation file changed")
    for row in torch.load(card["file"],map_location="cpu",weights_only=True):
        group = row["group"] if cohort=="revide" else camera_code(row["id"])
        yield cohort,row["id"],group,row["hazy"],row["clean"],None


def metrics(pred,ref,roi,include_ssim):
    error = (pred-ref).square()
    mse = error.mean() if roi is None else (error*roi).sum()/(3*roi.sum())
    result = dict(psnr=(-10*mse.clamp_min(1e-12).log10()).item(),ssim=None)
    if include_ssim:
        result["ssim"] = gaussian_ssim(pred,ref)[0].item()
    return result


def compare_methods(x,y,roi,forward,previous,state,shrink,include_ssim):
    a,b = (f(x).clamp(0,1) for f in forward)
    score,amount,lower = actions(previous,features(x,a))[V29_MAIN]
    keep = score.item()>previous["thresholds"][V29_MAIN] and amount.item()>0
    energy = (b-a).square().mean().sqrt()
    basis = windows(x)
    alpha,detail = scores_after_call(x,a,b,state,basis)
    alpha.update(selected_hard=torch.ones_like(amount),selected_half=torch.full_like(amount,.5),
                 v29_shrink=shrink*amount,v29_cap=cap_alpha(amount,lower,energy))
    base = metrics(a,y,roi,include_ssim)
    values = {method:dict(**metrics(torch.lerp(a,b,coefficient),y,roi,include_ssim),
                          mean_alpha=coefficient.mean().item()) for method,coefficient in alpha.items()}
    global_terms = paired_geometry(a,b,y,roi)
    true_step = targets(global_terms[:3],energy)[1]
    local_terms,dist,area = window_geometry(a,b,y,roi,basis)
    true_local = targets(local_terms,dist)[:,1]
    local_lower = detail["tile_lower_steps"]
    protected = {"post_lower":bool((detail["lower_step"]<=0)|(detail["lower_step"]<=true_step)),
                 "window_lower":bool(((local_lower<=0)|(local_lower<=true_local)|(area<=0)).all())}
    for name,ok in protected.items():
        if ok and values[name]["psnr"]<base["psnr"]-.001:
            raise ValueError("A covered projected step increased paired MSE")
    oracle = (global_terms[2]/global_terms[1].clamp_min(1e-12)).clamp(0,1)
    external = dict(mct=base,pgl=metrics(b,y,roi,include_ssim),
                    equal_all=metrics(.5*(a+b),y,roi,include_ssim),
                    scalar_oracle=metrics(torch.lerp(a,b,oracle),y,roi,include_ssim))
    predictions = dict(values=values,base=base,external=external,score=score.item(),called=keep,covered=protected)
    return predictions


def row(record,regime,method,call,score,alpha):
    return dict(dataset=record["dataset"],scene=record["scene"],group=record["group"],regime=regime,
                method=method,pgl_call=int(call),mean_alpha=alpha,**score,
                gain_vs_mct=score["psnr"]-record["base"]["psnr"])


def select_rows(records):
    cases = sorted(records,key=lambda t:(-t["score"],t["scene"]))
    count = round(CALL_SHARE*len(cases))
    output = []
    for i,v in enumerate(cases):
        for regime,taken in (("online",v["called"]),("matched",i<count)):
            for method,value in v["values"].items():
                if taken:
                    score = {k:value[k] for k in ("psnr","ssim")}
                else:
                    score = v["base"]
                output.append(row(v,regime,method,taken,score,value["mean_alpha"] if taken else 0.))
        output.extend(row(v,"all",method,method!="mct",score,0.) for method,score in v["external"].items())
    return output,cases[:count]


def summarize(rows):
    groups = defaultdict(list)
    for local in rows:
        groups[local["dataset"],local["regime"],local["method"]].append(local)
    result = {}
    for (dataset,regime,method),items in groups.items():
        n = len(items)
        gains = np.array([r["gain_vs_mct"] for r in items])
        active = [r for r in items if r["pgl_call"]]
        ssim = [r["ssim"] for r in items if r["ssim"] is not None]
        result[f"{dataset}/{regime}/{method}"] = dict(n=n,psnr=float(np.mean([r["psnr"] for r in items])),
                   average_gain=float(gains.mean()),ssim=float(np.mean(ssim)) if ssim else None,calls=len(active),
                   changed=sum(r["mean_alpha"]>1e-6 for r in active) if regime!="all" else None,
                   severe=int((gains<-LOSS_DB).sum()),worst=float(gains.min()))
    return result


def paired_intervals(rows):
    paired = defaultdict(dict)
    for r in rows:
        if r["regime"]!="all":
            paired[r["dataset"],r["regime"],r["group"],r["scene"]][r["method"]] = r["psnr"]
    differences = defaultdict(lambda: defaultdict(list))
    for (dataset,regime,group,_),scores in paired.items():
        for m in MODES:
            differences[dataset,regime,m][group].append(scores[m]-scores["v29_shrink"])
    answer = {}
    for (dataset,regime,method),by_group in differences.items():
        values = np.array([np.mean(v) for v in by_group.values()])
        rng = np.random.default_rng(30301)
        draws = values[rng.integers(0,len(values),size=(12000,len(values)))].mean(1)
        answer[f"{dataset}/{regime}/{method}"] = dict(groups=len(values),mean=float(values.mean()),
                         interval95=np.percentile(draws,[2.5,97.5]).tolist(),control="v29_shrink")
    return answer


def collect_demos(chosen,forward,state,previous,shrink):
    cases, timings = [], defaultdict(list)
    for i,row in enumerate(chosen):
        x = as_float(row["hazy"])
        a,b = (fn(x).clamp(0,1) for fn in forward)
        _,amount,_ = actions(previous,features(x,a))[V29_MAIN]
        if i in ({0,len(chosen)-1} if len(chosen)>1 else {0}):
            each,_ = scores_after_call(x,a,b,state,spatial=False)
            alpha = each[MAIN].item()
            title = f"{row['dataset']} {row['scene']} | alpha {alpha:.2f}"
            cases.append((title,[as_rgb(t) for t in (x,a,b,torch.lerp(a,b,amount*shrink),
                                                      torch.lerp(a,b,alpha),as_float(row["clear"]))]))
        operations = {
            "fixed_shrink":lambda: torch.lerp(a,b,amount*shrink),
            "post_step":lambda: torch.lerp(a,b,scores_after_call(x,a,b,state,spatial=False)[0]["post_step"]),
            "window_step":lambda: torch.lerp(a,b,scores_after_call(x,a,b,state)[0]["window_step"]),
        }
        for name,function in operations.items():
            function()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            function()
            torch.cuda.synchronize()
            timings[name].append(1000*(time.perf_counter()-begin))
    return cases,{k:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for k,v in timings.items()}


def render_cases(file,cases):
    captions = ("Hazy input","MCT","PGL-Net-T","V29 fixed shrink","Post-output step","Clear")
    canvas = Image.new("RGB",(1860,242*len(cases)),"#eff4f6")
    draw = ImageDraw.Draw(canvas)
    for i,(title,images) in enumerate(cases):
        draw.text((6,6+i*242),title,fill="#132e3b")
        for col,(name,image) in enumerate(zip(captions,images)):
            x,y = col*310,i*242
            image.thumbnail((304,200),Image.Resampling.LANCZOS)
            canvas.paste(image,(x+(310-image.width)//2,y+35))
            draw.text((x+6,y+20),name,fill="#132e3b")
    canvas.save(file)


@torch.inference_mode()
def evaluate(cohort):
    configure_cuda()
    previous,state,receipt = load()
    forward,authors = experts()
    control = read(V29/"audit.json")
    if control["fit_SHA256"]!=receipt["prior_gate_SHA256"]:
        raise ValueError("The prior proportional-shrink control changed")
    shrink = control["fitted_controls"]["fixed_shrink"]
    data = defaultdict(list)
    for dataset,scene,group,hazy,clear,roi in samples(cohort):
        valid = roi[None,None].cuda() if roi is not None else None
        data[dataset].append(dict(dataset=dataset,scene=scene,group=group,hazy=hazy,clear=clear,
                           **compare_methods(as_float(hazy),as_float(clear),valid,forward,previous,state,shrink,cohort in FRESH)))
        n = sum(map(len,data.values()))
        if n%40==0:
            print(dict(cohort=cohort,paired=n),flush=True)
    rows,selected,coverage = [],[],{}
    for dataset,records in data.items():
        batch,chosen = select_rows(records)
        selected+=chosen
        rows+=batch
        coverage[dataset] = {name:sum(r["covered"][name] for r in chosen) for name in ("post_lower","window_lower")}
    if not rows:
        raise ValueError("Post-fusion cohort contains no image pairs")
    out = ROOT/(cohort+"_metrics.csv")
    write_csv(out,rows)
    out_image = ROOT/(cohort+"_comparison.png")
    timing = {}
    image_sha = None
    if cohort in FRESH:
        cases,timing = collect_demos(selected,forward,state,previous,shrink)
        render_cases(out_image,cases)
        image_sha = digest(out_image)
    stats = summarize(rows)
    frozen = dict(cohort=cohort,count=sum(map(len,data.values())),rows=len(rows),summary=stats,coverage=coverage,
                  group_intervals=paired_intervals(rows),after_call_timing=timing,
                  image_SHA256=image_sha,table_SHA256=digest(out),weights_SHA256=receipt["weights_SHA256"],
                  gate_SHA256=receipt["prior_gate_SHA256"],weights_frozen_at=receipt["finished_utc"],
                  fixed_shrink=shrink,evaluator_SHA256=digest(__file__),authors=authors,finished_utc=now())
    atomic_write(ROOT/("evaluation_"+cohort+".json"),frozen)
    for key,entry in stats.items():
        if "/matched/" in key or key.endswith("/online/"+MAIN):
            print(key,{p:round(entry[p],4) for p in ("psnr","calls","changed","severe","worst")},flush=True)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=(*FRESH,"prior"))
    args = parser.parse_args()
    evaluate(args.cohort)


if __name__=="__main__":
    main()

"""Compare calibrated routing at its deployed and exactly matched PGL call rates."""
import argparse
from collections import defaultdict
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
import torch

from boundary_refinement.utils import write_csv
from boundary_refinement.training.calibrate_expert_choice import switches
from boundary_refinement.training.conditional_fusion import CALL_SHARE, LOSS_DB, MAIN, METHODS, actions, bedde_split, cap_alpha
from boundary_refinement.training.conditional_fusion import city_manifest, city_pairs, features, load_state, paired_geometry, relative_psnr, restore
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float, as_rgb
from boundary_refinement.evaluation.photographic.evaluate_preview_routing import development
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.inference.infer_guarded_expert import load_router
from boundary_refinement.data.staging.prepare_rw2ah import decode
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, long_edge
from boundary_refinement.data.staging.stage_fusion_holdouts import ROOT
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


CONTROLS = ("mct","pgl","equal_blend","v27_guarded","blend_oracle")


def evaluation_pairs(cohort):
    if cohort=="development":
        yield from development()
        cities = city_manifest()
        split = bedde_split(cities)
        for _,scene,city,hazy,clean,roi in city_pairs({k:v for k,v in cities.items() if split[k]=="dev"}):
            yield "bedde_dev",scene,city,hazy,clean,roi
        return
    if cohort=="ntire24_complement":
        record = read(ROOT/"ntire24_complement_manifest.json")
        if digest(record["file"])!=record["SHA256"]:
            raise ValueError("The unseen NTIRE24 holdout pixels changed")
        for row in torch.load(record["file"],map_location="cpu",weights_only=True):
            yield cohort,row["id"],row["id"],row["hazy"],row["clean"],None
        return
    record = read(ROOT/"rwhaze_manifest.json")
    if digest(record["file"])!=record["SHA256"]:
        raise ValueError("The RW-Haze author archive changed")
    with ZipFile(record["file"]) as source:
        for group,scene in sorted(record["groups"].items(),key=lambda x:int(x[0])):
            clear = decode(source.read(scene["clear"]["member"]))
            side = min(1024,max(clear.size))
            dest = image_tensor(long_edge(clear,side))
            for frame,image in sorted(scene.items()):
                if frame!="clear":
                    haze = image_tensor(long_edge(decode(source.read(image["member"])),side))
                    yield "rwhaze",group+"_"+frame,group,haze,dest,None


def measured(stats,alpha):
    value = torch.tensor(alpha,device=stats.device,dtype=stats.dtype)
    base_psnr = (-10*stats[0].log10()).item()
    change = relative_psnr(stats,value).item()
    return base_psnr+change,change


def row(record,method,choice,alpha,stage):
    psnr,gain = record["values"][method]
    if not choice and alpha==0:
        gain = 0.
        psnr = record["values"]["mct"][0]
    return dict(dataset=record["dataset"],scene=record["scene"],group=record["group"],regime=stage,method=method,
                pgl_call=int(choice),alpha=alpha,psnr=psnr,gain_vs_mct=gain,
                prior_score=record["score"].get(method),prior_alpha=record["prior_alpha"].get(method))


def summarize(rows):
    parts = defaultdict(list)
    for r in rows:
        parts[r["dataset"],r["regime"],r["method"]].append(r)
    answer = {}
    for (dataset,regime,method),data in parts.items():
        gain = np.array([v["gain_vs_mct"] for v in data])
        calls = sum(v["pgl_call"] for v in data)
        answer[f"{dataset}/{regime}/{method}"] = dict(n=len(data),calls=calls,mean_psnr=float(np.mean([v["psnr"] for v in data])),
                    mean_gain=float(gain.mean()),wins=int((gain>1e-6).sum()),losses=int((gain<-1e-6).sum()),
                    severe=int((gain<-LOSS_DB).sum()),worst=float(gain.min()),
                    mean_alpha_if_called=float(np.mean([v["alpha"] for v in data if v["pgl_call"]])) if calls else 0.)
    return answer


def match_calls(records):
    domains = defaultdict(list)
    for scene in records:
        domains[scene["dataset"]].append(scene)
    result = []
    for group in domains.values():
        n = round(CALL_SHARE*len(group))
        for method in METHODS:
            ordered = sorted(group,key=lambda v:(-v["score"][method],v["scene"]))
            result.extend(row(v,method,i<n,v["alpha"][method] if i<n else 0.,"matched") for i,v in enumerate(ordered))
    return result


def paired_intervals(rows):
    cache = defaultdict(dict)
    for r in rows:
        cache[r["dataset"],r["regime"],r["group"],r["scene"]][r["method"]] = r["psnr"]
    cohort = defaultdict(list)
    for (dataset,regime,group,_),values in cache.items():
        for ref in ("mct","raw_hard","geometry_half","geometry_open","anchor_blend"):
            if ref in values:
                cohort[dataset,regime,ref].append((group,values[MAIN]-values[ref]))
    results = {}
    for key,comparisons in cohort.items():
        totals = defaultdict(list)
        for group,delta in comparisons:
            totals[group].append(delta)
        points = np.array([np.mean(vals) for vals in totals.values()])
        rng = np.random.default_rng(29026)
        resampled = points[rng.integers(0,len(points),(12000,len(points)))].mean(1)
        results["/".join(key)] = dict(groups=len(points),paired_group_mean=float(points.mean()),
                                     interval95=np.percentile(resampled,[2.5,97.5]).tolist())
    return results


def render_cases(file,cases):
    canvas = Image.new("RGB",(1700,265*len(cases)),"#edf3f6")
    labels = ("Hazy input","MCT","PGL-Net-T","Conditional blend","Clear")
    draw = ImageDraw.Draw(canvas)
    for i,(title,images) in enumerate(cases):
        for j,(name,image) in enumerate(zip(labels,images)):
            image.thumbnail((334,229),Image.Resampling.LANCZOS)
            x,y = j*340,i*265
            canvas.paste(image,(x+(340-image.width)//2,y+29))
            draw.text((x+6,y+5),title+" | "+name,fill="#102839")
    canvas.save(file)


def call_sample(image,forward,state,variant):
    counts = [0,0]

    def traced(operation,x,i):
        counts[i]+=1
        return operation(x)

    operations = tuple(lambda x,i=i,op=op: traced(op,x,i) for i,op in enumerate(forward))
    output,action = restore(image,operations,state,variant)
    expected = [1,int(action["pgl"])] if variant!="raw_hard" else [int(not action["pgl"]),int(action["pgl"])]
    if counts!=expected:
        raise ValueError("Conditional implementation invoked the wrong specialists")
    return output,action,counts


def timed(inference,images):
    for image in images[:2]:
        inference(as_float(image))
    times = []
    for image in images:
        x = as_float(image)
        torch.cuda.synchronize()
        start = time.perf_counter()
        inference(x)
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    return dict(n=len(times),median_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)))


@torch.inference_mode()
def evaluate(cohort):
    configure_cuda()
    state,registered = load_state()
    forward,authors = experts()
    prior,base_record = load_router()
    rows,records,images,cases = [],[],[],[]
    program_data = defaultdict(lambda:dict(calls=0,theoretical_condition=0,cap_active=0,severe=0))
    worst = None
    for dataset,scene,group,hazy,clear,roi in evaluation_pairs(cohort):
        x,y = as_float(hazy),as_float(clear)
        a,b = (fn(x).clamp(0,1) for fn in forward)
        mask = roi[None,None].cuda() if roi is not None else None
        terms = paired_geometry(a,b,y,mask)
        other_energy,full_energy = terms[1],terms[3]
        actual_effective = terms[0].sqrt()*full_energy.clamp_min(1e-12).sqrt()/other_energy.clamp_min(1e-12).sqrt()
        proposed = actions(state,features(x,a))
        record = dict(dataset=dataset,scene=scene,group=group,score={},alpha={},prior_alpha={},values={})
        for method,(score,alpha,lower) in proposed.items():
            record["score"][method] = score.item()
            record["prior_alpha"][method] = alpha.item()
            preset = alpha
            if METHODS[method][1]=="geometry_capped":
                alpha = cap_alpha(alpha,lower,full_energy.sqrt())
            used = score.item()>state["thresholds"][method] and alpha.item()>0
            record["alpha"][method] = alpha.item()
            record["values"][method] = measured(terms,alpha.item())
            if used and METHODS[method][1]=="geometry_capped":
                meter = program_data[dataset+"/"+method]
                validity = lower.item()<=actual_effective.item()
                severe = record["values"][method][1]<-LOSS_DB-.001
                if validity and severe:
                    raise ValueError("The claimed loss cap does not hold for the recorded covered image")
                meter["calls"]+=1
                meter["theoretical_condition"]+=validity
                meter["cap_active"]+=preset.item()>alpha.item()
                meter["severe"]+=severe
        choices = dict(mct=(False,0.),pgl=(True,1.),equal_blend=(True,.5),
                       v27_guarded=(True,float(switches(prior,x,a,b)["guarded"])),
                       blend_oracle=(True,(terms[2]/terms[1].clamp_min(1e-12)).clamp(0,1).item()))
        for method,(extra,alpha) in choices.items():
            record["values"][method] = measured(terms,alpha)
            rows.append(row(record,method,extra,alpha,"calibrated"))
        for method in METHODS:
            alpha = record["alpha"][method]
            called = record["score"][method]>state["thresholds"][method] and alpha>0
            rows.append(row(record,method,called,alpha if called else 0.,"calibrated"))
        records.append(record)
        alpha = rows[-1]["alpha"]
        gain = rows[-1]["gain_vs_mct"]
        if not cases or (cohort=="development" and (dataset,scene)==("ohaze","005")):
            cases.append((f"{dataset} {scene}",[as_rgb(v) for v in (x,a,b,torch.lerp(a,b,alpha),y)]))
        if worst is None or gain<worst[0]:
            worst = gain,(f"{dataset} {scene} ({gain:+.2f} dB)",[as_rgb(v) for v in (x,a,b,torch.lerp(a,b,alpha),y)])
        if cohort!="development":
            images.append(hazy)
        if len(records)%40==0:
            print(dict(cohort=cohort,processed=len(records)),flush=True)
    if not records:
        raise ValueError("No evaluation dataset available for " + cohort)
    if worst[0]<0:
        cases.append(worst[1])
    rows.extend(match_calls(records))
    csvfile = ROOT/(cohort+"_metrics.csv")
    write_csv(csvfile,rows)
    figure = ROOT/(cohort+"_comparison.png")
    render_cases(figure,cases)
    device_cost = {}
    if images:
        selected = [images[k] for k in np.linspace(0,len(images)-1,min(24,len(images)),dtype=int)]
        for name,implementation in dict(mct=forward[0],pgl=forward[1],
                       equal_blend=lambda z:.5*(forward[0](z)+forward[1](z)),
                       raw_hard=lambda z:restore(z,forward,state,"raw_hard"),
                       geometry_capped=lambda z:restore(z,forward,state)).items():
            device_cost[name] = timed(implementation,selected)
        for image in selected[::4]:
            call_sample(as_float(image),forward,state,MAIN)
    results = summarize(rows)
    receipt = dict(cohort=cohort,count=len(records),rows=len(rows),scores=results,group_intervals=paired_intervals(rows),
                   cap_coverage=program_data,timing=device_cost,table_SHA256=digest(csvfile),image_SHA256=digest(figure),
                   trained_SHA256=registered["weights_SHA256"],v27_SHA256=base_record["calibration_SHA256"],
                   evaluated_after_freeze=registered["finished_utc"],frozen_authors=authors,
                   evaluator_SHA256=digest(__file__),finished_utc=now())
    atomic_write(ROOT/(cohort+".json"),receipt)
    for key,value in results.items():
        if key.endswith(("/"+MAIN,"/mct","/equal_blend","/geometry_open","/raw_hard","/geometry_half")):
            print(key,{p:round(value[p],4) for p in ("mean_psnr","mean_gain","calls","severe","worst")},flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("development","rwhaze","ntire24_complement"))
    args = parser.parse_args()
    evaluate(args.cohort)


if __name__=="__main__":
    main()

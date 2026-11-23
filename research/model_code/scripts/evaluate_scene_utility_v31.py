"""Evaluate the frozen routing heads with identical fusion and call counts."""
import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw
import torch

from dehaze_seg.utils import write_csv
from scripts.conditional_fusion_v29 import CALL_SHARE, MAIN as PREVIOUS, actions, features, load_state
from scripts.evaluate_curve_transfer_v25 import as_float, as_rgb
from scripts.evaluate_post_fusion_v30 import FRESH, samples as prior_samples
from scripts.evaluate_real_transfer_v22 import gaussian_ssim
from scripts.expert_fusion_v26 import experts
from scripts.projective_ledger_v12 import digest, now
from scripts.scene_utility_v31 import ALPHA, ARMS, CROSS, MAIN, POINTWISE, load, predict, restore
from scripts.stage_lmhaze_v31 import CARD, ROOT, samples
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


CONTROLS = ("v29", "hash_order", "utility_oracle")
RANKERS = (*ARMS, *CONTROLS)
CURVE = (0, .1, .2, .3, .5, .75, 1.)


def input_pairs(cohort):
    if cohort == "lmhaze":
        for dataset, scene, group, hazy, clear, roi, _ in samples({"test"}):
            yield dataset, scene, group, hazy, clear, roi
    else:
        for name in (*FRESH,"prior"):
            yield from prior_samples(name)


def evaluate_rgb(output, clear, roi, use_ssim):
    error = (output-clear).square()
    mse = error.mean() if roi is None else (error*roi).sum()/(3*roi.sum())
    return dict(psnr=(-10*mse.clamp_min(1e-12).log10()).item(),
                ssim=gaussian_ssim(output,clear)[0].item() if use_ssim else None)


def order(records, key):
    return sorted(records, key=lambda row: (-row["scores"][key],row["scene"]))


def integrate(row, method, regime, selected, output):
    return dict(dataset=row["dataset"],scene=row["scene"],group=row["group"],method=method,regime=regime,
                pgl_call=int(selected), alpha=ALPHA if selected and output=="blend" else float(output=="pgl"),
                rank_score=row["scores"].get(method), gain=row[output]["psnr"]-row["mct"]["psnr"], **row[output])


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"],row["regime"],row["method"]].append(row)
    out = {}
    for (dataset,regime,method), data in groups.items():
        gains = np.array([v["gain"] for v in data])
        ssim = [v["ssim"] for v in data if v["ssim"] is not None]
        out[f"{dataset}/{regime}/{method}"] = dict(n=len(data), calls=sum(v["pgl_call"] for v in data),
                    psnr=float(np.mean([v["psnr"] for v in data])), gain=float(gains.mean()),
                    ssim=float(np.mean(ssim)) if ssim else None, severe=int((gains < -3).sum()),
                    lost=int((gains < -1e-6).sum()), won=int((gains > 1e-6).sum()), worst=float(gains.min()))
    return out


def bootstrap(groups):
    values = np.array(list(groups.values()), dtype=np.float64)
    rng = np.random.default_rng(31041)
    sampled = values[rng.integers(len(values), size=(12000,len(values)))].mean(1)
    return dict(group_count=len(values), equal_group_mean=float(values.mean()),
                ci95=np.percentile(sampled,[2.5,97.5]).tolist())


def paired_comparisons(rows):
    images = defaultdict(dict)
    for row in rows:
        if row["regime"] == "matched":
            images[row["dataset"],row["group"],row["scene"]][row["method"]] = row["psnr"]
    grouped = defaultdict(lambda: defaultdict(list))
    for (dataset,group,_), scores in images.items():
        for control in ("v29","prior_ridge","joint_ridge",POINTWISE,CROSS):
            grouped[dataset,control][group].append(scores[MAIN]-scores[control])
    return {f"{d}/{m}":bootstrap({group:np.mean(values) for group,values in items.items()})
            for (d,m),items in grouped.items()}


def concordance(rows):
    scenes = defaultdict(list)
    for r in rows:
        scenes[r["group"]].append(r)
    result = defaultdict(dict)
    counts = {}
    for name, items in scenes.items():
        accepted = []
        for a,b in combinations(items,2):
            delta = a["scores"]["utility_oracle"] - b["scores"]["utility_oracle"]
            if abs(delta) > .15:
                accepted.append((a,b,delta))
        counts[name] = len(accepted)
        if not accepted:
            continue
        for method in (*ARMS,"v29","hash_order"):
            product = np.array([(a["scores"][method]-b["scores"][method])*d for a,b,d in accepted])
            result[method][name] = float(np.mean((product>0)+.5*(product==0)))
    if not result:
        raise ValueError("No same-scene comparison with a distinguishable utility")
    report = {m:bootstrap(values) for m,values in result.items()}
    report["paired_ablation"] = {m:bootstrap({g:v-result[m][g] for g,v in result[MAIN].items()})
                                  for m in (POINTWISE,CROSS,"v29")}
    report["informative_pairs"] = sum(counts.values())
    report["by_group"] = dict(result)
    return report


def render(root, cases):
    columns = ("Hazy input","MCT","PGL-Net-T","V29 selection + half","Same-scene selection + half","Clear")
    canvas = Image.new("RGB",(1680,225*len(cases)),"#edf2f1")
    draw = ImageDraw.Draw(canvas)
    for row,(label,frames) in enumerate(cases):
        draw.text((7,5+225*row),label,fill="#142e28")
        for i,(caption,frame) in enumerate(zip(columns,frames,strict=True)):
            frame.thumbnail((272,185),Image.Resampling.LANCZOS)
            draw.text((280*i+7,21+225*row),caption,fill="#142e28")
            canvas.paste(frame,(280*i+(280-frame.width)//2,35+225*row))
    canvas.save(root)


@torch.inference_mode()
def run(cohort):
    configure_cuda()
    state, fit = load()
    previous, old = load_state()
    forward, authors = experts()
    datasets = defaultdict(list)
    audit = []
    for i,(dataset,scene,group,hazy,clear,roi) in enumerate(input_pairs(cohort)):
        x,y = as_float(hazy),as_float(clear)
        mask = None if roi is None else roi[None,None].cuda()
        anchor = forward[0](x).clamp(0,1)
        fresh = dataset in ("lmhaze",*FRESH)
        attribute = features(x,anchor)
        scores = {method: value.item() for method,value in predict(state,attribute["anchor"]).items()}
        pscore,pamount,_ = actions(previous,attribute)[PREVIOUS]
        scores["v29"] = pscore.item()
        scores["hash_order"] = int.from_bytes(sha256(("budget-v31/"+scene).encode()).digest()[:6],"big")/2**48
        online = {method: scores[method] > state["thresholds"][method] for method in ARMS}
        online["v29"] = scores["v29"] > previous["thresholds"][PREVIOUS] and pamount.item() > 0
        second = forward[1](x).clamp(0,1)
        blend = torch.lerp(anchor,second,ALPHA)
        base,pgl,mix = (evaluate_rgb(v,y,mask,fresh) for v in (anchor,second,blend))
        scores["utility_oracle"] = mix["psnr"] - base["psnr"]
        if fresh:
            calls = [0,0]

            def reuse(operation, index):
                def get(image):
                    calls[index] += 1
                    return operation
                return get

            output, trace = restore(x,(reuse(anchor,0),reuse(second,1)),state)
            chosen = online[MAIN]
            expected = blend if chosen else anchor
            if calls != [1,int(chosen)] or trace["pgl"] != chosen or not torch.equal(output,expected):
                raise ValueError("Online path differs from audited scores")
            audit.append(dict(dataset=dataset,scene=scene,group=group,calls=calls))
        datasets[dataset].append(dict(dataset=dataset,scene=scene,group=group,scores=scores,online=online,
                                      mct=base,pgl=pgl,blend=mix))
        if (i+1)%45 == 0:
            print(dict(evaluation=cohort,pairs=i+1),flush=True)
    if not datasets:
        raise ValueError("Empty utility assessment")
    rows, curves = [], {}
    for dataset, records in datasets.items():
        curves[dataset] = {}
        for row in records:
            rows.extend(integrate(row,n,"all",n!="mct",n) for n in ("mct","pgl","blend"))
            rows.extend(integrate(row,name,"online",b,"blend" if b else "mct") for name,b in row["online"].items())
        for method in RANKERS:
            ordered = order(records,method)
            prefix = np.r_[0,np.cumsum([r["scores"]["utility_oracle"] for r in ordered])]
            n = len(records)
            calls = round(n*CALL_SHARE)
            rows.extend(integrate(r,method,"matched",j<calls,"blend" if j<calls else "mct") for j,r in enumerate(ordered))
            curves[dataset][method] = [float(prefix[round(n*z)]/n) for z in CURVE]
    target = ROOT / (cohort+"_utility_scores.csv")
    write_csv(target, rows)
    picture = ROOT / "lmhaze_routing_predictions.png"
    if cohort == "lmhaze":
        ordered = datasets[cohort]
        k = round(len(ordered)*CALL_SHARE)
        old_ids = {v["scene"] for v in order(ordered,"v29")[:k]}
        new_ids = {v["scene"] for v in order(ordered,MAIN)[:k]}
        # Show the earliest filename from each direction of the selection disagreement.
        targets = []
        for added, removed in ((new_ids,old_ids),(old_ids,new_ids)):
            candidates = [r for r in ordered if r["scene"] in added-removed]
            if candidates:
                targets.append(min(candidates,key=lambda r:r["scene"]))
        if not targets:
            targets = order(ordered,MAIN)[:2]
        shown = []
        names = {r["scene"]:r for r in targets}
        for _, scene, _, hazy, clear, _ in input_pairs(cohort):
            if scene not in names:
                continue
            x,y = as_float(hazy),as_float(clear)
            a,b = (op(x).clamp(0,1) for op in forward)
            mix = torch.lerp(a,b,ALPHA)
            gap = names[scene]["scores"]["utility_oracle"]
            shown.append((f"LMHaze {scene} | equal mix {gap:+.2f} dB vs MCT",
                          [as_rgb(v) for v in (x,a,b,mix if scene in old_ids else a,mix if scene in new_ids else a,y)]))
        render(picture,shown)
    count = {k:len(v) for k,v in datasets.items()}
    ledger = dict(cohort=cohort, pairs=count, comparisons=paired_comparisons(rows), summary=summarize(rows),
                  same_scene_concordance=concordance(datasets["lmhaze"]) if cohort=="lmhaze" else None,
                  budget_grid=list(CURVE), psnr_gain_curves=curves, weights_SHA256=fit["weights_SHA256"],
                  v29_gate_SHA256=old["weights_SHA256"], lmhaze_manifest_SHA256=digest(CARD) if cohort=="lmhaze" else None,
                  table_SHA256=digest(target), evaluator_SHA256=digest(__file__),
                  paired_preds_SHA256=digest(picture) if cohort=="lmhaze" else None,
                  online_audit=audit, authors=authors, finished_utc=now())
    atomic_write(ROOT/(cohort+"_utility.json"),ledger)
    for key,r in ledger["summary"].items():
        if key.split("/")[0] in (cohort,"rw_reserve","revide","dense","ohaze","i_haze") and key.endswith((MAIN,CROSS,"v29","utility_oracle")):
            print(key,{t:round(r[t],4) for t in ("psnr","gain","calls","severe")},flush=True)
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("lmhaze","prior"))
    run(parser.parse_args().cohort)

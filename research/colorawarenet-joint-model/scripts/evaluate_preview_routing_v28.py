"""Evaluate trained preview routes on development pairs and held-out BeDDE cities."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path, PurePosixPath
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from dehaze_seg.utils import write_csv
from scripts.calibrate_expert_choice_v27 import switches
from scripts.evaluate_curve_transfer_v25 import TESTS, as_float, as_rgb, samples
from scripts.expert_fusion_v26 import experts
from scripts.infer_guarded_expert_v27 import load_router
from scripts.prepare_rw2ah_v25 import ROOT as V25
from scripts.preview_routing_v28 import FEATURES, HARM, choose, full_descriptor, load_policies, preview_descriptors, restore
from scripts.preview_routing_v28 import specialist_outputs
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_detail_data_v24 import DATASETS, RealPairs, image_tensor
from scripts.real_haze_data_v22 import fingerprint, overlap_check
from scripts.stage_bedde_v28 import RECEIPT as BEDDE
from scripts.stage_preview_training_v28 import ROOT
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda
from scripts.evaluate_ntire24_v27 import DATA as NTIRE, train_signatures


CONTROLS = ("author_mct","author_pgl","equal_blend","v27_guarded","preview_sign")
MAIN = "scales_tail"


def development():
    with RealPairs(DATASETS) as registry:
        manifest = read(V25/"data_manifest.json")
        for dataset in TESTS:
            for scene,hazy,clear in samples(registry,dataset,manifest):
                group = scene.split("-")[0] if dataset=="rw2ah" else scene
                yield dataset,scene,group,hazy,clear,None
    source = read(NTIRE)
    if digest(source["file"])!=source["SHA256"]:
        raise ValueError("NTIRE24 cache changed")
    for row in torch.load(source["file"],map_location="cpu",weights_only=True):
        yield "ntire24",row["id"],row["id"],row["hazy"],row["clean"],None


def bedde():
    from scipy.io import loadmat

    registration = read(BEDDE)
    parent = Path(registration["directory"])
    cities = defaultdict(lambda: defaultdict(dict))
    for member in registration["members"]:
        path = PurePosixPath(member)
        city,kind = path.parts[-3].lower(),path.parts[-2].lower()
        full = parent/path
        if digest(full)!=registration["image_hashes"][member]:
            raise ValueError("An author BeDDE input changed")
        cities[city][kind][path.stem.lower()] = full
    previous = list(train_signatures())
    groups, excluded = {}, []
    for city,roles in sorted(cities.items()):
        if len(roles["gt"])!=1:
            raise ValueError("Expected one clear reference for BeDDE " + city)
        gt = next(iter(roles["gt"].values()))
        with Image.open(gt) as opened:
            clear = ImageOps.exif_transpose(opened).convert("RGB")
        signatures = dict(id=city,signature=fingerprint(clear),
                          normalized_clean_SHA256=hashlib.sha256(np.asarray(clear).tobytes()).hexdigest(),
                          clean={"SHA256":digest(gt)})
        duplicates,_ = overlap_check(previous,[signatures])
        if duplicates:
            excluded.extend(duplicates)
            continue
        scale = min(1024,max(clear.size))/max(clear.size)
        size = tuple(round(v*scale) for v in clear.size)
        rgb_ref = image_tensor(clear.resize(size,Image.Resampling.BILINEAR))
        groups[city] = len(roles["fog"])
        for name,hazy in sorted(roles["fog"].items()):
            mask_file = roles["mask"].get(name+"_mask")
            if mask_file is None:
                raise ValueError("Missing author ROI mask for " + name)
            mask = np.squeeze(loadmat(mask_file,variable_names=["mask"])["mask"])
            if mask.shape != (clear.height,clear.width):
                raise ValueError("BeDDE mask/clear coordinates differ for " + name)
            roi = torch.from_numpy(np.array(Image.fromarray(mask>0).resize(size,Image.Resampling.NEAREST))).bool()
            if not roi.any():
                raise ValueError("BeDDE valid mask is empty for " + name)
            with Image.open(hazy) as opened:
                hazy_rgb = image_tensor(ImageOps.exif_transpose(opened).convert("RGB").resize(size,Image.Resampling.BILINEAR))
            yield "bedde",name,city,hazy_rgb,rgb_ref,roi
    atomic_write(ROOT/"bedde_pairs.json",dict(city_counts=groups,excluded_training_overlaps=excluded,
                 archive_SHA256=registration["SHA256"],scope="DCT perceptual fingerprint and thumbnail correlation between authors' city references and local fit scenes. Supplied MAT mask only. Authored official VI/RI are outside these PSNR scores.",
                 finished_utc=now()))


def psnr(pred,target,mask):
    errors = (pred-target).square()
    mse = errors.mean() if mask is None else (errors*mask).sum()/(3*mask.sum())
    return (-10*mse.clamp_min(1e-12).log10()).item()


def metrics(rows,key=("dataset","method")):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in key)].append(row)
    summary = {}
    for label,population in groups.items():
        gain = np.array([r["gain_vs_MCT"] for r in population])
        picks = sum(r["pgl"] is True for r in population)
        summary["/".join(label)] = dict(n=len(gain),psnr=float(np.mean([r["psnr"] for r in population])),
                    gain=float(gain.mean()),pgl=picks,wins=int((gain>1e-6).sum()),losses=int((gain< -1e-6).sum()),
                    severe=int((gain<-HARM).sum()),worst=float(gain.min()),
                    excess_loss=float(np.maximum(-gain-HARM,0).mean()))
    return summary


def city_bootstrap(groups):
    by_policy = defaultdict(list)
    for key,row in groups.items():
        _,_,method = key.split("/")
        by_policy[method].append(row["gain"])
    rng = np.random.default_rng(28004)
    result = {}
    for key,row in by_policy.items():
        indices = rng.integers(0,len(row),size=(10000,len(row)))
        means = np.asarray(row)[indices].mean(1)
        result[key] = dict(n_cities=len(row),mean=float(np.mean(row)),ci95=np.percentile(means,(2.5,97.5)).tolist(),
                           positive_cities=int(np.count_nonzero(np.asarray(row)>1e-6)))
    return result


def coverage_curve(rows):
    methods = (*FEATURES,"preview_sign")
    reference = {(r["dataset"],r["scene"]):r["gain_vs_MCT"] for r in rows if r["method"]=="author_pgl"}
    answer = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        answer[dataset] = {}
        for method in methods:
            ordered = sorted((r for r in rows if r["dataset"]==dataset and r["method"]==method),
                             key=lambda x:(-x["decision_score"],x["scene"]))
            gain = np.array([reference[dataset,r["scene"]] for r in ordered])
            answer[dataset][method] = [dict(picks=n,severe=int((gain[:n]<-HARM).sum()),
                                            gain=float(gain[:n].sum()/len(gain))) for n in range(len(gain)+1)]
    return answer


def audit_and_time(image,models,forward,v27):
    expected = specialist_outputs(forward,image)
    methods = {"fixed_mct":forward[0],"fixed_pgl":forward[1],"v27_guarded":lambda x: full_v27(x,forward,v27)}
    result = {}
    for name in FEATURES:
        calls = []

        def monitored(operation,x,idx):
            calls.append((idx,tuple(x.shape[-2:])))
            return operation(x)

        tracked = tuple(lambda x,i=i,f=f: monitored(f,x,i) for i,f in enumerate(forward))
        value,selection = restore(image,tracked,models,name)
        full_calls = sum(p==tuple(image.shape[-2:]) for _,p in calls)
        want = 2 if name=="full_tail" else 1
        if full_calls!=want or not torch.equal(value,expected[int(selection)]):
            raise ValueError("Preview branch audit: incorrect full-image execution")
        result[name] = dict(calls=calls,full_calls=full_calls)
        methods[name] = lambda x,m=name: restore(x,forward,models,m)
    timings = {}
    for name, operation in methods.items():
        for _ in range(4):
            operation(image)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(12):
            operation(image)
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter()-start)*1000/12
    return dict(gpu=torch.cuda.get_device_name(),image_shape=list(image.shape),fp32=True,actual_full_calls=result,
                wall_ms=timings,segmentation_included=False)


def full_v27(image,forward,model):
    full = specialist_outputs(forward,image)
    chosen = switches(model,image,*full)["guarded"]
    return full[int(chosen)]


def save_cases(cases,cohort):
    canvas = Image.new("RGB",(1750,264*len(cases)),"#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    for i,(label,images) in enumerate(cases):
        for j,(name,image) in enumerate(zip(("Hazy input","Default MCT","PGL-Net-T","Preview choice","Clear reference"),images)):
            image.thumbnail((344,234),Image.Resampling.LANCZOS)
            x,y = j*350,i*264
            canvas.paste(image,(x+(350-image.width)//2,y+27))
            draw.text((x+6,y+7),f"{label} | {name}",fill="#142837")
    target = ROOT/(cohort+"_predictions.png")
    canvas.save(target)
    return digest(target)


@torch.inference_mode()
def evaluate(cohort):
    configure_cuda()
    models,training = load_policies()
    forward,authors = experts()
    v27,old = load_router()
    rows,cases,count = [],[],defaultdict(int)
    image_for_profile = None
    for dataset,scene,group,hazy,clear,roi in (bedde() if cohort=="bedde" else development()):
        x,y = (as_float(v) for v in (hazy,clear))
        valid = roi[None,None].cuda() if roi is not None else None
        props = preview_descriptors(forward,x)
        full = specialist_outputs(forward,x)
        props["full"] = full_descriptor(x,full)
        scores = choose(models,props)
        measured = [psnr(v,y,valid) for v in full]
        routes = {method:value>0 for method,value in scores.items()}
        routes["v27_guarded"] = switches(v27,x,*full)["guarded"]
        decisions = {**routes,"author_mct":False,"author_pgl":True,"oracle":measured[1]>measured[0]}
        for method,selected in decisions.items():
            rows.append(dict(dataset=dataset,scene=scene,group=group,method=method,pgl=selected,
                             psnr=measured[selected],gain_vs_MCT=measured[selected]-measured[0],
                             decision_score=scores.get(method)))
        blended = psnr(.5*(full[0]+full[1]),y,valid)
        rows.append(dict(dataset=dataset,scene=scene,group=group,method="equal_blend",pgl=None,
                         psnr=blended,gain_vs_MCT=blended-measured[0],decision_score=None))
        if (dataset,scene)==("ohaze","005") or (dataset=="bedde" and not cases):
            cases.append((f"{dataset} {scene}",[as_rgb(v) for v in (x,*full,full[routes[MAIN]],y)]))
        if (dataset=="dense" and count[dataset]==0) or (dataset=="bedde" and image_for_profile is None):
            image_for_profile = x
        count[dataset] += 1
        if count[dataset]%25==0:
            print(dict(dataset=dataset,images=count[dataset]),flush=True)
    if cohort=="development" and sum(count.values())!=187:
        raise ValueError("Previous development cohort membership changed")
    report = ROOT/(cohort+"_metrics.csv")
    write_csv(report,rows)
    curve_path = ROOT/(cohort+"_coverage.json")
    atomic_write(curve_path,coverage_curve(rows))
    grouped = metrics(rows,("dataset","group","method")) if cohort=="bedde" else None
    result = dict(cohort=cohort,counts=dict(count),methods=list((*CONTROLS,*FEATURES,"oracle")),rows=len(rows),
                  scores=metrics(rows),city_scores=grouped,site_bootstrap=city_bootstrap(grouped) if grouped else None,
                  tables_SHA256=digest(report),coverage_SHA256=digest(curve_path),sheet_SHA256=save_cases(cases,cohort),
                  cost=audit_and_time(image_for_profile,models,forward,v27),
                  trained_SHA256=training["weights_SHA256"],v27_SHA256=old["calibration_SHA256"],
                  author_sources=authors,protocol_SHA256=digest(__file__),finished_utc=now())
    atomic_write(ROOT/(cohort+".json"),result)
    for method in (*CONTROLS,*FEATURES,"oracle"):
        print(method,{key.split("/")[0]:[round(v["psnr"],3),v["pgl"],v["severe"]]
                      for key,v in result["scores"].items() if key.endswith("/"+method)},flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("development","bedde"))
    args = parser.parse_args()
    evaluate(args.cohort)


if __name__=="__main__":
    main()

"""Score locked V25 checkpoints and author references on the same paired RGB pixels."""
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from dehaze_seg.utils import write_csv
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_parallel_rgb_v8 import bind as bind_rgb
from scripts.evaluate_real_transfer_v22 import METRICS, quality
from scripts.prepare_rw2ah_v25 import ROOT
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import DATASETS, RealPairs, image_tensor, long_edge
from scripts.train_curve_transfer_v25 import METHODS, initial_model, load_fits, predict, starts
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


LONG = 1024
TESTS = ("rw2ah", *DATASETS)
CONTRASTS = (("direct_mixed","direct_nh"), ("direct_mixed","v24_start"), ("affine_field","direct_mixed"),
             ("curves","affine_field"), ("curves","direct_mixed"), ("monotone","curves"),
             ("uniform_edges","curves"), ("visible_edges","uniform_edges"))
NAMES = dict(rw2ah="RW2AH",nh_holdout="NH-HAZE holdout",dense="Dense-Haze",ohaze="O-HAZE",i_haze="I-HAZE")
DISPLAY = ("hazy","direct_mixed","affine_field","curves","monotone","visible_edges","author_pgl","author_mct","clear")
LABELS = dict(hazy="Hazy input",v24_start="V24 initial model",direct_nh="NH only",direct_mixed="Direct, mixed data",
              affine_field="Matched affine field",curves="Curve field",monotone="Positive diagonal slopes",
              uniform_edges="Uniform edges",visible_edges="Visible edges",author_pgl="PGL-Net-T (authors)",
              author_mct="MCT (authors)",clear="Clear reference")


def samples(registry, source, manifest):
    if source=="rw2ah":
        test = manifest["test"]
        if digest(test["file"])!=test["SHA256"]:
            raise ValueError("Registered RW2AH test cache changed")
        for row in torch.load(test["file"],map_location="cpu",weights_only=True):
            yield row["id"], row["hazy"], row["clean"]
        return
    for scene in registry.ids(source):
        hazy, clear = registry.read(source,scene)
        yield scene, *(image_tensor(long_edge(item,LONG)) for item in (hazy,clear))


def as_float(tensor):
    return tensor[None].to("cuda").float()/255


def as_rgb(tensor):
    return Image.fromarray((tensor[0]*255).round().to(torch.uint8).permute(1,2,0).cpu().numpy())


def summarize(rows):
    grouping = defaultdict(list)
    indexing = defaultdict(dict)
    for row in rows:
        grouping[row["dataset"],row["method"],row["start"]].append([row[k] for k in METRICS])
        indexing[row["dataset"]][row["scene"],row["method"],row["start"]] = row
    result = defaultdict(list)
    for (dataset,method,_), scores in grouping.items():
        result[f"{dataset}/{method}"].append(np.mean(scores,axis=0))
    means = {key:dict(zip(METRICS,np.mean(value,axis=0).tolist())) for key,value in result.items()}
    rng = np.random.default_rng(250812)
    comparisons = defaultdict(dict)
    for dataset, by_image in indexing.items():
        for treatment, control in CONTRASTS:
            scenes, start, prefix = defaultdict(list), defaultdict(list), defaultdict(list)
            for (scene,method,origin), row in by_image.items():
                if method!=treatment:
                    continue
                ref = by_image[scene,control,origin]
                diff = np.array([row[k]-ref[k] for k in METRICS])
                scenes[scene].append(diff)
                start[origin].append(float(diff[0]))
            matrix = np.array([np.mean(values,axis=0) for values in scenes.values()])
            idx = rng.integers(0,len(matrix),size=(4000,len(matrix)))
            intervals = np.percentile(matrix[idx].mean(1),[2.5,97.5],axis=0)
            for scene, diff in zip(scenes,matrix):
                prefix[scene.split("-")[0]].append(float(diff[0]))
            comparisons[dataset][f"{treatment}-{control}"] = dict(
                mean=dict(zip(METRICS,matrix.mean(0).tolist())),
                image_bootstrap95=dict(zip(METRICS,intervals.T.tolist())),
                PSNR_by_start={k:float(np.mean(v)) for k,v in start.items()},
                PSNR_by_prefix={k:float(np.mean(v)) for k,v in prefix.items()} if dataset=="rw2ah" else None)
    return means,comparisons


def save_examples(examples):
    folder = ROOT/"predictions"
    folder.mkdir(exist_ok=True)
    canvas = Image.new("RGB",(3*400,len(examples)*3*280),"#f1f5f8")
    draw = ImageDraw.Draw(canvas)
    exports = []
    for row,(dataset,scene,start,images,scores) in enumerate(examples):
        for i,method in enumerate(DISPLAY):
            x,y = (i%3)*400, (row*3+i//3)*280
            image = images[method]
            path = folder/f"{dataset}-{scene}-{method}.png"
            image.save(path)
            preview = image.copy()
            preview.thumbnail((394,239),Image.Resampling.LANCZOS)
            canvas.paste(preview,(x+(400-preview.width)//2,y+31))
            db = f"  {scores[method]:.2f} dB" if method in scores else ""
            draw.text((x+7,y+5),f"{NAMES[dataset]} {scene} | {LABELS[method]}{db}",fill="#152b3d")
            exports.append(dict(dataset=dataset,scene=scene,method=method,start=start,
                                file=str(path),SHA256=digest(path)))
    path = ROOT/"comparison.png"
    canvas.save(path)
    return dict(files=exports,comparison=str(path),SHA256=digest(path))


def profile(image, fitted, published):
    speeds = {}
    for name, forward in {
        **{k:partial(predict,v,k) for k,v in fitted.items()},
        **published,
    }.items():
        for _ in range(5):
            forward(image)
        torch.cuda.synchronize()
        previous = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        for _ in range(15):
            forward(image)
        b.record()
        b.synchronize()
        speeds[name] = dict(ms=a.elapsed_time(b)/15,extra_peak_MiB=(torch.cuda.max_memory_allocated()-previous)/2**20)
    return dict(input=list(image.shape),FP32_single_image_model_only=True,
                methods=speeds,gpu=torch.cuda.get_device_name())


@torch.inference_mode()
def main():
    configure_cuda()
    completed = read(ROOT/"completed.json")
    trained = load_fits()
    prior = {p[0]["video"]: initial_model(*p).eval() for p in starts()}
    bind_rgb()
    mct, mct_info = load_author_reference()
    pgl, pgl_info = load_pgl()
    published = {"author_mct":partial(author_rgb,mct), "author_pgl":partial(pgl_rgb,pgl)}
    manifest = read(ROOT/"data_manifest.json")
    first = next(iter(trained))
    registered = dict(plan_SHA256=completed["plan_SHA256"], fits_SHA256=digest(ROOT/"completed.json"),
                      sources={str(p):digest(p) for p in (Path(__file__),ROOT/"data_manifest.json",
                               REPO/"scripts/author_pgl_v25.py",REPO/"scripts/evaluate_real_transfer_v22.py")},
                      geometry="Common full aspect retained long edge 1024. RW2AH native sizes <= 1024 remain unchanged.",
                      metric="FP32 RGB clamped to [0,1] before PSNR, gaussian SSIM, biases and gradient MAE. Equal weight per image and per V24 starting model. Image resampling CIs condition on the observed starts and test cohort.",
                      fixed_examples="First sorted archived example for each of two datasets. Use the first registered V24 initialization, independent of scores.",
                      contrasts=list(map(list,CONTRASTS)), mct=mct_info, pgl=pgl_info)
    lock = ROOT/"evaluation_plan.json"
    if lock.exists() and read(lock)!=registered:
        raise ValueError("V25 evaluation differs from the registered scope")
    if not lock.exists():
        atomic_write(lock,registered)
    rows, examples, counts, sizes = [], [], {}, {}
    benchmark_image = None
    with RealPairs(DATASETS) as registry:
        for dataset in TESTS:
            count = 0
            for scene,hazy,clear in samples(registry,dataset,manifest):
                image,target = as_float(hazy),as_float(clear)
                h,w = target.shape[-2:]
                sizes[f"{dataset}/{scene}"] = [w,h]
                if dataset=="nh_holdout" and count==0:
                    benchmark_image = image
                keep = dataset in ("rw2ah","dense") and count==0
                saved, db = {"clear":as_rgb(target)} if keep else {}, {}

                def score(output, name, origin):
                    if not torch.isfinite(output).all():
                        raise ValueError(f"Nonfinite RGB during {name} evaluation")
                    restored = output.clamp(0,1)
                    values = quality(restored,target)[0]
                    rows.append(dict(dataset=dataset,scene=scene,start=origin,method=name,
                                     width=w,height=h,**dict(zip(METRICS,values))))
                    if keep and name in DISPLAY and origin in (first,"fixed"):
                        saved[name],db[name] = as_rgb(restored),values[0]

                score(image,"hazy","fixed")
                for name,forward in published.items():
                    score(forward(image),name,"fixed")
                for start,models in trained.items():
                    score(author_rgb(prior[start],image),"v24_start",start)
                    for name,model in models.items():
                        score(predict(model,name,image),name,start)
                if keep:
                    examples.append((dataset,scene,first,saved,db))
                count += 1
                if count%10==0:
                    print(dict(dataset=dataset,scored_pairs=count),flush=True)
            counts[dataset] = count
            print(dict(finished=dataset,scored_pairs=count),flush=True)
    expected = {"rw2ah":manifest["test"]["count"],"nh_holdout":10,"dense":55,"ohaze":45,"i_haze":30}
    if counts!=expected or len(rows)!=sum(counts.values())*(3+len(trained)*(len(METHODS)+1)):
        raise ValueError("Incorrect image or model count in V25 evaluation")
    table = ROOT/"paired_metrics.csv"
    write_csv(table,rows)
    means, contrasts = summarize(rows)
    images = save_examples(examples)
    latency = profile(benchmark_image,trained[first],published)
    report = dict(counts=counts, geometry=sizes, scores=means, contrasts=contrasts, author_pgl=pgl_info,
                  author_mct=mct_info,mct_params=sum(x.numel() for x in mct.parameters()),
                  prediction_artifacts=images,latency=latency,table_rows=len(rows),table_SHA256=digest(table),
                  finished_utc=now(),evaluation_plan_SHA256=digest(lock))
    atomic_write(ROOT/"restoration.json",report)
    print(dict(scored=counts,rows=len(rows),PSNR={k:round(v["psnr"],3) for k,v in means.items()}),flush=True)


if __name__=="__main__":
    main()

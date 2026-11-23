"""Evaluate the RGB complement, its independent controls, and the MCAF-Net comparison."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import torch

from boundary_refinement.models.segmentation.coarse_semantics import blend, native_labels, paint
from boundary_refinement.models.restoration.complementary_rgb import MODES, ROOT, adjustment, context, models
from boundary_refinement.training.decision_overlap import MANIFEST, pair_index
from boundary_refinement.inference.dual_view_inference import two_views
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.photographic.evaluate_curve_transfer import as_float
from boundary_refinement.evaluation.photographic.evaluate_overlap_quadratic import POPULATION, samples
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter, resize_image
from boundary_refinement.evaluation.paired.evaluate_real_transfer import gaussian_ssim
from boundary_refinement.evaluation.photographic.evaluate_relative_utility import HELD, HELD_DOMAIN, additional_samples
from boundary_refinement.evaluation.photographic.evaluate_scene_utility import bootstrap
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.data.cohorts.fresh_real import CARD, RR, image, registration
from boundary_refinement.baselines import mcaf_attention
from boundary_refinement.baselines.mcaf_reference import ROOT as MCAF, trained as mcaf_models
from boundary_refinement.training.overlap_quadratic import SEEDS, references
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor
from boundary_refinement.data.real_haze_data import fingerprint
from boundary_refinement.models.restoration.local_utility import fitted
from boundary_refinement.data.staging.satehaze_test import MANIFEST as SATE, samples as sate_samples, stage as sate_stage
from boundary_refinement.models.restoration.single_anchor_student import area_psnr
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


NEW = ROOT / "rrshid-fresh.json"
CONDITIONS = ("thin_fog", "moderate_fog", "thick_fog")
N = 60


def independent_crops():
    rr, _ = registration()
    origins = dict(archive=rr["SHA256"], older={p.name:digest(p) for p in (MANIFEST,CARD,HELD)})
    if NEW.exists():
        locked=read(NEW)
        if locked["origins"]!=origins:
            raise ValueError("A registered prior RRSHID split changed")
        return locked
    previously_used = read(MANIFEST)["pairs"] + read(HELD)["records"]
    previously_used += [r for r in read(CARD)["records"] if r["dataset"]=="rrshid"]
    ids={r["id"].split("/")[-1] for r in previously_used}
    prep=[]
    with ZipFile(RR/"RRSHID.zip") as z:
        train=pair_index(z,{"train"})
        images=[fingerprint(image(z,p["clean"])) for p in previously_used]
        for weather in CONDITIONS:
            available = [i for (_,c,i) in train if c==weather and i not in ids]
            order=sorted(available,key=lambda s:sha256(f"v59-fresh/{weather}/{s}".encode()).digest())
            initial=len(prep)
            for i in order:
                if i in ids:
                    continue
                rec=train["train",weather,i]
                clean=image(z,rec["clean"])
                if clean.size != (256,256) or image(z,rec["haze"]).size != clean.size:
                    raise ValueError("The additional paired tile is unaligned")
                p,d=fingerprint(clean)
                if any(np.count_nonzero(p!=q)<=8 and float(d @ y)>=.96 for q,y in images):
                    continue
                item=f"{weather}/{i}"
                prep.append(dict(id=item,group=item,condition=weather,**rec))
                ids.add(i)
                images.append((p,d))
                if len(prep)-initial==N:
                    break
            if len(prep)-initial != N:
                raise ValueError("The fixed unused RRSHID sample is incomplete")
    output=dict(origins=origins, records=prep,counts=dict(Counter(p["condition"] for p in prep)),locked_utc=now(),
                rule="Draw 60 previously unused original training crops per severity by filename hash. Exclude numeric IDs "
                     "and near-duplicate clear views shared with all previous train/validation/test and V56 holdout crops.")
    atomic_write(NEW,output)
    print(dict(new_samples=output["counts"]),flush=True)
    return output


def fresh_pairs(cohort):
    with ZipFile(RR/"RRSHID.zip") as data:
        for r in cohort["records"]:
            x,y=(image_tensor(image(data,r[k])) for k in ("haze","clean"))
            yield "rrshid_fresh",r["id"],r["group"],x,y,None,r["condition"]


def tables(rows):
    full = defaultdict(list)
    paired = defaultdict(lambda:defaultdict(list))
    for r in rows:
        scopes = (r["dataset"], f"{r['dataset']}/{r['condition']}") if r["condition"] in (
            *CONDITIONS,"thin","moderate","thick") else (r["dataset"],)
        for d in scopes:
            full[d,r["method"]].append(r)
            paired[d,r["group"],r["id"]][r["method"]].append(r["psnr"])
    stats = {"/".join(k):{f:mean(r[f] for r in v if r[f] is not None) for f in ("psnr","ssim")
                         if any(r[f] is not None for r in v)} for k,v in full.items()}
    groups = defaultdict(lambda:defaultdict(list))
    for (d,group,_), values in paired.items():
        for l,r in (("local","base"),("local","global"),("projected","local"),("projected","base"),
                    ("projected","projected_without_subtraction"),("local","mct"),("local","mcaf")):
            if l in values and r in values:
                groups[d,l+"-vs-"+r][group].append(mean(values[l])-mean(values[r]))
    return stats, {"/".join(k):bootstrap({g:mean(v) for g,v in s.items()}) for k,s in groups.items()}


@torch.inference_mode()
def evaluate():
    configure_cuda()
    pack=independent_crops()
    external=sate_stage()
    prior=read(HELD)
    expected={**POPULATION,HELD_DOMAIN:len(prior["records"]),"rrshid_fresh":len(pack["records"]),
              "satehaze":len(external["records"])}
    recipe=dict(code_SHA256=digest(__file__), fits_sha256=digest(ROOT/"completed.json"),
        sources={p.name:digest(p) for p in (HELD,NEW,SATE,CARD)},
        mcaf_SHA256=digest(MCAF/"completed.json"), mcaf_attention_SHA256=digest(mcaf_attention.__file__),
        population=expected,
        comparison="Frozen V56 seed 32003 as a common start. No output-amplitude tuning. Keep RGB corrections clipped "
                   "to the display interval. Compare three matched small heads plus the frozen projected head with "
                   "its spatial subtraction removed. Run the short-budget MCAF-Net comparison on the three RRSHID "
                   "cohorts using the native 256-pixel grid and efficient global attention with unchanged projections and scale.")
    lock=ROOT/"evaluation_protocol.json"
    if lock.exists() and read(lock)!=recipe:
        raise ValueError("V59 scoring recipe has changed")
    if not lock.exists():
        atomic_write(lock,recipe)
    rgb, prior_models, bank, mcaf = references(), fitted(), models(), mcaf_models()
    for model in mcaf.values():
        mcaf_attention.memory_efficient(model)
    count, rows, panels = Counter(), [], []
    gallery = ("rrshid_fresh","satehaze","nt_haze","lmhaze","realhyper_csnc")
    stream = chain(samples(),additional_samples(prior),fresh_pairs(pack),sate_samples(external))
    for d,i,g,x,y,roi,weather in stream:
        view, truth = as_float(x), as_float(y)
        mask = roi.cuda() if roi is not None else None
        ctx, mct = context(view,rgb,prior_models)
        rgb_out={("base",0):ctx.anchor,("mct",0):mct}
        for seed in SEEDS:
            for m in MODES:
                rgb_out[m,seed] = adjustment(ctx,bank[m,seed],m).clamp(0.,1.)
            rgb_out["projected_without_subtraction",seed] = adjustment(ctx,bank["projected",seed],"local").clamp(0.,1.)
        if d.startswith("rrshid"):
            for seed in SEEDS:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    output = mcaf[seed](view)[0]
                rgb_out["mcaf",seed] = output.float().clamp(0.,1.)
        for (name, seed), output in rgb_out.items():
            score = area_psnr(output,truth,mask)
            ssim = float(gaussian_ssim(output,truth)[0]) if seed in (0,SEEDS[0]) else None
            rows.append(dict(dataset=d,id=i,group=g,condition=weather,method=name,seed=seed,psnr=score,ssim=ssim))
        if d in gallery and not count[d]:
            picture = [view,mct,ctx.anchor,*[rgb_out[m,SEEDS[0]] for m in MODES]]
            names = ["Input","Published MCT","Previous V56","Global RGB","Local RGB","Projected RGB"]
            if ("mcaf",SEEDS[0]) in rgb_out:
                picture.append(rgb_out["mcaf",SEEDS[0]])
                names.append("MCAF-Net short fit")
            panels.append((f"{d} / {i}",[to_rgb(t) for t in (*picture,truth)],(*names,"Clear reference")))
        count[d]+=1
        if count[d]==expected[d]:
            print(d,count[d],{m:round(mean(r["psnr"] for r in rows if r["dataset"]==d and r["method"]==m),4)
                for m in ("mct","base",*MODES,"mcaf") if any(r["dataset"]==d and r["method"]==m for r in rows)},flush=True)
    if count!=expected:
        raise ValueError(f"V59 evaluator lost samples {count}")
    stats, contrast=tables(rows)
    result, panel = ROOT/"individual.json",ROOT/"predictions.png"
    atomic_write(result,rows)
    draw(panel,panels)
    atomic_write(ROOT/"test.json",dict(counts=dict(count),rows=len(rows),means=stats,paired=contrast,
        table_SHA256=digest(result),panel_SHA256=digest(panel),protocol_SHA256=digest(lock),finished_utc=now()))


@torch.inference_mode()
def infer(path, directory, mode):
    configure_cuda()
    keys=("raw","base","color_result","mask","color_mask","overlay","comparison")
    if path.resolve() in {(directory/(k+".png")).resolve() for k in keys}:
        raise ValueError("Use a separate inference location")
    with Image.open(path) as handle:
        pixels=np.array(ImageOps.exif_transpose(handle).convert("RGB"))
    x=as_float(torch.from_numpy(pixels.copy()).permute(2,0,1))
    ctx,_=context(x,references(),fitted())
    corrected=to_rgb(adjustment(ctx,models()[mode,SEEDS[0]],mode).clamp(0.,1.))
    _,_,pred=two_views(*load_segmenter()[:3],resize_image(pixels)[0])
    mask=native_labels(pred,pixels.shape[:2]).byte().cpu().numpy()
    if corrected.shape[:2]!=mask.shape or not np.isin(mask,np.arange(19)).all():
        raise ValueError("The double output differs from its original pixel or label grid")
    frames=dict(raw=pixels,base=to_rgb(ctx.anchor),color_result=corrected,mask=mask,color_mask=paint(mask),overlay=blend(pixels,mask))
    directory.mkdir(parents=True,exist_ok=True)
    for name,rgb in frames.items():
        Image.fromarray(rgb).save(directory/(name+".png"))
    draw(directory/"comparison.png",(
        (path.name,[frames[k] for k in ("raw","base","color_result")],("Hazy RGB","V56","V59 RGB output")),
        ("Cityscapes 19 classes",[frames[k] for k in ("raw","color_mask","overlay")],("Original","Mask","Overlay"))))
    atomic_write(directory/"metadata.json",dict(size=list(mask.shape[::-1]),mode=mode,source_SHA256=digest(path),
        files={k+".png":digest(directory/(k+".png")) for k in keys},finished_utc=now()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("register")
    commands.add_parser("test")
    demo=commands.add_parser("infer")
    demo.add_argument("--input",type=Path,required=True)
    demo.add_argument("--output",type=Path,required=True)
    demo.add_argument("--mode",choices=MODES,default="local")
    arguments=parser.parse_args()
    if arguments.command=="register":
        independent_crops()
    elif arguments.command=="test":
        evaluate()
    else:
        infer(arguments.input,arguments.output,arguments.mode)

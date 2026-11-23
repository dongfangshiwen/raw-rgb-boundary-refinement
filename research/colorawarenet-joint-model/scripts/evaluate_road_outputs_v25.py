"""Compare raw and restored segmentation inputs, and export native dual outputs."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from dehaze_seg.metrics import confusion_matrix
from scripts.author_pgl_v25 import load_pgl, pgl_rgb
from scripts.evaluate_author_mct_v8 import load_author_reference
from scripts.evaluate_parallel_rgb_v8 import bind as bind_rgb
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, aggregate, load_pairs, load_segmenter, measured_record, resize_image
from scripts.infer_mct_segformer_v24 import CITY_COLORS
from scripts.prepare_rw2ah_v25 import ROOT
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_detail_data_v24 import ROOT as V24, image_tensor
from scripts.train_curve_transfer_v25 import METHODS, load_fits, predict
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


DATA = {"foggy_zurich_testv2":40,"foggy_driving_fine":33}
V25 = ("direct_mixed","curves","monotone","visible_edges")


def logits(model, mean, std, image, size):
    value = model(pixel_values=(image-mean)/std).logits.float()
    return F.interpolate(value,size,mode="bilinear",align_corners=False)


@torch.inference_mode()
def export(image_path, destination, rgb_infer, segmenter, mean, std, card, restoration):
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    native = image_tensor(image)[None].cuda().float()/255
    raw,_ = resize_image(np.asarray(image))
    restored = rgb_infer(native).clamp(0,1)
    prediction = logits(segmenter,mean,std,raw,native.shape[-2:])
    if restored.shape!=native.shape or not torch.isfinite(restored).all() or not torch.isfinite(prediction).all():
        raise ValueError("Invalid native RGB or semantic prediction")
    mask = prediction.argmax(1)[0].byte().cpu().numpy()
    pixels = (255*restored[0]).round().byte().permute(1,2,0).cpu().numpy()
    overlay = np.rint(.6*pixels+.4*CITY_COLORS[mask]).astype(np.uint8)
    if mask.shape!=pixels.shape[:2] or not np.isin(mask,range(19)).all():
        raise ValueError("Native semantic geometry or labels changed")
    destination.mkdir(parents=True,exist_ok=True)
    files = {}
    for name,data in (("restored.png",pixels),("mask.png",mask),("overlay.png",overlay)):
        path = destination/name
        Image.fromarray(data).save(path)
        files[name] = digest(path)
    record = dict(input=str(image_path.resolve()),input_SHA256=digest(image_path),native_size=list(image.size),
                  restoration=restoration,segmentation_input="raw original",segmentation=card["model"],
                  labels=segmenter.config.id2label,output_SHA256=files,finished_utc=now())
    atomic_write(destination/"metadata.json",record)
    return record


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--method",choices=METHODS,default="curves")
    args = parser.parse_args()
    if (args.input is None)!=(args.output is None):
        parser.error("--input and --output are used together")
    configure_cuda()
    fitted = load_fits()
    first = next(iter(fitted))
    fits = read(ROOT/"completed.json")
    first_receipt = next(v for v in fits["runs"] if v["video"]==first)
    segmenter,mean,std,card = load_segmenter()
    restore = partial(predict,fitted[first][args.method],args.method)
    branch = dict(method=args.method,start=first,checkpoint_SHA256=first_receipt["checkpoint_SHA256"])
    if args.input is not None:
        row = export(args.input,args.output,restore,segmenter,mean,std,card,branch)
        print(dict(written=list(row["output_SHA256"]),native=row["native_size"]),flush=True)
        return

    bind_rgb()
    mct,mct_info = load_author_reference()
    pgl,pgl_info = load_pgl()
    rows = []
    for dataset,n in DATA.items():
        count = 0
        for scene,sequence,rgb,label in load_pairs(dataset):
            if rgb.shape[:2]!=label.shape or not np.isin(np.unique(label),[*range(19),255]).all():
                raise ValueError("Invalid real fog segmentation label grid")
            image,_ = resize_image(rgb)
            truth = torch.from_numpy(label.astype(np.int64,copy=False))[None].cuda()
            expected = torch.bincount(truth.flatten(),minlength=256)[:19].cpu()

            def score(array,name,start):
                prediction = logits(segmenter,mean,std,array.clamp(0,1),label.shape)
                matrix = confusion_matrix(prediction.argmax(1),truth,19)
                if not torch.equal(matrix.sum(1),expected):
                    raise ValueError("A segmentation input changed scoreable labels")
                rows.append(measured_record(dataset,scene,sequence,name,start,0,matrix))

            score(image,"raw_dual_branch","fixed")
            score(author_rgb(mct,image),"author_mct_then_seg","fixed")
            score(pgl_rgb(pgl,image),"author_pgl_then_seg","fixed")
            for start,models in fitted.items():
                for method in V25:
                    score(predict(models[method],method,image),method,start)
            count += 1
            if count%10==0:
                print(dict(dataset=dataset,scored_pairs=count),flush=True)
        if count!=n:
            raise ValueError("Incomplete real road evaluation cohort")

    groups, summary = aggregate(rows)
    table = ROOT/"road_confusions.json"
    atomic_write(table,rows)
    example = V24/"native-dual-output/metadata.json"
    original = read(example)
    path = Path(original["input"])
    if digest(path)!=original["input_sha256"]:
        raise ValueError("Road dual-output reference input changed")
    out = ROOT/"native-dual-output"
    native = export(path,out,restore,segmenter,mean,std,card,branch)
    with Image.open(out/"mask.png") as current, Image.open(example.parent/"mask.png") as cached:
        same = bool(np.array_equal(np.asarray(current),np.asarray(cached)))
    if not same:
        raise ValueError("Raw segmentation must be invariant to the changed RGB branch")
    receipt = dict(data=DATA,mIoU=summary,confusion_rows=len(rows),confusion_SHA256=digest(table),
                   per_start=groups,source_SHA256={str(p):digest(p) for p in (Path(__file__),ZURICH,DRIVING)},
                   published={"mct":mct_info,"pgl":pgl_info},training_SHA256=digest(ROOT/"completed.json"),
                   original_pixels=native["native_size"],native_export_SHA256=digest(out/"metadata.json"),
                   raw_mask_equal_to_V24_MCT_system=same,scope="Two reused segmentation diagnostic cohorts. No road labels entered V25 fitting. Same frozen SegFormer for all inputs, ignore 255, native logit grid. The dual branch uses raw RGB independently of restoration.",
                   finished_utc=now())
    atomic_write(ROOT/"road_semantics.json",receipt)
    print(dict(counts=DATA,mIoU={k:round(v,5) for k,v in summary.items()},native_size=native["native_size"],
               mask_preserved=same),flush=True)


if __name__=="__main__":
    main()

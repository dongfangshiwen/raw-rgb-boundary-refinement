"""Matched-size external check of the intact Cityscapes Mask2Former-Swin-T release."""
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

from boundary_refinement.evaluation.semantic import idd_fog_evaluate as test
from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.data.cohorts.idd_fog_data import DATA,MAP,OUT,fixed,get
from boundary_refinement.evaluation.semantic.idd_fog_reference import iou_counts
from boundary_refinement.evaluation.semantic.idd_peer_rescore import compute_summary
from boundary_refinement.baselines.idd_peer_source import BASE,NAME,REV,SHA
from boundary_refinement.data.provenance import digest,read,write


@torch.inference_mode()
def run():
    fit.configure_cuda()
    for name in ("config.json","preprocessor_config.json"):
        p=BASE/name
        if not p.is_file():
            p.write_bytes(get(f"https://huggingface.co/{NAME}/resolve/{REV}/{name}"))
    cfg=read(BASE/"preprocessor_config.json")
    if digest(BASE/"model.safetensors")!=SHA:
        raise ValueError("Changed official segmenter weights")
    fixed(OUT/"mask2former-protocol.json",dict(code=digest(__file__),sha=SHA,revision=REV,configuration=digest(BASE/"config.json"),
        preprocessing=digest(BASE/"preprocessor_config.json"),evaluation=digest(OUT/"external-index.json"),
        method="Full, unmodified Cityscapes Mask2Former-Swin-T. Same raw 512-long-side RGB, pinned author mean/std, "
               "CUDA BF16, normalized zero padding to 32 with valid pixel mask, postprocess query masks on padded grid "
               "then crop and bilinear resize to native. Diagnostic after the primary endpoint. Segmentation-only model."))
    model=Mask2FormerForUniversalSegmentation.from_pretrained(str(BASE),use_safetensors=True,local_files_only=True).cuda().eval()
    if model.config.id2label[0]!="road" or model.config.id2label[18]!="bicycle":
        raise ValueError("Official class order changed")
    mean=torch.tensor(cfg["image_mean"],device="cuda")[None,:,None,None]
    std=torch.tensor(cfg["image_std"],device="cuda")[None,:,None,None]
    root=OUT/"mask2former-counts";root.mkdir(exist_ok=True)
    meta={r["id"]:r for r in read(DATA/"evaluation.json")["rows"]}
    rows=[]
    for i,index in enumerate(read(OUT/"external-index.json")["rows"]):
        row=meta[index["id"]];path=root/(row["id"].replace("/","_")+".npz")
        if not path.is_file():
            with Image.open(DATA/row["rgb"]) as im:
                rgb=np.array(im.convert("RGB"))
            x,(h,w)=fit.peer_input(rgb)
            x=(x[...,:h,:w]-mean)/std
            pad=(0,(-w)%32,0,(-h)%32)
            x=F.pad(x,pad)
            valid=F.pad(torch.ones((1,h,w),dtype=torch.long,device="cuda"),pad)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                raw=model(pixel_values=x,pixel_mask=valid)
                scores=fit.peer.query_to_scores(raw.class_queries_logits,raw.masks_queries_logits,x.shape[-2:],(h,w))
            truth=test.native_truth(row)
            ids=fit.j.resize(scores,truth.shape).argmax(1)[0]
            cm=np.stack(test.Counts(truth).score(ids))
            with path.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,counts=cm)
            path.with_suffix(".part").replace(path)
            ex=OUT/"examples"/row["id"].replace("/","_")
            if ex.exists():
                Image.fromarray(ids.cpu().byte().numpy()).save(ex/"author-mask2former512.png")
        with np.load(index["file"]) as f,np.load(path) as new:
            old=np.stack([f["full"],f["edge"]])
            methods=f["arms"].tolist()
            ours=old[:,methods.index("weighted512")]
            author=np.repeat(new["counts"][:,None],3,axis=1)
            if not np.array_equal(author.sum(-1),ours.sum(-1)):
                raise ValueError("Raw published model has different target support")
            values=np.stack((ours,old[:,methods.index("uniform512")],author),axis=1)
            rows.append(dict(group=index["group"],cohorts=index["cohorts"],v=iou_counts(values,sorted(MAP.values()))))
        if (i+1)%80==0:
            print("intact Mask2Former",i+1,flush=True)
    out={}
    for c in ("fog","mixed"):
        part=compute_summary([r for r in rows if c in r["cohorts"]])
        out[c]=dict(frames=part["images"],drives=part["drives"],metrics={k:dict(weighted512=v["weighted"],uniform512=v["initial_peer"],
                    mask2former512=v["corrected_peer"],difference=v["ours_minus_corrected"],ci95=v["paired_ci95"])
                    for k,v in part["scores"].items()})
    write(OUT/"mask2former-summary.json",out)
    print("intact Mask2Former complete",out["fog"],flush=True)


if __name__=="__main__":
    run()

"""Check the parallel predictor against frozen published upsamplers on equal Mask2Former logits."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from boundary_refinement.evaluation.semantic import idd_fog_evaluate as test
from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.evaluation.paired import muses_transmission as j
from boundary_refinement.models.operators import sapa as sapa
from boundary_refinement.data.cohorts.idd_fog_data import DATA,MAP,OUT,fixed
from boundary_refinement.evaluation.semantic.idd_fog_reference import iou_counts
from boundary_refinement.evaluation.semantic.idd_guide_transfer import D
from boundary_refinement.evaluation.semantic.idd_peer_rescore import compute_summary
from boundary_refinement.inference.parallel_fog import ParallelPredictor,export
from boundary_refinement.data.provenance import digest,read,write


@torch.inference_mode()
def run():
    fixed(D/"operator-protocol.json",dict(code=digest(__file__),prior=digest(D/"results.json"),
        cuda_sources=[digest(j.base.ROOT/"completed.json"),digest(sapa.ROOT/"completed.json")],
        design="Additional diagnostic on the 121 already scored midpoints. Same frozen 512-input Mask2Former logits "
               "and native truth. JBU Gaussian parameters fixed at V92 values, authors' DySample and SAPA-context "
               "trained on the same MUSES crops for 320 steps and transferred unchanged. SAPA uses both raw and restored "
               "RGB and its bilinear context. Compare the proposed raw-only uniform donor. No IDD tuning."))
    pred=ParallelPredictor();sampler=j.base.load();sapas=sapa.nets()
    rows=[];refs=read(D/"source.json")["rows"];arms=["our_raw","jbu","dysample","sapa_context"]
    for i,r in enumerate(refs):
        name=r["id"].replace("/","_")
        path=D/"operator-counts"/(name+".npz");path.parent.mkdir(exist_ok=True)
        if not path.exists():
            rgb=np.array(Image.open(DATA/r["rgb"]).convert("RGB"));truth=test.native_truth(r)
            metric=test.Counts(truth)
            z,guides,_,work=pred.features(rgb)
            stencil=j.base.guide_stencil(guides[:1],z.argmax(1)[0])
            delta=j.base.prediction("jbu",None,z,stencil)
            jbu=np.stack(metric.score(pred.native_mask(z,delta,work,truth.shape)))
            dp=[];sp=[]
            for seed in fit.SEEDS:
                delta=j.base.prediction("dysample",sampler["dysample",seed],z,None)
                dp.append(np.stack(metric.score(pred.native_mask(z,delta,work,truth.shape))))
                scores=sapa.run("sapa_context",sapas["sapa_context",seed],z,guides)
                delta=scores-F.interpolate(z,scale_factor=4.,mode="bilinear",align_corners=False)
                sp.append(np.stack(metric.score(pred.native_mask(z,delta,work,truth.shape))))
            with path.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,jbu=np.repeat(jbu[:,None],3,axis=1),dysample=np.stack(dp,axis=1),sapa_context=np.stack(sp,axis=1))
            path.with_suffix(".part").replace(path)
        with np.load(D/"counts"/(name+".npz")) as ours,np.load(path) as others:
            values=np.stack([ours["uniform_raw"],*(others[k] for k in arms[1:])],axis=1)
        if not np.all(values.sum(-1)==values.sum(-1)[:,:1,:1,:]):
            raise ValueError("Operators were evaluated on different native pixels")
        rows.append(dict(group=r["group"],v=iou_counts(values,sorted(MAP.values()))))
    sums={}
    for idx,a in enumerate(arms[1:],start=1):
        stat=compute_summary([{**r,"v":r["v"][:,[0,0,idx]]} for r in rows])
        sums[a]={k:dict(ours=x["weighted"],reference=x["corrected_peer"],delta=x["ours_minus_corrected"],ci95=x["paired_ci95"])
                 for k,x in stat["scores"].items()}
    # An independent native CLI output on the only RGB-screened fog drive among these 121 drives.
    record=next(x for x in refs if x["group"]=="train/135")
    show=OUT/"parallel-prediction"
    if not (show/"prediction.json").exists():
        export(DATA/record["rgb"],show,fit.SEEDS[0])
    original=read(show/"prediction.json")
    if original["input"]!=record["rgb_sha"] or any(digest(show/name)!=sha for name,sha in original["files"].items()):
        raise ValueError("Changed exported PNG")
    ids=np.array(Image.open(show/"mask_train_ids.png"))
    target=test.native_truth(record)
    check=np.stack(test.Counts(target).score(torch.from_numpy(ids).cuda()))
    with np.load(D/"counts"/(record["id"].replace("/","_")+".npz")) as f:
        if not np.array_equal(f["uniform_raw"][:,0],check):
            raise ValueError("Standalone predictor differs from held confusion matrices")
    rgb=np.array(Image.open(DATA/record["rgb"]).convert("RGB"))
    for crop in (rgb[:217,:257],rgb[:311,:157]):
        restored,labels=pred.predict(crop)
        if restored.shape!=crop.shape or labels.shape!=crop.shape[:2] or labels.max()>18:
            raise ValueError("Invalid partial-cell native output")
    write(D/"operator-comparisons.json",dict(frames=len(rows),seeds=list(fit.SEEDS),ours="Uniform raw-only, same 512-input Mask2Former",
         metrics=sums,exported_native_example=record["id"],checked_odd_sizes=[[217,257],[311,157]]))
    print("transferred operator comparisons",sums,flush=True)


if __name__=="__main__":
    run()

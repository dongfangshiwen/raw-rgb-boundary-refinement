"""Frozen-decoder transfer to an independent segmentation backbone on the screened fog drives."""
import argparse
import numpy as np
from PIL import Image
import torch
from transformers import Mask2FormerForUniversalSegmentation

from boundary_refinement.evaluation.paired import muses_transmission as j
from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.evaluation.semantic import idd_fog_evaluate as test
from boundary_refinement.data.cohorts.idd_fog_data import DATA,OUT,MAP,fixed
from boundary_refinement.baselines.idd_peer_source import BASE,SHA
from boundary_refinement.evaluation.semantic.idd_fog_reference import iou_counts
from boundary_refinement.evaluation.semantic.idd_peer_rescore import compute_summary
from boundary_refinement.data.provenance import digest,read,write


def plan():
    fixed(OUT/"decoder-transfer-protocol.json",dict(script=digest(__file__),segmenter=SHA,heads=digest(j.weighted.ROOT/"completed.json"),
        data=digest(DATA/"evaluation.json"),rgb=digest(j.calib.ROOT/"coefficients.json"),seeds=list(fit.SEEDS),
        purpose="Exploratory backbone transfer following the primary LiDSeg result. Fixed 512 input on all screened IDD fog "
                "frames; do not retrain, normalize temperatures, or choose methods on these images. Use original Mask2Former "
                "query logits and mask-grid sigmoid then query summation. Compare direct native interpolation of these 19 log "
                "scores, same MUSES-trained 9-scalar decoder with uniform vs semantic donor, and the complete authored Mask2Former "
                "upsample-query-mask-first output. Publish full and boundary mIoU with paired drive intervals. Only guide is the "
                "identical raw/restored calibrated RGB pair in both donor arms; no LiDSeg adapter enters the transferred logits."))


@torch.inference_mode()
def run():
    plan();fit.configure_cuda()
    model=Mask2FormerForUniversalSegmentation.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).cuda().eval()
    cfg=read(BASE/"preprocessor_config.json")
    mean,std=(torch.tensor(cfg[k],device="cuda")[None,:,None,None] for k in ("image_mean","image_std"))
    core=fit.FastJoint()
    dec=j.weighted.load()
    output=OUT/"backbone-transfer";output.mkdir(exist_ok=True)
    rows=[]
    for i,r in enumerate(row for row in read(DATA/"evaluation.json")["rows"] if "fog" in row["cohorts"]):
        path=output/(r["id"].replace("/","_")+".npz")
        if not path.exists():
            with Image.open(DATA/r["rgb"]) as im:
                rgb=np.array(im.convert("RGB"))
            t=test.native_truth(r);metric=test.Counts(t)
            x,size=fit.peer_input(rgb)
            x=x[...,:size[0],:size[1]]
            if any(n%32 for n in size):
                raise ValueError("Registered transfer dataset has changed geometry")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                outputs=model(pixel_values=(x-mean)/std,pixel_mask=torch.ones((1,*size),device="cuda",dtype=torch.long))
            mask_grid=outputs.masks_queries_logits.shape[-2:]
            if tuple(4*n for n in mask_grid)!=size:
                raise ValueError("Original query mask grid no longer covers the input")
            z=fit.peer.query_to_scores(outputs.class_queries_logits,outputs.masks_queries_logits,mask_grid,mask_grid)
            native=j.resize(z,t.shape)
            g=j.road.geometry(x,core.engine.core.rgb)
            u,a,w=j.calib.endpoints(g,core.engine.core.bank,core.heads,j.road.SEEDS[0])
            restored=j.calib.mixture(u,a,w,"curve",core.coeff)
            guides=j.base.render_guides(rgb,restored,mask_grid)
            wt=j.weighted.color_weights(z,"dual_core")
            template=j.base.guide_stencil(guides[:0],z.argmax(1)[0])
            coarse=np.stack(metric.score(native.argmax(1)[0]))
            res={"bilinear":np.repeat(coarse[:,None],3,axis=1)}
            for method,weight in (("uniform",torch.full_like(wt,1./16)),("weighted",wt)):
                stencil=j.weighted.prepare(guides,z,"dual_core",template,weight)
                seeds=[]
                for s in fit.SEEDS:
                    delta=j.base.prediction("dual_free",dec["dual_core",s],z,stencil)
                    seeds.append(np.stack(metric.score((native+j.resize(delta,t.shape)).argmax(1)[0])))
                res[method]=np.stack(seeds,axis=1)
            with path.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,**res)
            path.with_suffix(".part").replace(path)
        orig=OUT/"mask2former-counts"/path.name
        with np.load(orig) as extra,np.load(path) as m:
            auth=np.repeat(extra["counts"][:,None],3,axis=1)
            counts=np.stack([m["weighted"],m["uniform"],m["bilinear"],auth],axis=1)
        if not np.all(counts.sum(-1)==counts.sum(-1)[:,:1,:1,:]):
            raise ValueError("Different valid pixels for backbone transfer")
        rows.append(dict(group=r["group"],cohorts=r["cohorts"],v=iou_counts(counts,sorted(MAP.values()))))
        if (i+1)%60==0:
            print("decoder transfer",i+1,flush=True)
    results={}
    arm_names=["weighted","uniform","bilinear","author"]
    for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3)):
        reduced=[{**r,"v":r["v"][:,[a,a,b]]} for r in rows]
        s=compute_summary(reduced)
        results[arm_names[a]+"-"+arm_names[b]]={k:
            dict(first=v["weighted"],second=v["corrected_peer"],delta=v["ours_minus_corrected"],ci95=v["paired_ci95"])
            for k,v in s["scores"].items()}
    write(OUT/"decoder-transfer.json",dict(frames=len(rows),drives=len({r["group"] for r in rows}),results=results))
    print("transfer complete",results["weighted-uniform"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part",choices=("register","run"))
    if ap.parse_args().part=="run":
        run()
    else:
        plan()

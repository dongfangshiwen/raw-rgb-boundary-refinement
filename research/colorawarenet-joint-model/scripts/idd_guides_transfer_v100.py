"""First-use drive test of frozen color-guided upsampling on a second published backbone."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from PIL import Image
import torch
from transformers import Mask2FormerForUniversalSegmentation

from scripts import idd_fog_evaluate_v100 as test
from scripts import idd_fog_models_v100 as fit
from scripts import joint_tone_prototype_v99 as j
from scripts.idd_fog_data_v100 import DATA,MAP,OUT,REV,SOURCE,catalog,fetch,fixed,get
from scripts.idd_fog_reference_v100 import iou_counts
from scripts.idd_peer_rescore_v100 import compute_summary
from scripts.idd_peer_source_v100 import BASE,SHA
from scripts.projective_ledger_v12 import digest,read,write


D=OUT/"new-drive-guide-controls"
ARMS=("author","bilinear","uniform_dual","uniform_raw","uniform_tone","uniform_shift","weighted_dual")


def register():
    previous=read(DATA/"evaluation.json")["rows"]
    old_groups={r["group"] for r in previous}
    groups=catalog()["groups"]
    selection=[dict(group=g,**rows[len(rows)//2]) for g,rows in sorted(groups.items()) if g not in old_groups]
    fixed(D/"protocol.json",dict(code=digest(__file__),catalog=digest(DATA/"catalog.json"),prior=digest(DATA/"evaluation.json"),
        complete_previous_group_exclusion=sorted(old_groups),new_rows=selection,weather_lock=digest(OUT/"weather-lock.json"),
        model_sha=SHA,frozen_transport=digest(j.weighted.ROOT/"completed.json"),arms=list(ARMS),seeds=list(fit.SEEDS),
        primary="Every remaining IDD-AW author drive; one RGB midpoint/drive. RGB was seen in the original weather "
            "screen; no masks, model scores or predictions used for model choice in these drives. Mixed adverse weather. "
            "Fixed 512 long side. Transfer the frozen 9-parameter decoder to the native Mask2Former query score grid. "
            "Compare uniform raw+restored vs the published upsampling; for an independently beneficial restored spatial "
            "guide require boundary gain against identical raw+raw, moment-matched raw, and displaced RGB-residual controls. "
            "Same donor tap weights, coefficients and low-resolution logits across these guide arms. Include weighted vs "
            "uniform. Paired 5000-drive bootstrap, 17-class reference, no retuning. Full-image results must accompany edges. "
            "Three planned guide contrasts are descriptive pointwise 95% intervals; no inference across independent locations.",
        split="All held drives come from the original author's public training partition; our experiment uses them for testing only."))
    return selection


def stage():
    samples=register()
    def verify(r):
        folder=r["mask"].rsplit("/",1)[0]
        official=json.loads(get(f"https://huggingface.co/api/datasets/{SOURCE}/tree/{REV}/{folder}?limit=1000"))
        rec=next(v for v in official if v["path"]==r["mask"])
        p=fetch(r["mask"]); rgb=fetch(r["rgb"])
        if digest(p)!=rec["lfs"]["oid"]:
            raise ValueError("Remote native mask mismatch")
        rf=r["rgb"].rsplit("/",1)[0]
        arc=json.loads(get(f"https://huggingface.co/api/datasets/{SOURCE}/tree/{REV}/{rf}?limit=1000"))
        rgb_sha=digest(rgb)
        if next(v for v in arc if v["path"]==r["rgb"])["lfs"]["oid"]!=rgb_sha:
            raise ValueError("RGB checksum differs from the author release")
        with Image.open(rgb) as a,Image.open(p) as b:
            if a.size!=b.size or b.mode!="L":
                raise ValueError("Unpaired RGB/GT")
        return dict(**r,mask_sha=rec["lfs"]["oid"],rgb_sha=rgb_sha)
    loc=D/"source.json"
    if loc.exists():
        return read(loc)["rows"]
    with ThreadPoolExecutor(8) as p:
        rows=list(p.map(verify,samples))
    priors={r["rgb_sha"] for r in read(DATA/"evaluation.json")["rows"]}
    hashes={r["rgb_sha"] for r in rows}
    if len(hashes)!=len(rows) or priors&hashes:
        raise ValueError("Cross-cohort duplicate RGB")
    fixed(loc,dict(protocol=digest(D/"protocol.json"),rows=rows))
    print("verified first-use midpoint pairs",len(rows),flush=True)
    return rows


@torch.inference_mode()
def run():
    rows=stage()
    fit.configure_cuda()
    m=Mask2FormerForUniversalSegmentation.from_pretrained(str(BASE),local_files_only=True,use_safetensors=True).cuda().eval()
    config=read(BASE/"preprocessor_config.json")
    mean,std=(torch.tensor(config[k],device="cuda")[None,:,None,None] for k in ("image_mean","image_std"))
    core=fit.FastJoint()
    dec=j.weighted.load();packets=[]
    for i,row in enumerate(rows):
        loc=D/"counts"/(row["id"].replace("/","_")+".npz")
        loc.parent.mkdir(exist_ok=True)
        if not loc.is_file():
            with Image.open(DATA/row["rgb"]) as im:
                rgb=np.array(im.convert("RGB"))
            truth=test.native_truth(row);stats=test.Counts(truth)
            x,size=fit.peer_input(rgb);x=x[...,:size[0],:size[1]]
            if any(n%32 for n in size):
                raise ValueError("Invalid registered camera size")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                q=m(pixel_values=(x-mean)/std,pixel_mask=torch.ones((1,*size),device="cuda",dtype=torch.long))
            grid=q.masks_queries_logits.shape[-2:]
            def scores(shape):
                return fit.peer.query_to_scores(q.class_queries_logits,q.masks_queries_logits,shape,shape)
            z=scores(grid);native=j.resize(z,truth.shape)
            direct=j.resize(scores(size),truth.shape).argmax(1)[0]
            cm={}
            for name,ids in (("author",direct),("bilinear",native.argmax(1)[0])):
                cm[name]=np.repeat(np.stack(stats.score(ids))[:,None],3,axis=1)
            g=j.road.geometry(x,core.engine.core.rgb)
            u,a,w=j.calib.endpoints(g,core.engine.core.bank,core.heads,j.road.SEEDS[0])
            rgb_out=j.calib.mixture(u,a,w,"curve",core.coeff)
            guides=j.base.render_guides(rgb,rgb_out,grid)
            template=j.base.guide_stencil(guides[:0],z.argmax(1)[0])
            weighted=j.weighted.color_weights(z,"dual_core");uniform=torch.full_like(weighted,1./16)
            for arm,guide,weight in (("uniform_dual",1,uniform),("uniform_raw",0,uniform),("uniform_tone",2,uniform),
                                      ("uniform_shift",3,uniform),("weighted_dual",1,weighted)):
                stencil=j.weighted.prepare(guides[[0,guide]],z,"dual_core",template,weight)
                counts=[]
                for s in fit.SEEDS:
                    delta=j.base.prediction("dual_free",dec["dual_core",s],z,stencil)
                    counts.append(np.stack(stats.score((native+j.resize(delta,truth.shape)).argmax(1)[0])))
                cm[arm]=np.stack(counts,axis=1)
            with loc.with_suffix(".part").open("wb") as f:
                np.savez_compressed(f,**cm)
            loc.with_suffix(".part").replace(loc)
        with np.load(loc) as f:
            arr=np.stack([f[a] for a in ARMS],axis=1)
        if not np.all(arr.sum(-1)==arr.sum(-1)[:,:1,:1,:]):
            raise ValueError("Different evaluation support in the controls")
        packets.append(dict(group=row["group"],v=iou_counts(arr,sorted(MAP.values()))))
        if (i+1)%40==0:
            print("new guide-controls drives",i+1,flush=True)
    out={}
    for a,b in (("uniform_dual","author"),("uniform_dual","uniform_raw"),("uniform_dual","uniform_tone"),
                ("uniform_dual","uniform_shift"),("weighted_dual","uniform_dual"),("uniform_raw","author"),
                ("uniform_dual","bilinear"),("uniform_raw","bilinear")):
        ix=[ARMS.index(a),ARMS.index(a),ARMS.index(b)]
        d=compute_summary([{**r,"v":r["v"][:,ix]} for r in packets])
        out[a+"-"+b]={k:dict(a=v["weighted"],b=v["corrected_peer"],delta=v["ours_minus_corrected"],
                           ci95=v["paired_ci95"]) for k,v in d["scores"].items()}
    fog=set(read(OUT/"weather-lock.json")["confirmed"])
    write(D/"results.json",dict(drives=len(rows),previously_screened_as_fog=[r["group"] for r in rows if r["group"] in fog],
        source_split=dict(Counter(r["group"].split("/")[0] for r in rows)),effects=out,protocol=digest(D/"protocol.json")))
    print("first-use drive contrasts",len(rows),out,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("part",choices=("register","run"))
    if cli.parse_args().part=="run":
        run()
    else:
        register()

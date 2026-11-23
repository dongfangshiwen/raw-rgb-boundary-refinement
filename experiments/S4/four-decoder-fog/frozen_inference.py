"""Repeat four frozen decoders on the published IDD-AW fog cohort with the project CUDA environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True, help="Project root containing benchmark_data and experiment_exports")
parser.add_argument("--output", type=Path, default=HERE)
args = parser.parse_args()
project = args.project.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
runs = project / "experiment_exports"
history = runs / "kernel-attribution-2026-11-23"
sys.path.insert(0, str(history))
import audit
from scripts import eval_range_v101 as ev

ARMS = ["bilinear", "gaussian_jbu", "fixed_four", "source_fit"]
SEEDS = [101, 211, 307]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def record(path, obj):
    if path.exists():
        assert json.loads(path.read_text("utf8")) == obj, str(path)
    else:
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf8")
        temp.replace(path)

@torch.inference_mode()
def run():
    ev.setup.configure_cuda()
    assert torch.cuda.is_available()
    old_lock = ev.read(history / "protocol.json")
    ident = ev.read(runs / "decoder-resolution-2026-11-21" / "protocol.json")["frames"]
    original_rows = {r["id"]:r for r in ev.read(ev.IDD / "evaluation.json")["rows"] if "fog" in r["cohorts"]}
    rows = [original_rows[row["id"]] for row in ident]
    for row, archived in zip(rows,ident):
        assert all(row[k] == archived[k] for k in ["id","group","rgb_sha","mask_sha"])
    ids = [r["id"] for r in rows]
    groups = [r["group"] for r in rows]
    assert len(set(ids)) == len(rows) == 146 and len(set(groups)) == 24
    plan = {
        "role":"Retrospective matched comparison on 24 IDD-AW drives already used in development. No new test recordings or fitting.",
        "arms":ARMS,"seeds":SEEDS,"classes":audit.CLASSES,"rows":[{k:r[k] for k in ("id","group","rgb_sha","mask_sha")} for r in rows],
        "primary":"Source fit minus fixed Gaussian JBU on the native radius-six boundary for both target backbones. Report all signs and whole-image effects as well.",
        "operators":"Frozen identical class grid and raw RGB within each target. Bilinear uses the same coarse grid. JBU uses the fixed 3 x 3 Gaussian in Table 4 (spatial sigma 0.65 cells, RGB RMS sigma 0.05). Fixed four uses the original bounded basis initialization. The source fit is the original two slots with identical raw guides at transfer.",
        "bootstrap":"5000 paired multinomial drive resamples (seed 11221), identical across arms, two regions and both targets. Sum confusion counts before class averaging. The three source seeds are averaged, not counted as independent drives. All intervals are pointwise.",
        "checks":"Compare bilinear, fixed four and the three source-seed confusion matrices with the preceding run for every frame. All four methods must use the same labeled pixels.",
        "dataset_lock":digest(history/"protocol.json"),"backbone_sha256":old_lock["backbone_hashes"],
        "decoder_sha256":old_lock["checkpoint_hashes"],"code_sha256":digest(__file__),
        "source_code_sha256":{Path(f).name:digest(f) for f in (audit.__file__,ev.__file__,ev.legacy.__file__,sys.modules[ev.RangeDecoder.__module__].__file__)},
        "environment":{"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}
    }
    record(out/"protocol.json",plan)
    lock = digest(out/"protocol.json")
    fits=ev.original.load()
    default=ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    for target in ("mask2former","segformer"):
        with np.load(history/f"counts-{target}.npz",allow_pickle=False) as earlier:
            assert earlier["ids"].tolist()==ids and earlier["groups"].tolist()==groups and earlier["seeds"].tolist()==SEEDS
            curves=earlier["counts"][:,:,[1,2,6]]
        path=out/f"counts-{target}.npz"
        shape=(len(rows),2,len(ARMS),len(SEEDS),19,19)
        if path.exists():
            with np.load(path, allow_pickle=False) as c:
                assert c["lock"].item()==lock and c["ids"].tolist()==ids
                counts, start=c["counts"],int(c["complete"])
        else:
            counts, start=np.zeros(shape,np.int32),0
        if start==len(rows):
            continue
        model=ev.Target(target)
        for i in range(start,len(rows)):
            rgb,truth=ev.scene(rows[i],"idd")
            size=tuple(truth.shape)
            assert size==(1536,2048)
            metric=audit.ScoreFrame(truth)
            image=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
            image=ev.resize(image,(384,512),antialias=True).contiguous()
            z,released=model(image,size)
            del released
            guide=image.clamp(0,1).mul(255).round().byte()
            base=ev.resize(z,size)
            stencil=ev.legacy.guide_stencil(guide,z.argmax(1)[0])
            dual=ev.legacy.guide_stencil(guide[[0,0]],z.argmax(1)[0])
            def conf(delta):
                return metric.confusion(metric.labels(base+ev.resize(delta,size)))
            counts[i,:,0]=metric.confusion(metric.labels(base))[:,None]
            counts[i,:,1]=conf(ev.legacy.prediction("jbu",None,z,stencil))[:,None]
            counts[i,:,2]=conf(default.correction(z,ev.prepare(z,guide)))[:,None]
            for j,seed in enumerate(SEEDS):
                counts[i,:,3,j]=conf(ev.legacy.prediction("dual_free",fits["dual_core",seed],z,dual))
            np.testing.assert_array_equal(counts[i][:,[0,2,3]],curves[i],err_msg=target+" "+ids[i])
            assert np.all(counts[i].sum(-1)==counts[i,:,:1,:1].sum(-1))
            assert np.all(counts[i,1]<=counts[i,0])
            if i==start or (i+1)%24==0 or i+1==len(rows):
                temp=path.with_suffix(".part")
                with temp.open("wb") as stream:
                    np.savez_compressed(stream,counts=counts,ids=ids,groups=groups,seeds=SEEDS,
                        arms=ARMS,lock=lock,complete=i+1)
                temp.replace(path)
                print(f"{target}: {i+1}/{len(rows)} native masks verified",flush=True)
        del model
        torch.cuda.empty_cache()

if __name__=="__main__":
    run()

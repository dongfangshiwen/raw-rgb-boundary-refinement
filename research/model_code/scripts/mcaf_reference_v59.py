"""Short-budget RRSHID comparison using the published MCAF-Net architecture."""
import argparse
import ast
from time import perf_counter
from types import ModuleType

import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.decision_overlap_v55 import MANIFEST, rr_records
from scripts.overlap_quadratic_v53 import SEEDS
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


SOURCE = REPO.parent / "benchmark_data/mcaf-author-v59/MCAFNet.py"
ROOT = REPO / "results/mcaf-reference-v59"
REVISION = "fe1bd635d8e8413e79f0d0523cf77e7154561ad4"
HASH = "ff3f8fa102afc68803d1fe27e907eca56736cf83cb30eec5113f02aa1090ac67"
STEPS, CROP, BATCH = 600, 96, 2


def architecture():
    if digest(SOURCE) != HASH:
        raise ValueError("The published MCAF-Net source changed")
    graph = ast.parse(SOURCE.read_text("utf-8"), filename=str(SOURCE))
    if any(isinstance(node, ast.Name) and node.id=="BaseModule" for node in ast.walk(graph)):
        raise ValueError("Cannot omit the mmengine import when its base class is used")
    graph.body = [x for x in graph.body if not (isinstance(x, ast.ImportFrom) and x.module == "mmengine.model")]
    author = ModuleType("mcafnet_published")
    exec(compile(graph,str(SOURCE),"exec"), author.__dict__)
    return author.MCAFNet_l


def paired_crop(training, generator):
    batch = []
    for _ in range(BATCH):
        x,y = training[int(torch.randint(len(training),(),generator=generator))]
        h,w = x.shape[-2:]
        a,b = int(torch.randint(h-CROP+1,(),generator=generator)), int(torch.randint(w-CROP+1,(),generator=generator))
        p = torch.cat((x,y))[:,a:a+CROP,b:b+CROP]
        if torch.rand((),generator=generator) < .5:
            p = p.flip(-1)
        batch.append(p)
    return torch.stack(batch).cuda().float().div_(255.).split(3,dim=1)


def registration():
    return dict(code_SHA256=digest(__file__), manifest_SHA256=digest(MANIFEST), architecture_SHA256=HASH,
                author_revision=REVISION, author_paper="https://arxiv.org/html/2503.17966v2",
                steps=STEPS, batch=BATCH, crop=CROP, seeds=list(SEEDS),
                provenance="Use the published MCAFNet_l body with the unused mmengine import removed. "
                           "From scratch on the same 240 registered RRSHID pairs that were available to the correction heads. "
                           "Train all weights using RGB L1 on unit-scaled images and AdamW. This short run uses "
                           "a different loss, amount of training data and budget than the published 300-epoch recipe.")


def fit(smoke=False):
    configure_cuda()
    build = architecture()
    config = registration()
    lock = ROOT/"protocol.json"
    if not smoke:
        if lock.exists() and read(lock) != config:
            raise ValueError("The comparison fit has changed")
        if not lock.exists():
            atomic_write(lock,config)
    training = [(x,y) for _,_,_,x,y in rr_records("train")]
    if len(training) != 240:
        raise ValueError("The MCAF-Net RRSHID fit data changed")
    results = []
    for seed in SEEDS[:1] if smoke else SEEDS:
        path = ROOT/f"mcaf-{seed}.pth"
        note = path.with_suffix(".json")
        if not smoke and note.exists():
            out = read(note)
            if out["checkpoint_SHA256"] != digest(path) or out["protocol_SHA256"] != digest(lock):
                raise ValueError("The recorded MCAF-Net fit changed")
            results.append(out)
            continue
        set_seed(seed)
        rng = torch.Generator().manual_seed(seed+59057)
        net = build().cuda()
        optimizer = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        dec = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=.00003)
        start = perf_counter()
        for step in range(6 if smoke else STEPS):
            x,y = paired_crop(training,rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pred = net(x)[0]
            loss = F.l1_loss(pred.float(),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.,error_if_nonfinite=True)
            optimizer.step()
            dec.step()
            if (step+1)%200==0:
                print(dict(seed=seed,step=step+1,loss=float(loss.detach()),seconds=round(perf_counter()-start,1)),flush=True)
        torch.cuda.synchronize()
        report = dict(seed=seed,steps=step+1,loss=float(loss.detach()),parameters=sum(p.numel() for p in net.parameters()),
                      seconds=perf_counter()-start)
        if not smoke:
            torch_save(path,{k:v.detach().cpu() for k,v in net.state_dict().items()})
            report.update(checkpoint_SHA256=digest(path),protocol_SHA256=digest(lock))
            atomic_write(note,report)
            results.append(report)
        print(report,flush=True)
    if not smoke:
        atomic_write(ROOT/"completed.json",dict(runs=results,updates=sum(v["steps"] for v in results),
            protocol_SHA256=digest(lock),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),finished_utc=now()))


def trained():
    lock=ROOT/"protocol.json"
    card=read(ROOT/"completed.json")
    if read(lock)!=registration() or card["protocol_SHA256"]!=digest(lock):
        raise ValueError("MCAF-Net fit does not match the registered implementation")
    author=architecture()
    models={}
    for fit_run in card["runs"]:
        path=ROOT/f"mcaf-{fit_run['seed']}.pth"
        if digest(path)!=fit_run["checkpoint_SHA256"]:
            raise ValueError("A MCAF-Net checkpoint changed")
        network=author()
        network.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
        models[fit_run["seed"]]=network.cuda().eval().requires_grad_(False)
    if set(models)!=set(SEEDS):
        raise ValueError("Expected prior and same-budget MCAF seeds")
    return models


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("stage",choices=("fit","smoke"))
    fit(args.parse_args().stage=="smoke")

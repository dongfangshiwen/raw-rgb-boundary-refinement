"""Record the deployed seed's training objectives without changing any formal checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2] / "research" / "model_code"))
from dehaze_seg.utils import set_seed
from scripts import road_tone_control_v98 as rgb
from scripts import semantic_subpixel_v94 as sem
from scripts.train_regional_guidance_v18 import configure_cuda


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def compare(model, path):
    original = torch.load(path, map_location="cpu", weights_only=True)
    delta = max((p.detach().cpu() - original[k]).abs().max().item() for k, p in model.state_dict().items())
    return delta


def replay_rgb():
    seed = rgb.SEEDS[0]
    reference, _ = rgb.road.load_source()
    bank = rgb.road.load_trained("rgb")
    pool = rgb.train_inputs(rgb.road.kitti(), reference)
    with torch.no_grad():
        states = [(rgb.state(g, bank, seed), y) for g,y in pool]
    del pool, reference, bank
    out = {}
    for arm in ("affine", "luma"):
        with torch.no_grad():
            samples = [rgb.training_sample(rgb.geometry(q, arm), y) for q,y in states]
        set_seed(seed)
        model = rgb.ColorHead().cuda()
        objective = rgb.road.batch_loss(rgb.tone_loss, samples, "local")
        generator = torch.Generator().manual_seed(seed + 68027)
        probe = np.linspace(0,len(samples)-1,16,dtype=int)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,rgb.STEPS,eta_min=.00008)
        series, checks = [], []
        for iteration in range(rgb.STEPS + 1):
            if iteration % 10 == 0:
                with torch.no_grad():
                    value = torch.stack([rgb.tone_loss(model,samples[i],"local") for i in probe]).mean().item()
                checks.append([iteration,value])
            if iteration == rgb.STEPS:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model,generator)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.5,error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            series.append(loss.item())
        receipt = rgb.read(rgb.ROOT / f"{arm}-{seed}.json")
        out[arm] = dict(steps=rgb.STEPS,seed=seed,minibatch=series,probe=checks,probe_indices=probe.tolist(),
                        archived_last_loss=receipt["loss"],checkpoint_max_difference=compare(model,rgb.ROOT/f"{arm}-{seed}.pth"))
        print(arm, checks[0], checks[-1], out[arm]["checkpoint_max_difference"],flush=True)
    save("rgb-traces.json",out)


def replay_seg():
    seed, arm = sem.SEEDS[0], "dual_core"
    base = sem.parent
    frames = base.train_frames(base.DualScaleFeatures(),base.anchors(),False)
    set_seed(seed)
    model = base.ColorTransport().cuda()
    rng = np.random.default_rng(92001+seed)
    probe_rng = np.random.default_rng(991119)
    probes = [base.train_crop(frames,probe_rng,0,f,step) for step in range(8) for f in (False,True)]
    optimizer = torch.optim.AdamW(model.parameters(),lr=.035,weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,sem.STEPS,eta_min=.0035)
    series, checks = [], []
    for iteration in range(sem.STEPS+1):
        if iteration % 10 == 0:
            with torch.no_grad():
                checks.append([iteration,torch.stack([sem.loss(model,arm,x) for x in probes]).mean().item()])
        if iteration == sem.STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = sum(sem.loss(model,arm,base.train_crop(frames,rng,0,f,iteration)) for f in (False,True))/2.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        series.append(loss.item())
    receipt = sem.read(sem.ROOT/f"{arm}-{seed}.json")
    info = dict(seed=seed,steps=sem.STEPS,probe_seed=991119,probe_size=len(probes),minibatch=series,probe=checks,
                checkpoint_max_difference=compare(model,sem.ROOT/f"{arm}-{seed}.pth"),archived_last_loss=receipt["loss"])
    save("seg-trace.json",info)
    print(arm,checks[0],checks[-1],info["checkpoint_max_difference"],flush=True)


if __name__ == "__main__":
    configure_cuda()
    print(torch.__version__,torch.cuda.get_device_name(),flush=True)
    if not (OUT/"rgb-traces.json").exists():
        replay_rgb()
    torch.cuda.empty_cache()
    if not (OUT/"seg-trace.json").exists():
        replay_seg()

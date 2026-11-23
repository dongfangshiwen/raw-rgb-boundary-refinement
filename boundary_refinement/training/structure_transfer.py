"""Train single-input restoration with controlled same-scene structure supervision."""
import argparse
from collections import Counter
import time

import torch
import torch.nn.functional as F

from boundary_refinement.utils import set_seed
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/structure-transfer-v32"
ARMS = ("point", "uniform_pair", "local_pair", "local_reference")
MAIN = "local_pair"
SEEDS = (32003, 32017, 32029)
STEPS = 144
PAIRS = 3
SIDE = 160
PAIR_WEIGHT = .5


def detail(x):
    return x-F.avg_pool2d(x,5,1,2,count_include_pad=False)


def local_average(x):
    return F.avg_pool2d(x,9,1,4,count_include_pad=False)


def descriptor(x):
    return detail(x.mean(1,keepdim=True))


@torch.no_grad()
def reliability(light, dense, target):
    """Identify reference edges observed in light haze and attenuated in dense haze."""
    clear = descriptor(target)
    ref = local_average(clear.square())

    def visibility(rgb):
        signal = descriptor(rgb)
        covariance = local_average(clear*signal)
        ratio = (covariance/(ref+1e-5)).clamp(0,1)
        coherence = (covariance/(ref*local_average(signal.square())+1e-9).sqrt()).clamp(0,1)
        return ratio, coherence

    light_ratio, match = visibility(light)
    dense_ratio, _ = visibility(dense)
    strength = (ref.sqrt()/.035).clamp(0,1)
    return strength*match.square()*(light_ratio-dense_ratio).clamp_min(0)


@torch.no_grad()
def edge_visibility(frames, clear):
    grids = F.interpolate(torch.cat((clear[None],frames)).float()/255,
                          size=(192,256),mode="bilinear",align_corners=False,antialias=True)
    base, view = descriptor(grids).split((1,len(frames)))
    return (view*base).mean((1,2,3))/base.square().mean().clamp_min(1e-6)


def training_scenes():
    card = read(TRAIN_CARD)
    if card["scene_counts"] != {"fit":32,"calibrate":8}:
        raise ValueError("Registered LMHaze split changed")
    output = []
    for row in card["scenes"]:
        if row["role"] != "fit":
            continue
        if digest(row["file"]) != row["SHA256"]:
            raise ValueError("Registered training RGB changed")
        value = torch.load(row["file"],map_location="cpu",weights_only=True)
        clear = value["clean"].cuda()
        frames = torch.stack(value["frames"]).cuda()
        if len(frames) != 8 or clear.shape != frames.shape[1:]:
            raise ValueError("Expected eight aligned intensity samples per reference")
        visibility = edge_visibility(frames,clear)
        order = visibility.argsort(descending=True)
        output.append(dict(name=row["scene"],clear=clear,frames=frames[order],
                           visibility=visibility[order].cpu().tolist()))
    return output


def scene_stream(scenes, seed):
    rng = torch.Generator().manual_seed(seed+8900)
    schedule = torch.cat([torch.randperm(len(scenes), generator=rng)
                          for _ in range((STEPS*PAIRS+len(scenes)-1)//len(scenes))])[:STEPS*PAIRS]
    for step, indices in enumerate(schedule.split(PAIRS)):
        views = []
        for i in indices.tolist():
            row = scenes[i]
            a = int(torch.randint(4,(),generator=rng))
            b = 4+int(torch.randint(4,(),generator=rng))
            stack = torch.stack((row["frames"][a],row["frames"][b],row["clear"]))
            h,w = stack.shape[-2:]
            if step%3 == 0:
                rgb = F.interpolate(stack.float()/255,size=(SIDE,SIDE),mode="bilinear",
                                    align_corners=False,antialias=True)
            else:
                top = int(torch.randint(h-SIDE+1,(),generator=rng))
                left = int(torch.randint(w-SIDE+1,(),generator=rng))
                rgb = stack[...,top:top+SIDE,left:left+SIDE].float()/255
            if bool(torch.rand((),generator=rng)<.5):
                rgb = rgb.flip(-1)
            views.append(rgb)
        yield torch.stack(views).transpose(0,1).contiguous().unbind(0)


def objective(pred, target, weight, method):
    light,dense = pred.float().chunk(2)
    truth = torch.cat((target,target))
    rgb = ((pred.float()-truth).square()+1e-6).sqrt().mean()
    gt_edges = detail(target)
    light_edges,dense_edges = detail(torch.cat((light,dense))).chunk(2)
    per_view = ((light_edges-gt_edges).abs().mean()+(dense_edges-gt_edges).abs().mean())/2
    aux = torch.zeros_like(rgb)
    if method == "uniform_pair":
        aux = (dense_edges-light_edges.detach()).abs().mean()
    elif method in (MAIN,"local_reference"):
        target_edges = gt_edges if method == "local_reference" else light_edges.detach()
        error = (dense_edges-target_edges).abs().mean(1,keepdim=True)
        aux = ((weight*error).mean((1,2,3))/weight.mean((1,2,3)).clamp_min(.015)).mean()
    return rgb+.2*per_view+PAIR_WEIGHT*aux, (rgb.detach(),per_view.detach(),aux.detach())


def profile(scenes, model):
    light,dense,gt = next(scene_stream(scenes,SEEDS[0]))
    weight = reliability(light,dense,gt)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        pred = model(torch.cat((light,dense)))
    loss,_ = objective(pred,gt,weight,MAIN)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
    torch.cuda.synchronize()
    print(dict(smoke_loss=loss.item(), mean_weight=weight.mean().item(),
               supported_pixels=(weight>.05).float().mean().item(),
               peak_gib=torch.cuda.max_memory_allocated()/2**30),flush=True)


def fit_one(model, initial, scenes, seed, arm, protocol):
    path = ROOT/f"{arm}-{seed}.pth"
    ledger = ROOT/f"{arm}-{seed}.json"
    if ledger.exists():
        saved = read(ledger)
        if saved["checkpoint_SHA256"] != digest(path) or saved["protocol_SHA256"] != protocol:
            raise ValueError("Stored ablation does not match the current experiment")
        return saved
    set_seed(seed)
    model.load_state_dict(initial,strict=True)
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.00012,weight_decay=.01)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=.00002)
    history, weighted_mass = [], []
    started = time.perf_counter()
    for n,(light,dense,clean) in enumerate(scene_stream(scenes,seed),1):
        weight = reliability(light,dense,clean)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            predicted = model(torch.cat((light,dense)))
        loss, values = objective(predicted,clean,weight,arm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step()
        lr.step()
        history.append(torch.stack(values))
        weighted_mass.append(weight.mean())
        if n%48 == 0:
            print(dict(arm=arm,seed=seed,completed=n, loss=torch.stack(history[-48:]).mean(0).tolist(),
                       seconds=round(time.perf_counter()-started,1)),flush=True)
    if n != STEPS:
        raise ValueError("Matched optimization budget changed")
    torch_save(path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    saved = dict(arm=arm,seed=seed,updates=n,protocol_SHA256=protocol,checkpoint_SHA256=digest(path),
                 file=str(path),mean_mask=torch.stack(weighted_mass).mean().item(),
                 losses=dict(zip(("rgb","detail","auxiliary"),torch.stack(history[-48:]).mean(0).tolist())),
                 seconds=round(time.perf_counter()-started,2),finished_utc=now())
    atomic_write(ledger,saved)
    return saved


def completed():
    result = read(ROOT/"completed.json")
    if {(r["arm"],r["seed"]) for r in result["runs"]} != {(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("Incomplete restoration controls")
    stamp = digest(ROOT/"protocol.json")
    for visit in result["runs"]:
        if visit["protocol_SHA256"] != stamp or digest(visit["file"]) != visit["checkpoint_SHA256"]:
            raise ValueError("Restoration weights changed after freezing")
    return result


def load_fit(model, row):
    if digest(row["file"]) != row["checkpoint_SHA256"]:
        raise ValueError("Unrecognized restoration checkpoint")
    model.load_state_dict(torch.load(row["file"],map_location="cpu",weights_only=True),strict=True)
    return model.eval().requires_grad_(False)


def train(smoke=False):
    configure_cuda()
    ROOT.mkdir(parents=True,exist_ok=True)
    model,pgl = load_pgl()
    scenes = training_scenes()
    if smoke:
        profile(scenes,model.train())
        return
    design = dict(hypothesis="Reliable high-frequency transfer between real haze levels should improve dense single-image outputs beyond uniform matching and same-weight clean-reference reweighting.",
                  source_SHA256=digest(__file__),training_manifest_SHA256=digest(TRAIN_CARD),backbone=pgl,
                  fit_groups=len(scenes),arms=list(ARMS),main=MAIN,seeds=list(SEEDS),steps=STEPS,views_per_update=2*PAIRS,
                  crop_size=SIDE,full_view_frequency="Every third shared update uses a square downsample of the whole scene.",
                  input_pair="Rank eight frames per training scene by low-resolution signed edge visibility relative to the clean reference. Sample one from each half. Use identical crop and flip for both and their common target.",
                  controls="Same RGB and detail target every step. Uniform pair matches the light prediction everywhere. Reliable local pair matches the same prediction with a weight from reference edge strength, observable light-edge alignment and the light-to-dense visibility drop. The local reference control uses those same weights with direct clean detail supervision.",
                  opt="Full backbone trainable. BF16 CUDA forwards, FP32 objective, AdamW and cosine decay. All arms use the same seed-conditioned inputs and final step. No model from the test split participates in fitting.",
                  evaluation="Primary comparison of local vs point, uniform and local clean reference on held LMHaze scenes. Also report the lower original visibility half, clean-reference edge error, actual predicted RGB and independent benchmark transfer. Tests previously evaluated by V31 are not new blind datasets. Native dual output still runs the raw Cityscapes19 head.")
    protocol = ROOT/"protocol.json"
    if protocol.exists() and read(protocol) != design:
        raise ValueError("Restoration protocol changed after registration")
    if not protocol.exists():
        atomic_write(protocol,design)
    key = digest(protocol)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    runs = [fit_one(model,initial,scenes,seed,arm,key) for seed in SEEDS for arm in ARMS]
    result = dict(runs=runs,protocol_SHA256=key,updates=sum(r["updates"] for r in runs),
                  group_visibility={r["name"]:r["visibility"] for r in scenes}, gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__),finished_utc=now())
    atomic_write(ROOT/"completed.json",result)
    print(dict(fits=len(runs),updates=result["updates"]),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    train(parser.parse_args().smoke)

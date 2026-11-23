"""Compare native crops and coefficient upsampling on the real NH training split."""
import argparse
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from dehaze_seg.utils import set_seed
from scripts.appearance_refiner_v22 import appearance_loss
from scripts.evaluate_boundary_consistency_v23 import load_paper, training_lock
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import ROOT, RealPairs, image_tensor, letterbox_rgb, long_edge
from scripts.train_boundary_consistency_v23 import ROOT as V23
from scripts.train_real_transfer_v22 import author_rgb
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


EPOCHS = 20
BATCH = 4
NATIVE_LONG = 768
H, W = 192, 256
METHODS = ("whole_frame", "native_crop", "mixed_scale", "affine_field")


def field_rgb(model, image, factor=2):
    """Estimate the author's gain and RGB bias on a coarser grid."""
    original = 2*image-1
    if factor == 1:
        low = original
    else:
        size = tuple(max(16, dim//factor) for dim in image.shape[-2:])
        low = F.interpolate(original, size, mode="bilinear", align_corners=False, antialias=True)
    height, width = low.shape[-2:]
    fields = model.forward_features(model.check_image_size(low)).float()[..., :height, :width]
    expanded = F.interpolate(fields, original.shape[-2:], mode="bilinear", align_corners=False)
    gain, bias = expanded.split((1, 3), dim=1)
    return .5*(gain*original-bias+original+1)


def train_pairs():
    output = []
    with RealPairs(("nh_train",)) as registry:
        ids = registry.ids("nh_train")
        if ids != [f"{i:02d}" for i in range(1, 46)]:
            raise ValueError("Only original NH-HAZE training pairs can enter this fit")
        for scene in ids:
            hazy, clear = registry.read("nh_train", scene)
            x, y, (x1, y1, x2, y2) = letterbox_rgb(hazy, clear)
            valid = torch.zeros((H, W), device="cuda", dtype=torch.bool)
            valid[y1:y2, x1:x2] = True
            local = tuple(image_tensor(long_edge(p, NATIVE_LONG)).cuda() for p in (hazy, clear))
            if min(t.shape[-2] for t in local) < H or min(t.shape[-1] for t in local) < W:
                raise ValueError("Native pair needs more crop support")
            output.append(((x.cuda(), y.cuda(), valid), local))
    return output


def select_batch(scenes, ids, generator):
    full, crop, mixed = [], [], []
    valid = torch.ones((H, W), device="cuda", dtype=torch.bool)
    for k, index in enumerate(ids):
        small, (hazy, clear) = scenes[index]
        h, w = hazy.shape[-2:]
        top = int(torch.randint(h-H+1, (), generator=generator))
        left = int(torch.randint(w-W+1, (), generator=generator))
        frame = (hazy[..., top:top+H, left:left+W], clear[..., top:top+H, left:left+W], valid)
        do_flip = bool(torch.rand((), generator=generator) < .5)
        if do_flip:
            small = tuple(t.flip(-1) for t in small)
            frame = tuple(t.flip(-1) for t in frame)
        full.append(small)
        crop.append(frame)
        mixed.append(frame if k % 2 else small)

    def pack(items):
        image, reference, mask = (torch.stack(v) for v in zip(*items))
        return image.float()/255, reference.float()/255, mask

    joined = pack(mixed)
    return dict(whole_frame=pack(full), native_crop=pack(crop), mixed_scale=joined, affine_field=joined)


def starts():
    prior, _, paper = training_lock()
    finals = {r["video"]: r for r in paper["runs"] if r["stage"] == "consistency"}
    return [(entry, finals[entry["video"]]) for entry in prior["author"]]


@torch.no_grad()
def start_model(original, latest):
    return load_paper(original, latest)["dehazeformer_t_continued"]


def register(pairs):
    kept = [REPO / "scripts" / p for p in ("train_real_detail_v24.py", "real_detail_data_v24.py",
                                           "evaluate_boundary_consistency_v23.py", "appearance_refiner_v22.py")]
    kept.extend((V23 / "author_completed.json", V23 / "plan.json"))
    plan = dict(starts=[list(p) for p in pairs], methods=list(METHODS), scene_ids=[f"{i:02d}" for i in range(1, 46)],
                epochs=EPOCHS, batch=BATCH, grid=[W,H], native_crop_from_long_edge=NATIVE_LONG,
                training="All four arms reuse each of 45 real NH-HAZE pairs once per epoch. Frames and crops are drawn with identical scene order and flips. Mixed exposes paired 256×192 full-frame and native-res crop views within each batch. The affine field arm takes the same mixed inputs and upsamples the half-resolution scalar gain and RGB bias before applying them to original pixels.",
                label_scope="NH-HAZE 01–45 paired RGB only. No semantic labels or older evaluation pairs.",
                architecture="Author DehazeFormer-T backbone from the three final V23 comparison fits. Same weights and parameter count across four variants. We test alternate training views and coordinate of applying its own affine output.",
                loss="V22 appearance_loss. AdamW initial LR 0.00025, decay .01, cosine to 0.000025, clip 1.0. CUDA BF16 training, FP32 evaluation.",
                assessment="All four final outputs to be compared on fixed previously seen 140 RGB pairs, at the common long-edge 1024 target. Direct authored MCT is a separate public reference with different and incompletely known pretraining.",
                source_SHA256={str(p): digest(p) for p in kept})
    path = ROOT / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("V24 registered training plan changed")
    if not path.exists():
        atomic_write(path, plan)
    return digest(path)


def update(network, optimizer, method, batch):
    image, target, mask = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted = field_rgb(network, image) if method == "affine_field" else author_rgb(network, image)
        value = appearance_loss(predicted, target, mask)
    value.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return value.detach()


def opt(model):
    return torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=.01)


def preflight(pairs, scene_data):
    example = select_batch(scene_data, list(range(BATCH)), torch.Generator().manual_seed(240000))
    model = start_model(*pairs[0])
    with torch.no_grad():
        image = example["whole_frame"][0]
        error = float((author_rgb(model, image)-field_rgb(model, image, 1)).abs().max())
        if error > 2e-6:
            raise ValueError("Affine formula differs from author's native route")
        if not torch.equal(example["mixed_scale"][0], example["affine_field"][0]):
            raise ValueError("Architectural arms have different training inputs")
    results = {method: float(update(model, opt(model), method, example[method])) for method in METHODS}
    if not np.isfinite(list(results.values())).all():
        raise ValueError("Nonfinite V24 CUDA pilot")
    row = dict(passed=True, check_author_error=error, discarded_updates=len(METHODS), loss=results,
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), finished_utc=now())
    atomic_write(ROOT / "smoke.json", row)
    print(row, flush=True)


def fit(initial, final, samples, plan):
    video = initial["video"]
    folder = ROOT / video
    folder.mkdir(exist_ok=True)
    path, artifact = folder / "training.json", folder / "trained.pth"
    if path.exists():
        record = read(path)
        if record["plan_SHA256"] != plan or digest(artifact) != record["checkpoint_SHA256"]:
            raise ValueError("V24 completed fit changed")
        return record
    seed = 24_000_000 + sum(video.encode("utf-8"))
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    models = {name: start_model(initial, final) for name in METHODS}
    optimizers = {k: opt(m) for k,m in models.items()}
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, EPOCHS, eta_min=2.5e-5) for o in optimizers.values()]
    torch.cuda.reset_peak_memory_stats()
    started, curves, updates = time.perf_counter(), [], 0
    for epoch in range(1,EPOCHS+1):
        numbers = []
        for scene in torch.randperm(len(samples), generator=generator).split(BATCH):
            batch = select_batch(samples, scene.tolist(), generator)
            values = [update(network, optimizers[name], name, batch[name]) for name, network in models.items()]
            numbers.append(torch.stack(values).tolist())
            updates += 1
        for schedule in schedulers:
            schedule.step()
        row = dict(video=video, epoch=epoch, steps_per_method=updates, losses=dict(zip(METHODS, np.mean(numbers, axis=0).tolist())),
                   peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,3), seconds=round(time.perf_counter()-started,2), updated_utc=now())
        curves.append(row)
        atomic_write(ROOT / "progress.json", row)
        print(row, flush=True)
    if updates != EPOCHS*((len(samples)+BATCH-1)//BATCH):
        raise ValueError("Inconsistent update budget")
    torch_save(artifact, dict(format="real-detail-v24", video=video, initialized_from=final["checkpoint_SHA256"],
                              models={name: model.state_dict() for name, model in models.items()}, plan_SHA256=plan))
    atomic_write(folder/"curves.json", curves)
    record = dict(video=video, checkpoint=str(artifact), checkpoint_SHA256=digest(artifact), steps_per_method=updates,
                  methods=list(METHODS), plan_SHA256=plan, finished_utc=now())
    atomic_write(path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    ROOT.mkdir(parents=True, exist_ok=True)
    pairs = starts()
    sha = register(pairs)
    samples = train_pairs()
    if args.smoke:
        preflight(pairs, samples)
        return
    if not read(ROOT/"smoke.json")["passed"]:
        raise ValueError("Need passed V24 CUDA preflight")
    complete = []
    for original, final in pairs:
        complete.append(fit(original, final, samples, sha))
        gc.collect()
        torch.cuda.empty_cache()
    summary = dict(all_complete=True, runs=complete, plan_SHA256=sha, fits=len(complete)*len(METHODS),
                   updates=sum(r["steps_per_method"] for r in complete)*len(METHODS), finished_utc=now())
    atomic_write(ROOT/"completed.json", summary)
    print({k:v for k,v in summary.items() if k!="runs"},flush=True)


if __name__ == "__main__":
    main()

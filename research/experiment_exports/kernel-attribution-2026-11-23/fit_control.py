"""Fit the range and spatial terms in isolation on the manuscript's frozen source scores."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "colorawarenet-joint-model"))
from scripts import fit_range_v101 as archived
from scripts.range_decoder_v101 import RangeDecoder, prepare
from scripts.restoration_affinity_v75 import boundary_band
from scripts import subpixel_transport_v92 as source
from audit import SEEDS, write_json

TRAINABLE = {"colour_only": ("mix",), "spatial_only": ("spatial",), "joint": ("mix", "spatial")}
STEPS = 320


def register():
    plan = {
        "role": "Retrospective targeted follow-up after frozen-factor substitution on the previously scored 146 IDD-AW fog frames. No independent confirmation.",
        "contrast": "Frozen four-range kernel versus training only its four colour amplitudes, only its one geometric exponent, or both. Identical raw RGB, coarse class transitions, donor grid, and score interface.",
        "source": "96 dense fog frames from 9 MUSES training recordings. Use original frozen source score crops, same two crops per step, weighted CE, seed-specific scores and crop stream as the archived six-family fit. Zero target training labels.",
        "stages": {"seeds": list(SEEDS), "updates_per_fit": STEPS, "crops_per_update": 2, "AdamW_lr": 0.035,
                   "weight_decay": 0.0, "cosine_end_lr": 0.0035, "gradient_clip": 5, "class": "bilinear_bounded4"},
        "trainable": TRAINABLE,
        "evaluation": "Apply each fit without target adaptation at input long edge 512 to the same 146 frames and 24 recording IDs using both Mask2Former and SegFormer-B1. Use the final update from all seeds. Primary descriptive contrast is colour-only refit minus fixed for each backbone on the native six-pixel boundary. Secondary contrasts cover geometry-only and joint, plus whole-image and outside-band mIoU. Same 5000 paired recording-bootstrap draws, common 17 classes, same 24 drives and inputs as the immediately preceding substitution experiment. No minimum effect declared for confirmation here, no tuning and no exclusion of negative arms.",
        "previous_inference_protocol": source.digest(HERE / "protocol.json"),
        "source_fit_manifest": source.digest(source.DATA / "fit.json"),
        "anchor": source.digest(source.ANCHOR / "completed.json"),
        "code": {Path(p).name: source.digest(p) for p in (__file__, archived.__file__, source.__file__,
                          sys.modules[RangeDecoder.__module__].__file__, sys.modules[boundary_band.__module__].__file__)},
    }
    path = HERE / "refit-protocol.json"
    if path.exists():
        import json
        assert json.loads(path.read_text("utf8")) == json.loads(json.dumps(plan))
    else:
        write_json(path, plan)
    return source.digest(path)


def make_nets():
    nets = {}
    for name, allowed in TRAINABLE.items():
        net = RangeDecoder("bilinear_bounded4").cuda()
        for param_name, p in net.named_parameters():
            p.requires_grad_(param_name in allowed)
        nets[name] = net
    return nets


def crop_items(pool, generator, trial, step):
    items = []
    for focused in (False, True):
        scores, guides, truth = source.train_crop(pool, generator, trial, focused, step)
        stencil = prepare(scores, guides[:1])
        base = F.interpolate(scores, scale_factor=4, mode="bilinear", align_corners=False)
        target = truth[8:-8, 8:-8]
        edge = boundary_band(truth)[8:-8, 8:-8]
        weight = (target < 19).float() * (1 + 3*edge + 2*((target==6) | (target==7) | (target==14)))
        items.append((scores, stencil, base, target, weight))
    return items


def objective(net, items):
    total = items[0][0].new_zeros(())
    for scores, stencil, base, target, weight in items:
        prediction = (base + net.correction(scores, stencil))[..., 8:-8, 8:-8]
        loss = F.cross_entropy(prediction, target[None], ignore_index=255, reduction="none")[0]
        total = total + .5 * (loss*weight).sum()/weight.sum().clamp_min(1)
    return total


def main():
    source.configure_cuda()
    protocol = register()
    reference = source.read(archived.OUT / "training.json")
    archived_fits = {x["seed"]: x for x in reference["runs"] if x["arm"]=="bilinear_bounded4"}
    pool = source.train_frames(source.DualScaleFeatures(), source.anchors(), False)
    assert (sum(map(len, pool)), len(pool)) == (96, 9)
    clock = monotonic()
    for trial, seed in enumerate(SEEDS):
        dest = HERE / f"source-fit-{seed}.json"
        if dest.exists():
            assert source.read(dest)["protocol"] == protocol
            continue
        source.set_seed(seed)
        rng = np.random.default_rng(92001+seed)
        nets = make_nets()
        opt = {key: torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.035, weight_decay=0.)
               for key, net in nets.items()}
        sched = {key: torch.optim.lr_scheduler.CosineAnnealingLR(o, STEPS, eta_min=.0035) for key, o in opt.items()}
        curves = {key: [] for key in nets}
        for step in range(STEPS):
            items = crop_items(pool, rng, trial, step)
            for key, net in nets.items():
                opt[key].zero_grad(set_to_none=True)
                loss = objective(net, items)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
                opt[key].step()
                sched[key].step()
                curves[key].append(loss.detach())
        sampler_hash = sha256(str(rng.bit_generator.state).encode()).hexdigest()
        assert sampler_hash == archived_fits[seed]["sampler_end"]
        assert nets["colour_only"].spatial.item() == 0.0
        assert torch.all(nets["spatial_only"].mix == -.5).item()
        check = torch.load(archived.OUT / archived_fits[seed]["weights"], map_location="cpu", weights_only=True)
        deviations = {k: (v.cpu()-check[k]).abs().max().item() for k,v in nets["joint"].state_dict().items()}
        arms = {key: {"weights": {k:v.tolist() for k,v in net.state_dict().items()},
                      "loss": torch.stack(curves[key]).cpu().tolist(),
                      "trainable_scalars": sum(p.numel() for p in net.parameters() if p.requires_grad)}
                for key,net in nets.items()}
        write_json(dest, {"protocol":protocol, "seed":seed, "updates_per_arm":STEPS, "sampler_sha256":sampler_hash,
                          "joint_max_absolute_weight_difference_from_archive":deviations,
                          "source_gpu":torch.cuda.get_device_name(), "torch":str(torch.__version__), "arms":arms})
        print(f"Seed {seed}, three controlled fits complete, {monotonic()-clock:.1f}s.",flush=True)


if __name__ == "__main__":
    main()

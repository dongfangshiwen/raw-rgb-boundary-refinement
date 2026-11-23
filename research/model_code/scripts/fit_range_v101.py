"""Fit all six models on exactly the same MUSES source crops and frozen logits."""
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.nn.functional as F

from scripts import subpixel_transport_v92 as old
from scripts.restoration_affinity_v75 import boundary_band
from scripts.range_decoder_v101 import FAMILIES, RangeDecoder, prepare

OUT = old.REPO.parent / "experiment_exports/raw-guidance-validation-2026-11-20"
PLAN = OUT / "protocol.json"


def fit():
    old.configure_cuda()
    assert list(FAMILIES) == old.read(PLAN)["families"]
    assert tuple(old.SEEDS) == (101,211,307)
    protocol_hash = old.digest(PLAN)
    start = monotonic()
    source = {"protocol": protocol_hash, "fit": old.digest(old.DATA/"fit.json"),
              "anchor": old.digest(old.ANCHOR/"completed.json"), "trainer":old.digest(__file__),
              "model": old.digest(Path(__file__).with_name("range_decoder_v101.py"))}
    receipts = []
    pool = old.train_frames(old.DualScaleFeatures(), old.anchors(), False)
    for trial, seed in enumerate(old.SEEDS):
        old.set_seed(seed)
        rng = np.random.default_rng(92001 + seed)
        nets = {a: RangeDecoder(a).cuda() for a in FAMILIES}
        opts = {a:torch.optim.AdamW(net.parameters(), lr=.035, weight_decay=0.) for a,net in nets.items()}
        scheds = {a:torch.optim.lr_scheduler.CosineAnnealingLR(o,320,eta_min=.0035) for a,o in opts.items()}
        losses = {a:[] for a in FAMILIES}
        for step in range(320):
            items = []
            for focused in (False, True):
                z, guides, truth = old.train_crop(pool,rng,trial,focused,step)
                st = prepare(z,guides[:1])
                base = F.interpolate(z,scale_factor=4,mode="bilinear",align_corners=False)
                edge = boundary_band(truth)[8:-8,8:-8]
                t = truth[8:-8,8:-8]
                wt = (t<19).float()*(1+3*edge+2*((t==6)|(t==7)|(t==14)))
                items.append((z, st, base, t, wt))
            for arm, net in nets.items():
                opts[arm].zero_grad(set_to_none=True)
                value = z.new_zeros(())
                for logits, st, base, target, weight in items:
                    pred = (base + net.correction(logits,st))[...,8:-8,8:-8]
                    ce = F.cross_entropy(pred,target[None],ignore_index=255,reduction="none")[0]
                    value = value + .5*(ce*weight).sum()/weight.sum().clamp_min(1)
                value.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
                opts[arm].step()
                scheds[arm].step()
                losses[arm].append(value.detach())
            if (step+1)%80 == 0:
                print(f"seed {seed}, {step+1}/320, six matched fits",flush=True)
        for arm,net in nets.items():
            path=OUT/f"{arm}-{seed}.pth"
            old.torch_save(path,{k:v.detach().cpu() for k,v in net.state_dict().items()})
            receipts.append(dict(arm=arm,seed=seed,weights=path.name,sha256=old.digest(path),
                                 steps=320,parameters=sum(p.numel() for p in net.parameters()),
                                 values={k:v.tolist() for k,v in net.state_dict().items()},
                                 losses=torch.stack(losses[arm]).cpu().tolist(),
                                 sampler_end=sha256(str(rng.bit_generator.state).encode()).hexdigest()))
        old.atomic_write(OUT/f"fits-{seed}.json", [r for r in receipts if r["seed"]==seed])
    old.atomic_write(OUT/"training.json",dict(inputs_sha256=source,runs=receipts,
                     source_frames=sum(map(len,pool)),source_groups=len(pool),updates=320*len(receipts),
                     torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                     seconds=round(monotonic()-start,2),finished_utc=old.now()))


if __name__ == "__main__":
    fit()

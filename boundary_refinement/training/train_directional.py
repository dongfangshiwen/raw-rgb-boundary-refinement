"""Fit three matched move rules on fixed MUSES source recordings and seal the weights."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
import torch.nn.functional as F
from boundary_refinement.models.operators import directional_veto as route
from boundary_refinement.evaluation.semantic import eval_native_guide as base
from boundary_refinement.evaluation.controls.audit_dual_output import AUTHOR_HASH, author_loader
from boundary_refinement.evaluation.semantic.real_fog_correction import DATA, paths
from boundary_refinement.data.provenance import digest, now
from boundary_refinement.training.train_regional_guidance import configure_cuda, torch_save
from boundary_refinement.data.staging.weatherproof_files import OUT, write


@torch.inference_mode()
def collect():
    author, model, source = author_loader()
    model.eval()
    mean, std = (torch.tensor(v, device="cuda")[None, :, None, None] for v in (author.MEAN, author.STD))
    segfix = base.SegFix()
    old = base.original.load()
    bank = {s: defaultdict(list) for s in route.SEEDS}
    history = []
    for key, group, _, _, truth, rgb in paths("fit"):
        raw = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float()/255
        padded, (h, w) = base.peer_input(rgb)
        small = padded[..., :h, :w].contiguous()
        z = model.predict((small-mean)/std).float()
        z = F.interpolate(z, (h//4, w//4), mode="bilinear", align_corners=False)
        native = base.resize(z, truth.shape)
        stencil = route.paper_stencil(z, small)
        lookup, offset = segfix.lookup(rgb, True)
        colors = route.DirectionalColors(raw)
        rec = dict(id=key, group=group, seeds={})
        for seed in route.SEEDS:
            frames = bank[seed][group]
            scores = route.paper_scores(z, stencil, old["dual_core", seed], native)
            labels, moved, ids, x = route.conflicts(scores, lookup, offset, colors)
            target = truth.flatten()[ids]
            yes, no = moved.flatten()[ids] == target, labels.flatten()[ids] == target
            index = (yes | no).nonzero().flatten()
            rec["seeds"][seed] = dict(proposals=len(ids), informative=len(index), accept=int(yes.sum()))
            if len(index):
                gen = torch.Generator(device="cuda").manual_seed(int.from_bytes(sha256((key+str(seed)).encode()).digest()[:4],"big"))
                index = index[torch.randperm(len(index), generator=gen, device="cuda")[:6144]]
                frames.append(torch.cat((x[index], yes[index, None]), -1).cpu().numpy())
            del scores
        history.append(rec)
        if len(history)%16==0:
            print("MUSES source frames",len(history),flush=True)
    assert len(history)==96 and all(len(g)==9 and any(g.values()) for g in bank.values())
    print("Informative frames by source group", {k:len(v) for k,v in bank[route.SEEDS[0]].items()},flush=True)
    return bank, history, source


def run():
    configure_cuda()
    start = monotonic()
    bank, history, source = collect()
    records = []
    for seed in route.SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 10300)
        generator = torch.Generator(device="cuda").manual_seed(seed + 10300)
        groups = [[torch.from_numpy(v).cuda() for v in frames] for _, frames in sorted(bank[seed].items())]
        models = [route.MoveDecision(mode).cuda() for mode in route.MODES]
        opt = torch.optim.Adam([p for n in models for p in n.parameters()], lr=.035)
        traces = [[] for _ in models]
        for step in range(300):
            samples = []
            # Sample uniformly over recordings conditional on at least one informative move.
            while len(samples) < 4:
                group = groups[rng.integers(len(groups))]
                if not group:
                    continue
                frame = group[rng.integers(len(group))]
                samples.append(frame[torch.randint(len(frame),(512,),generator=generator,device="cuda")])
            x = torch.cat(samples)
            opt.zero_grad(set_to_none=True)
            losses = [F.binary_cross_entropy_with_logits(m(x[:, :6]), x[:,6]) for m in models]
            sum(losses).backward()
            opt.step()
            for l, trace in zip(losses, traces):
                trace.append(float(l.detach()))
        for m, trace in zip(models,traces):
            path = OUT/f"{m.mode}-{seed}.pth"
            state = {k:v.detach().cpu() for k,v in m.state_dict().items()}
            torch_save(path,state)
            records.append(dict(mode=m.mode,seed=seed,steps=300,weights=path.name,sha256=digest(path),loss=trace,
                                coefficients=m.linear.weight.detach().flatten().tolist(),bias=m.linear.bias.item()))
        print("Sealed three fits", seed,flush=True)
    write(OUT/"training.json",dict(records=records,frames=history,manifest=digest(DATA/"fit.json"),source_code=source,
          lidseg=AUTHOR_HASH,segfix=base.SHA,protocol=digest(OUT/"protocol.json"),trainer=digest(__file__),
          implementation=digest(route.__file__),cuda=torch.version.cuda,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
          seconds=monotonic()-start,finished=now()))


def load():
    from boundary_refinement.data.staging.weatherproof_files import read
    trained = read(OUT/"training.json")
    assert trained["implementation"]==digest(route.__file__) and trained["protocol"]==digest(OUT/"protocol.json")
    nets={}
    for r in trained["records"]:
        path=OUT/r["weights"]
        assert digest(path)==r["sha256"] and r["steps"]==300
        net=route.MoveDecision(r["mode"])
        net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
        nets[r["mode"],r["seed"]]=net.cuda().eval().requires_grad_(False)
    assert len(nets)==9
    return nets


if __name__=="__main__":
    run()

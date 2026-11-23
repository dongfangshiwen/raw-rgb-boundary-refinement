"""Matched two-stage FeatUp control on raw semantic scores, trained only on MUSES."""
from boundary_refinement.data.source_integrity import matches_registered
import argparse
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from boundary_refinement.evaluation.controls import audit_fixed_ranges as paired
from boundary_refinement.training import fit_range_kernel as training

ext, io = paired.ext, paired.score
old = training.old
OUT = ext.WORK / "experiment_exports/featup-control-v107"
WEIGHTS = old.REPO / "pretrained/featup-dino16-v107"
REV = "6b5a6c0e91f75e69194807128dcbc39c3084a30d"
FILES = {"upsamplers.py": "e213e36c1f592c30aebbc527954539c006342f5d1a7fa01e382f5166c6cb0004",
         "dino16.ckpt": "407dff491058280abfbf38647e9365bf6397bc53d03b13251ed5a6715ef8d1b2",
         "LICENSE": "322bd9cc6ef4960a4caf228200b4ee8bf6e816cb9e1162dabba0ba0d72c89c2a"}
ARMS = ("released", "fixed4", "paper", "featup_frozen", "featup_fit", "featup_fit_gated")
PAIRS = {"paper": ("featup_frozen", "featup_fit", "featup_fit_gated", "fixed4", "released"),
         "featup_fit": ("released", "featup_frozen"), "featup_fit_gated": ("featup_fit", "released")}
PLAN = dict(status="Post-development comparison on previously examined targets. No independent holdout claim.",
    source="FeatUp JBUStack DINO16 COCO-Stuff from the authors, revision "+REV+
        ". Load both 2x learned bilateral stages up1 and up2 strictly. Keep their internal range and fixup projections. Omit stages 3/4 and final feature-specific channel head to preserve fourfold output and label channels. This is an adapted operator.",
    files=FILES, checkpoint_url="https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt",
    training="Same 96 MUSES images, nine recordings, tensors, source seeds, paired crops, weighted pixel cross-entropy, 8px trim, 320 updates and crop order as Appendix A.1. ImageNet-normalized raw guide. Fine-tune two pretrained blocks with AdamW lr 0.0003, cosine to 0.00003, weight decay 0 and clip 5. Keep the last step; also report the unchanged author initialization. No target tuning.",
    evaluation="Two road backbones, same original coarse scores, fourfold lattice and native bilinear-plus-correction projection. Report fitted operator on all pixels as primary; also apply the manuscript's coarse transition gate to exactly the same weights as secondary. Gate only at inference. Report released and fixed-four and paper references on every image.",
    panels="Full 128 BDD sample and 62 fog IDD images in 16 drives from v106. Native 6-pixel boundary-band mIoU, whole-image mIoU, 19/17 classes, seed mean and 5000 paired grouped bootstrap draws (10517). No selection on either cohort.")


class TensorAdaptiveConv:
    @staticmethod
    def apply(image, weights):
        k = weights.shape[-1]
        windows = image.unfold(2, k, 1).unfold(3, k, 1)
        return (windows * weights[:, None]).sum(dim=(-1,-2))


@lru_cache(maxsize=1)
def author_class():
    for name, checksum in FILES.items():
        assert old.digest(WEIGHTS / name) == checksum, name
    path = WEIGHTS / "upsamplers.py"
    src = path.read_text(encoding="utf-8")
    extension = "from featup.adaptive_conv_cuda.adaptive_conv import AdaptiveConv"
    assert src.count(extension) == 1
    namespace = {"__name__": "featup_pinned", "AdaptiveConv": TensorAdaptiveConv}
    exec(compile(src.replace(extension, ""), str(path), "exec"), namespace)
    return namespace["JBULearnedRange"]


class TwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        block = author_class()
        self.up1 = block(3,19,32,radius=3)
        self.up2 = block(3,19,32,radius=3)
        state = torch.load(WEIGHTS/"dino16.ckpt", weights_only=True, map_location="cpu")["state_dict"]
        kept = {k.removeprefix("upsampler."):v for k,v in state.items() if k.startswith(("upsampler.up1.","upsampler.up2."))}
        self.load_state_dict(kept, strict=True)
        self.register_buffer("mean", torch.tensor([.485,.456,.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std", torch.tensor([.229,.224,.225]).view(1,3,1,1), persistent=False)

    def forward(self, scores, guide):
        guide = (guide - self.mean) / self.std
        for stage in (self.up1,self.up2):
            shape = tuple(2*d for d in scores.shape[-2:])
            scores = stage(scores,F.adaptive_avg_pool2d(guide,shape))
        return scores


def verify_filter():
    gen = torch.Generator(device="cuda").manual_seed(420)
    image = torch.randn((2,3,9,11),generator=gen,device="cuda",dtype=torch.float64,requires_grad=True)
    weights = torch.randn((2,7,9,3,3),generator=gen,device="cuda",dtype=torch.float64,requires_grad=True)
    auto = TensorAdaptiveConv.apply(image,weights)
    direct = sum(image[:,:,i:i+7,j:j+9] * weights[:,None,:,:,i,j] for i in range(3) for j in range(3))
    deriv = torch.randn(auto.shape,generator=gen,device="cuda",dtype=torch.float64)
    adj_a = torch.autograd.grad(auto,(image,weights),deriv)
    adj_b = torch.autograd.grad(direct,(image,weights),deriv)
    errors = [(a-b).abs().max().item() for a,b in zip((auto,*adj_a),(direct,*adj_b))]
    assert max(errors)<1e-11
    return errors


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/"protocol.json"
    if path.exists():
        assert io.read(path)==PLAN
    else:
        io.write(path,PLAN)
    return old.digest(path)


def train():
    protocol = register()
    old.configure_cuda()
    errors = verify_filter()
    previous = old.read(training.OUT/"training.json")
    expected = {r["seed"]:r["sampler_end"] for r in previous["runs"] if r["arm"]=="bilinear_bounded4"}
    start = monotonic()
    pool = old.train_frames(old.DualScaleFeatures(),old.anchors(),False)
    assert len(pool)==9 and sum(map(len,pool))==96
    receipts=[]
    for trial,seed in enumerate(paired.SEEDS):
        receipt = OUT/f"run-{seed}.json"; path=OUT/f"featup-{seed}.pth"
        if receipt.exists():
            run=io.read(receipt)
            assert run["protocol"]==protocol and old.digest(path)==run["weights_sha256"] and run["sampler_end"]==expected[seed]
            receipts.append(run);continue
        old.set_seed(seed)
        net=TwoStage().cuda().train()
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=0)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,320,eta_min=.00003)
        rng=np.random.default_rng(92001+seed); history=[]
        for step in range(320):
            opt.zero_grad(set_to_none=True)
            inputs=[old.train_crop(pool,rng,trial,focused,step) for focused in (False,True)]
            logits=torch.cat([t[0] for t in inputs])
            guide=torch.cat([t[1][:1] for t in inputs]).float()/255
            targets=[t[2] for t in inputs]
            pred=net(logits,guide)[...,8:-8,8:-8]
            loss=pred.new_zeros(())
            for i,truth in enumerate(targets):
                edge=training.boundary_band(truth)[8:-8,8:-8]; labels=truth[8:-8,8:-8]
                weight=(labels<19).float()*(1+3*edge+2*((labels==6)|(labels==7)|(labels==14)))
                ce=F.cross_entropy(pred[i:i+1],labels[None],ignore_index=255,reduction="none")[0]
                loss=loss+.5*(ce*weight).sum()/weight.sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
            opt.step();schedule.step();history.append(float(loss.detach()))
            if (step+1)%80==0:
                print("FeatUp source fit",seed,step+1,round(float(np.mean(history[-20:])),5),flush=True)
        sampler=sha256(str(rng.bit_generator.state).encode()).hexdigest()
        assert sampler==expected[seed]
        old.torch_save(path,{k:v.detach().cpu() for k,v in net.state_dict().items()})
        run=dict(seed=seed,steps=320,parameters=sum(p.numel() for p in net.parameters()),loss=history,
                 protocol=protocol,sampler_end=sampler,weights=path.name,weights_sha256=old.digest(path))
        io.write(receipt,run);receipts.append(run)
    io.write(OUT/"training.json",dict(runs=receipts,source_frames=96,source_recordings=9,
             appendix_sampler_matched=True,filter_forward_and_gradient_max_abs_errors=errors,torch=str(torch.__version__),
             cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),source_manifest=old.digest(old.DATA/"fit.json"),
             seconds=monotonic()-start,code=old.digest(__file__)))


@torch.inference_mode()
def evaluate():
    register();old.configure_cuda()
    records=io.read(OUT/"training.json")
    assert matches_registered(records['code'], old.digest(__file__))
    models={s:ext.prior.base.prior.Target(s) for s in paired.ROAD}
    frozen=TwoStage().cuda().eval();fitted=[]
    for row in records["runs"]:
        path=OUT/row["weights"];assert old.digest(path)==row["weights_sha256"]
        net=TwoStage().cuda().eval();net.load_state_dict(torch.load(path,map_location="cuda",weights_only=True))
        fitted.append(net)
    all_counts=io.read(paired.OUT/"evaluation.json")
    card={}; start=monotonic()
    for panel in ("bdd","idd"):
        rows,manifest=paired.rows_for(panel)
        lock=sha256("".join(old.digest(p) for p in [OUT/"protocol.json",OUT/"training.json",paired.OUT/"evaluation.json",manifest,Path(__file__)]).encode()).hexdigest()
        dest=OUT/"counts"/panel;dest.mkdir(exist_ok=True,parents=True)
        for i,r in enumerate(rows):
            key=r["id"].replace("/","_")+".npz";file=dest/key
            if file.exists():
                with np.load(file,allow_pickle=False) as f: assert f["lock"].item()==lock
                continue
            rgb,ys=ext.prior.old.open_frame(r)
            padded,(h,w)=ext.prior.base.peer_input(rgb);small=padded[...,:h,:w].contiguous()
            outputs={}
            with np.load(paired.OUT/"counts"/panel/key,allow_pickle=False) as prev:
                assert prev["lock"].item()==all_counts["panels"][panel]["lock"] and prev["id"].item()==r["id"]
                for name,target in models.items():
                    z,released=target(small,rgb.shape[:2])
                    resize=ext.prior.base.resize
                    native=resize(z,rgb.shape[:2]);shape=tuple(4*n for n in z.shape[-2:])
                    guide=small if small.shape[-2:]==shape else resize(small,shape,antialias=True)
                    guide=guide.clamp(0,1).mul(255).round()/255; grid_base=resize(z,shape)
                    label=z.argmax(1,keepdim=True).float()
                    gate=(F.max_pool2d(label,3,1,1)!=-F.max_pool2d(-label,3,1,1)).repeat_interleave(4,2).repeat_interleave(4,3)
                    mask=lambda value:value.argmax(1)[0].byte().cpu().numpy()
                    masks={"released":[mask(released)],"featup_frozen":[],"featup_fit":[],"featup_fit_gated":[]}
                    for j,net in enumerate([frozen,*fitted]):
                        change=net(z,guide)-grid_base
                        masks["featup_frozen" if j==0 else "featup_fit"].append(mask(native+resize(change,rgb.shape[:2])))
                        if j: masks["featup_fit_gated"].append(mask(native+resize(change*gate,rgb.shape[:2])))
                    names=list(masks); packet=ext.count(masks,ys[0],None,names)
                    history=prev[name]
                    np.testing.assert_array_equal(packet[:,0],history[:,paired.ARMS.index("released")])
                    outputs[name]=np.stack([packet[:,names.index(arm)] if arm in names else history[:,paired.ARMS.index(arm)] for arm in ARMS],axis=1)
            tmp=file.with_suffix(".part")
            with tmp.open("wb") as f:np.savez_compressed(f,**outputs,lock=lock,id=r["id"],group=r["group"])
            tmp.replace(file)
            if i==0 or (i+1)%32==0 or i+1==len(rows):print("FeatUp",panel,i+1,len(rows),round(monotonic()-start,1),"s",flush=True)
        card[panel]=dict(lock=lock,frames=len(rows))
    io.write(OUT/"evaluation.json",dict(arms=list(ARMS),seeds=list(paired.SEEDS),panels=card,seconds=monotonic()-start))


def score():
    card=io.read(OUT/"evaluation.json");res={}
    for panel,meta in card["panels"].items():
        res[panel]={}
        rows,_=paired.rows_for(panel)
        for suite in paired.ROAD:
            bank=defaultdict(lambda:0)
            for row in rows:
                with np.load(OUT/"counts"/panel/(row["id"].replace("/","_")+".npz"),allow_pickle=False) as f:
                    assert f["lock"].item()==meta["lock"] and f["id"].item()==row["id"] and f["group"].item()==row["group"]
                    cm=f[suite]
                gt=cm.sum(-1);hit=cm.diagonal(axis1=-2,axis2=-1)
                assert (cm>=0).all() and (cm[1]<=cm[0]).all() and (gt==gt[:,:1,:1]).all()
                bank[row["group"]]+=np.stack([hit,gt+cm.sum(-2)-hit,gt],-1)
            groups=sorted(bank)
            classes=[c for c in range(19) if panel!="idd" or c not in (9,16)]
            data=np.take(np.stack([bank[g] for g in groups]),classes,axis=-2)
            res[panel][suite]={r:io.metrics(data[:,j,None],groups,ARMS,[suite],PAIRS,10517) for j,r in enumerate(("full","boundary"))}
    io.write(OUT/"results.json",dict(panels=res,bootstrap_draws=5000,bootstrap_seed=10517))
    print("FeatUp control results saved",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("part",nargs="?",default="all",choices=("train","evaluate","score","all"))
    part=p.parse_args().part
    if part in ("train","all"):train()
    if part in ("evaluate","all"):evaluate()
    if part in ("score","all"):score()

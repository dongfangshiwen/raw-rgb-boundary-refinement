"""Diagnose and repair the source fit of the published 19-class joint comparator."""
import argparse
from collections import defaultdict

import numpy as np
from boundary_refinement.data.source_integrity import matches_registered
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from boundary_refinement.inference import idd_fog_models as old
from boundary_refinement.training.camvid_query_loss import known_pixel_query_loss
from boundary_refinement.data.cohorts.idd_fog_data import OUT,fixed
from boundary_refinement.baselines.idd_peer_source import BASE,REV,SHA
from boundary_refinement.data.provenance import digest,read,write
from boundary_refinement.training.train_regional_guidance import configure_cuda,torch_save


DIR=OUT/"peer-road-init"
CHECKS=(160,320,480)


def plan(*, frozen=False):
    cfg=dict(code=digest(__file__),road_revision=REV,road_file_sha=SHA,camvid=digest(old.CAM),
        source={p:digest(old.base.DATA/(p+".json")) for p in ("fit","calibrate")},
        status="Follow-up after the first transfer predicted only two classes externally. No IDD image, label or "
               "score selects any checkpoint. Repeat evaluation on inspected IDD is descriptive, not independent confirmation.",
        initialization="Identical locally CamVid-trained Dehaze-Seg-S RGB backbone, original frequency neck and Mask2Former "
            "port. Transfer all matching pixel decoder, query transformer and classifier tensors from the official 19-class "
            "Cityscapes Mask2Former-Swin-T. No Swin backbone in the joint model. Mismatched input projection tensors use "
            "the local CamVid weights. The original release training and this adaptation differ.",
        training="Freeze the RGB backbone. Fit all semantic components on the same 96 MUSES fog frames, 9 drives, using "
            "one drive-uniform image per update at 512. Mask2Former Hungarian set targets (2/5/5 class, mask BCE, "
            "Dice; no-object 0.1); 12544 points for the final query output and 2048 for two auxiliary stages. "
            "Average objective with weights .25/.25/.5; ignore padded pixel centers. AdamW neck .0002 and head .00006, "
            "cosine x0.1 over 480 updates. Choose highest 512-grid 19-class mIoU at 160,320,480 on MUSES 24 frames "
            "from four separate calibration drives. All three seeds and selected epochs are reported.",
        diagnostic="Also score the initial pixel-CE comparator on all 24 calibration frames from each seed; record "
            "number of predicted classes, dominant pixel fraction, and mean IoU. No tuning of the proposed decoder.")
    path = DIR / "protocol.json"
    if frozen:
        if not matches_registered(read(path), cfg):
            raise ValueError("Frozen model protocol changed")
    else:
        fixed(path, cfg)
    return digest(path)


def initialize(seed):
    model=old.peer_model(seed)
    path=BASE/"model.safetensors"
    if digest(path)!=SHA:
        raise ValueError("Road pretrained release SHA mismatch")
    source=load_file(str(path))
    prefixes={"pixel_decoder.":"model.pixel_level_module.decoder.","query_decoder.":"model.transformer_module.",
              "class_predictor.":"class_predictor."}
    existing=model.seg_head.head.state_dict()
    copied=[]; unmatched=[]
    for k,t in existing.items():
        prefix=next(x for x in prefixes if k.startswith(x))
        src=prefixes[prefix]+k[len(prefix):]
        if src in source and t.shape==source[src].shape:
            existing[k]=source[src];copied.append(k)
        else:
            unmatched.append(k)
    if "class_predictor.weight" not in copied or len(copied)<250:
        raise ValueError("Incomplete transfer of the released road query head")
    model.seg_head.head.load_state_dict(existing)
    fixed(DIR/"road-initialization.json",dict(copied_tensors=len(copied),untouched=unmatched,source_sha=SHA))
    return model


@torch.no_grad()
def cache(model,scope):
    bank=defaultdict(list)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        for _,g,_,_,truth,rgb in old.paths(scope):
            x,size=old.peer_input(rgb)
            _,*features=model.backbone((x-model.mean)/model.std)
            y=F.interpolate(truth[None,None].float(),size,mode="nearest-exact")[0,0].byte()
            bank[g].append(([t.detach().to(device="cpu",dtype=torch.bfloat16) for t in features],size,y.cpu()))
    return bank


@torch.inference_mode()
def assess(model,pool):
    model.eval();table=np.zeros((19,19),dtype=np.int64)
    for rows in pool.values():
        for features,size,y in rows:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                scores=old.peer_forward(model,[t.cuda() for t in features],size)
            ids=scores.argmax(1)[0].cpu().numpy()
            truth=y.numpy().astype(np.int32)
            good=truth<19
            table+=np.bincount(19*truth[good]+ids[good],minlength=361).reshape(19,19)
    d=table.diagonal();u=table.sum(0)+table.sum(1)-d
    return dict(miou=float(np.divide(100.*d,u,out=np.zeros(19),where=u>0).mean()),
                classes_predicted=int((table.sum(0)>0).sum()),dominant_fraction=float(table.sum(0).max()/table.sum()),
                counts=table.tolist())


def stages(model,feat):
    head=model.seg_head.head
    pixel=head.pixel_decoder(model.freq_neck(feat),return_dict=True)
    query=head.query_decoder(pixel.multi_scale_features,pixel.mask_features,output_hidden_states=True)
    return [(head.class_predictor(query.intermediate_hidden_states[k].transpose(0,1)),query.masks_queries_logits[k])
            for k in (-7,-4,-1)]


def fit():
    configure_cuda();lock=plan()
    first=old.peer_model(old.SEEDS[0]).eval()
    train=cache(first,"fit");cal=cache(first,"calibrate")
    if (len(train),len(cal))!=(9,4) or set(train)&set(cal):
        raise ValueError("Source drives overlap")
    del first
    frame_groups=list(train.values())
    for s in old.SEEDS:
        path=DIR/f"queries-{s}.pth"
        receipt=path.with_suffix(".json")
        if receipt.is_file():
            note=read(receipt)
            if digest(path)!=note["weights"] or note["protocol"]!=lock:
                raise ValueError("Changed repaired comparison")
            continue
        old.set_seed(s)
        previous=old.trained_peer(s)
        before=assess(previous,cal)
        del previous
        model=initialize(s)
        model.backbone.eval().requires_grad_(False)
        opt=torch.optim.AdamW([dict(params=model.freq_neck.parameters(),lr=.0002),
                               dict(params=model.seg_head.parameters(),lr=.00006)],weight_decay=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:.1+.9*(1+np.cos(np.pi*k/CHECKS[-1]))/2)
        rng=np.random.default_rng(100920+s)
        gen=torch.Generator(device="cuda").manual_seed(870000+s)
        scores={};history=[]
        best=-1.;selected=0
        for step in range(1,CHECKS[-1]+1):
            model.freq_neck.train();model.seg_head.train()
            group=frame_groups[int(rng.integers(len(frame_groups)))]
            features,size,y=group[int(rng.integers(len(group)))]
            truth=F.pad(y.cuda().long(),(0,512-size[1],0,512-size[0]),value=255)[None]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                outputs=stages(model,[t.cuda() for t in features])
                loss=sum(coeff*known_pixel_query_loss(cls,mask,truth,points=n,generator=gen)
                         for (cls,mask),coeff,n in zip(outputs,(.25,.25,.5),(2048,2048,12544)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();sched.step()
            if step==1 or step%80==0:
                history.append(dict(step=step,loss=float(loss.detach())))
            if step in CHECKS:
                val=assess(model,cal);scores[str(step)]=val
                print("source MUSES",s,step,round(val["miou"],3),val["classes_predicted"],flush=True)
                if val["miou"]>best:
                    best=val["miou"];selected=step
                    torch_save(path,{k:v.detach().cpu() for k,v in model.state_dict().items() if k.startswith(("freq_neck.","seg_head."))})
        write(receipt,dict(seed=s,protocol=lock,weights=digest(path),previous_source=before,source_scores=scores,
                           selected=selected,selected_miou=best,history=history))
        del model,opt,sched


def load(seed):
    path=DIR/f"queries-{seed}.pth"
    r=read(path.with_suffix(".json"))
    if digest(path)!=r["weights"] or r["protocol"]!=plan(frozen=True):
        raise ValueError("Unverified repaired checkpoint")
    model=old.peer_model(seed)
    state=model.state_dict()
    part=torch.load(path,weights_only=True,map_location="cpu")
    if set(part)!={k for k in state if k.startswith(("freq_neck.","seg_head."))}:
        raise ValueError("Missing adapted semantic weights")
    state.update(part);model.load_state_dict(state)
    return model.eval().requires_grad_(False)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part",choices=("register","fit"))
    if ap.parse_args().part=="fit":
        fit()
    else:
        plan()

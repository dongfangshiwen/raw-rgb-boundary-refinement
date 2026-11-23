"""Same-image, native-grid comparisons and a 512-pixel-input dual-output comparison."""
import argparse
from collections import defaultdict
from hashlib import sha256

import numpy as np
from PIL import Image
import torch

from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.evaluation.paired import muses_transmission as j
from boundary_refinement.data.cohorts import mechanism_holdouts as synthetic
from boundary_refinement.data.cohorts.idd_fog_data import DATA,MAP,OUT,fixed
from boundary_refinement.data.provenance import digest,read
from boundary_refinement.training.train_regional_guidance import atomic_write,configure_cuda


DEPLOY=("bilinear","uniform","weighted","dysample","constant_variance","local_variance")
SMALL=("uniform512","weighted512","dysample512","published512")
SEEDS=fit.SEEDS


def register():
    fixed(OUT/"score-protocol.json", dict(code=digest(__file__),model=digest(OUT/"model-protocol.json"),
        data_rule=digest(OUT/"selection-rule.json"),deploy=list(DEPLOY),matched=list(SMALL),seeds=list(SEEDS),
        primary="Frozen weighted vs uniform at native deployed resolution on visually screened fog. Same head, "
                "anchor, RGB and native labels. Positive generalization only if the recording-paired lower 95% limit "
                "is above zero. Also require positive effects over uniform on full image to claim both endpoints. "
                "For candidate local variance require positive paired difference vs the unmodified weighted AND "
                "its parameter-matched constant-variance control. No tuning on IDD. DySample is a trained decoder control.",
        peer="Shared raw input at 512 long side for all, native GT. Our raw-detail head receives the same single-scale "
             "semantic result in both slots. Dehaze-Seg-S uses its author frequency neck and Mask2Former, local CamVid "
             "initialization and MUSES adaptation. Not identical source pretraining. GPU BF16 for its source-adapted "
             "forward, in line with its training. No clean reference on IDD. Show a fixed-order actual RGB and segmentation pair.",
        synthetic="Supplementary complete-output check on one previously inspected clear MUSES frame per each of 13 "
                  "recordings, lowest SHA256(idd100/photo/ID) in the V96 reserve, with t=.70 and .40 and airlight=.85 "
                  "in linear RGB. Both models take 512-long-side inputs. PSNR on identical 512 RGB grid against antialiased "
                  "clear RGB; semantic scores against all 19 native classes. Also score veiled input. Not new natural fog. "
                  "Bootstrap by recording, preserving the two veils together. Do not use to choose any weights."))


def native_truth(row):
    path=DATA/row["mask"]
    if digest(path)!=row["mask_sha"]:
        raise ValueError("Label digest mismatch")
    with Image.open(path) as im:
        labels=np.array(im)
    lut=np.full(256,255,dtype=np.uint8)
    for key,value in MAP.items():
        lut[key]=value
    return torch.from_numpy(lut[labels]).cuda().long()


class Counts:
    def __init__(self,truth):
        target=truth.cpu().numpy().astype(np.int32).ravel()
        edge=j.boundary_band(truth).cpu().numpy().ravel()
        self.valid=target<19
        self.base=((edge[self.valid].astype(np.int32)*19+target[self.valid])*19)
        self.edge_count=int(edge[self.valid].sum())

    def score(self,labels):
        ids=labels.detach().byte().cpu().numpy().ravel()[self.valid]
        if np.any(ids>=19):
            raise ValueError("A prediction is outside the Cityscapes ontology")
        c=np.bincount(self.base+ids,minlength=2*19*19).reshape(2,19,19)
        if c[1].sum()!=self.edge_count:
            raise ValueError("Invalid edge count")
        return c.sum(axis=0),c[1]


def utc_packet(file,cm,edge,rgb,arms):
    temp=file.with_suffix(".part")
    with temp.open("wb") as f:
        np.savez_compressed(f,full=cm,edge=edge,arms=np.asarray(arms),seeds=np.asarray(SEEDS),photo=rgb)
    temp.replace(file)


class Evaluation:
    def __init__(self):
        self.core=fit.FastJoint()
        self.raw=j.base.anchors()
        self.decoder=j.weighted.load()
        self.simple=j.base.load()
        self.variance={}
        self.peers={}
        lock=fit.register()
        for s in SEEDS:
            for m in (*fit.METHODS,"published"):
                if not fit.existing(m,s,lock):
                    raise RuntimeError(f"Missing complete fit {m}-{s}")
                if m=="published":
                    self.peers[s]=fit.trained_peer(s)
                else:
                    model=fit.DonorVariance().cuda()
                    model.load_state_dict(torch.load(fit.pdf_path(m,s),weights_only=True))
                    self.variance[m,s]=model.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self,rgb,truth,arms,reference=None,example=None):
        h,w=truth.shape
        full=np.zeros((len(arms),len(SEEDS),19,19),np.int64)
        edge=np.zeros_like(full)
        scores=Counts(truth)
        def add(arm,i,labels):
            full[arms.index(arm),i],edge[arms.index(arm),i]=scores.score(labels)
            if example is not None and i==0:
                Image.fromarray(labels.cpu().byte().numpy()).save(example/(arm+".png"))
        photo=np.zeros((0,3),np.float64)
        for small in (False,True):
            relevant=[m for m in arms if m!="published512" and m.endswith("512")==small]
            if not relevant:
                continue
            views,restored,_=self.core(rgb) if not small else self.core.encode(rgb,512)[:3]
            if example is not None:
                Image.fromarray(j.to_rgb(j.resize(restored,(h,w)))).save(example/("restored512.png" if small else "restored.png"))
            guides=j.base.render_guides(rgb,restored,views.fine.shape[-2:])
            for i,s in enumerate(SEEDS):
                z=j.raw_anchor(self.raw["raw_detail",s],views)
                native=j.resize(z,(h,w))
                t=j.base.guide_stencil(guides[:0],z.argmax(1)[0])
                weight=j.weighted.color_weights(z,"dual_core")
                for arm in relevant:
                    mode=arm.removesuffix("512")
                    if mode=="bilinear":
                        add(arm,i,native.argmax(1)[0]);continue
                    if mode=="dysample":
                        delta=j.base.prediction(mode,self.simple[mode,s],z,None)
                    else:
                        if mode in fit.METHODS:
                            stencil=self.variance[mode,s].stencil(guides,z,mode,weight,t)
                        else:
                            weights=weight if mode=="weighted" else torch.full_like(weight,1./16)
                            stencil=j.weighted.prepare(guides,z,"dual_core",t,weights)
                        delta=j.base.prediction("dual_free",self.decoder["dual_core",s],z,stencil)
                    add(arm,i,(native+j.resize(delta,(h,w))).argmax(1)[0])
            if small and reference is not None:
                photo=np.empty((3,3),np.float64)
                clear=j.resize(reference,restored.shape[-2:],antialias=True)
                x,_=fit.peer_input(rgb)
                def error(img):
                    return float((img-clear).square().mean().item())
                photo[0,:]=error(x[...,:clear.shape[-2],:clear.shape[-1]])
                photo[1,:]=error(restored)
        if "published512" in arms:
            x,size=fit.peer_input(rgb)
            first=self.peers[SEEDS[0]]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                img,*feat=first.backbone((x-first.mean)/first.std)
                restored=(img.float()*first.std+first.mean)[...,:size[0],:size[1]].clamp(0.,1.)
                for i,s in enumerate(SEEDS):
                    logits=fit.peer_forward(self.peers[s],feat,size)
                    add("published512",i,j.resize(logits.float(),(h,w)).argmax(1)[0])
                if reference is not None:
                    photo[2,:]=error(restored)
                if example is not None:
                    Image.fromarray(j.to_rgb(j.resize(restored,(h,w)))).save(example/"published-rgb.png")
        if np.any(full.sum((-1,-2))!=scores.valid.sum()) or np.any(edge.sum((-1,-2))!=scores.edge_count):
            raise ValueError("Unfilled or unpaired result")
        return full,edge,photo


def main(which):
    configure_cuda();register()
    data=read(DATA/"evaluation.json")
    receipts={f"{m}-{s}":digest(fit.pdf_path(m,s)) for m in (*fit.METHODS,"published") for s in SEEDS}
    packet=dict(score=digest(OUT/"score-protocol.json"),models=receipts,manifest=digest(DATA/"evaluation.json"),
                weather=digest(OUT/"weather-lock.json"))
    fixed(OUT/"inference-lock.json",packet)
    engine=Evaluation()
    rows=[]
    directory=OUT/"counts"/which
    directory.mkdir(parents=True,exist_ok=True)
    if which=="external":
        arms=(*DEPLOY,*SMALL)
        examples={min((r["id"] for r in data["rows"] if cohort in r["cohorts"]),
                      key=lambda k:sha256(("idd100/figure/"+k).encode()).digest()) for cohort in ("mixed","fog")}
        for i,row in enumerate(data["rows"]):
            key=row["id"]; file=directory/(key.replace("/","_")+".npz")
            if not file.exists():
                p=DATA/row["rgb"]
                if digest(p)!=row["rgb_sha"]:
                    raise ValueError("RGB digest mismatch")
                with Image.open(p) as im:
                    rgb=np.array(im.convert("RGB"))
                truth=native_truth(row)
                example=None
                if key in examples:
                    example=OUT/"examples"/key.replace("/","_")
                    example.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(rgb).save(example/"input.png")
                    Image.fromarray(truth.cpu().byte().numpy()).save(example/"reference.png")
                full,edge,photo=engine.forward(rgb,truth,arms,example=example)
                utc_packet(file,full,edge,photo,arms)
            rows.append(dict(id=key,group=row["group"],cohorts=row["cohorts"],file=str(file),sha=digest(file)))
            if (i+1)%20==0:
                print("external predictions",i+1,"/",len(data["rows"]),flush=True)
    else:
        arms=SMALL
        groups=defaultdict(list)
        for ds,key,g,dt,rgb,truth in synthetic.scene_cases():
            if ds=="new_muses_clear":
                groups[g].append((key,dt,rgb,truth.cpu()))
        if len(groups)!=13:
            raise ValueError("Clear drive counts changed")
        for group,members in groups.items():
            key,_,raw,truth=min(members,key=lambda r:sha256(("idd100/photo/"+r[0]).encode()).digest())
            target=torch.from_numpy(raw).cuda().permute(2,0,1)[None].float()/255.
            for transmission in (.70,.40):
                label=f"t{round(transmission*100)}"
                file=directory/(key.replace("/","_")+"_"+label+".npz")
                if not file.exists():
                    hazy=j.to_rgb(j.srgb(transmission*j.linear_rgb(target)+(1.-transmission)*.85))
                    full,edge,photo=engine.forward(hazy,truth.cuda(),arms,reference=target)
                    utc_packet(file,full,edge,photo,arms)
                rows.append(dict(id=key+"_"+label,group=group,cohorts=["synthetic",label],file=str(file),sha=digest(file)))
    fixed(OUT/(which+"-index.json"),dict(inference=digest(OUT/"inference-lock.json"),rows=rows))
    print(which,"complete",len(rows),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part",choices=("register","external","synthetic"))
    which=ap.parse_args().part
    if which=="register":
        register()
    else:
        main(which)

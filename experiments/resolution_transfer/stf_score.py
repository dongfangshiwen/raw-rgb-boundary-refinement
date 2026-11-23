"""Frozen semantic predictions at measured fog-scene reference points."""
from pathlib import Path
import argparse
from collections import Counter
from hashlib import sha256
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

from experiments.resolution_transfer.cohort import OUT, ROOT, digest, read, write
from experiments.resolution_transfer.scale_training import SEEDS, image_at, range_model
from experiments.resolution_transfer.evaluate import TARGETS, save
from experiments.resolution_transfer.stf_data import TO_CITY,calibrate,project
from experiments.recording_gate import gate

ARMS=("released","coarse","fixed","paper","shared","by_scale","shifted_paper","shifted_shared")
REGIONS=("all_points","near6","near12","near24")
SHIFTS=((0,0),(-2,0),(2,0),(0,-2),(0,2))
PAIRS={"paper":("released","coarse","fixed","shifted_paper"),
       "shared":("paper","fixed","shifted_shared"),"by_scale":("shared","paper","fixed"),"fixed":("released",)}
# High-contrast colours for manual verification on dark as well as daylight imagery.
COLOURS={0:(204,142,253),1:(244,73,190),2:(188,163,129),4:(239,143,95),5:(255,220,50),
         7:(255,88,30),8:(72,228,86),9:(155,183,31),11:(255,30,30),13:(38,196,255),
         14:(33,123,255),17:(178,79,234),18:(129,236,255)}


def preview():
    rows=read(OUT/"stf-manifest.json")["rows"]
    dates=sorted({r["day"] for r in rows},key=lambda s:sha256(("stf-overlay-113/"+s).encode()).digest())[:4]
    tr,P,shape,_=calibrate()
    width,height=660,352
    canvas=Image.new("RGB",(2*width, len(dates)*(height+32)+41),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("times.ttf",22)
    draw.text((8,5),"Native RGB",fill="#152c35",font=font)
    draw.text((width+8,5),"Projected human labels",fill="#152c35",font=font)
    table=[]
    for n,day in enumerate(dates):
        row=min([r for r in rows if r["day"]==day],key=lambda v:v["id"])
        rgb,xy,lab,dist,counts=project(row,tr,P,shape)
        im=Image.fromarray(rgb).resize((width,height),Image.Resampling.LANCZOS)
        annotated=im.copy();pen=ImageDraw.Draw(annotated)
        xy=np.rint(xy*np.array((width/shape[1],height/shape[0]))).astype(int)
        for k in np.unique(lab):
            pen.point([tuple(p) for p in xy[lab==k]],fill=COLOURS[k])
        top=39+n*(height+32)
        canvas.paste(im,(0,top));canvas.paste(annotated,(width,top))
        draw.text((8,top+height+1),f"{day}    {row['weather'].replace('_',' ')}",fill="#243a42",font=font)
        table.append({"id":row["id"],"weather":row["weather"],**counts})
    canvas.save(OUT/"stf-projection-audit.jpg",quality=92)
    write(OUT/"stf-preview.json",dict(rows=table,selection="One first timestamp on four hash-selected dates",colours=COLOURS))


@torch.inference_mode()
def predict(rgb,xy,target,model,fixed,learned,models):
    h,w=rgb.shape[:2];im=image_at(rgb,768)
    z,official=model(im,(h,w));coarse=gate.ev.resize(z,(h,w))
    assert tuple(4*n for n in z.shape[-2:])==tuple(im.shape[-2:])
    if target=="segformer":assert torch.equal(coarse,official)
    guide=im.mul(255).round().byte()
    X=torch.from_numpy(xy[:,0]).cuda()
    Y=torch.from_numpy(xy[:,1]).cuda()
    ids=torch.stack([(Y+dy)*w+X+dx for dx,dy in SHIFTS])
    all_pred=np.empty((len(SHIFTS),len(ARMS),len(SEEDS),len(xy)),np.uint8)
    def store(arm,scores,seed=None):
        predicted=scores.argmax(1)[0].flatten()[ids].byte().cpu().numpy()
        if seed is None:all_pred[:,ARMS.index(arm)]=predicted[:,None]
        else:all_pred[:,ARMS.index(arm),seed]=predicted
    store("released",official);store("coarse",coarse)
    def native(d):return coarse+gate.ev.resize(d,(h,w))
    for shifted in (False,True):
        g=torch.roll(guide,(guide.shape[-2]//2,guide.shape[-1]//2),(-2,-1)) if shifted else guide
        st=gate.ev.prepare(z,g)
        old=gate.ev.legacy.guide_stencil(g[[0,0]],z.argmax(1)[0])
        if not shifted:store("fixed",native(fixed.correction(z,st)))
        for j,seed in enumerate(SEEDS):
            store("shifted_paper" if shifted else "paper",native(gate.ev.legacy.prediction("dual_free",learned["dual_core",seed],z,old)),j)
            store("shifted_shared" if shifted else "shared",native(models["shared",seed].correction(z,st)),j)
            if not shifted:store("by_scale",native(models["by_scale",seed].correction(z,st)),j)
    return all_pred


def run():
    rows=read(OUT/"stf-manifest.json")["rows"]
    assert read(OUT/"stf-visual-check.json")["proceed"]
    file=OUT/"stf-scoring.json"
    write(file,dict(implementation=digest(__file__),projection=digest(Path(__file__).with_name("stf_data.py")),training=digest(OUT/"training.json"),
                    protocol=digest(OUT/"stf-protocol.json"),manifest=digest(OUT/"stf-manifest.json"),geometry_review=digest(OUT/"stf-visual-check.json"),
                    padding_amendment=digest(OUT/"stf-padding-amendment.json"),
                    arms=list(ARMS),offset_xy=[list(z) for z in SHIFTS],regions=list(REGIONS),seeds=list(SEEDS)),True)
    sig=digest(file)
    gate.ev.setup.configure_cuda()
    tr,P,shape,_=calibrate()
    models={(v["method"],v["seed"]):range_model(v["effective"]["768"]) for v in read(OUT/"training.json")["runs"] if v["method"] in ("shared","by_scale")}
    learned=gate.ev.original.load();fixed=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    for target in TARGETS:
        model=gate.ev.Target(target)
        root=OUT/"measured-points"/target;root.mkdir(exist_ok=True,parents=True)
        for i,row in enumerate(rows):
            f=root/(row["id"]+".npz")
            if f.is_file():
                with np.load(f,allow_pickle=False) as a:assert a["lock"].item()==sig and a["id"].item()==row["id"]
                continue
            rgb,xy,label,dist,counts=project(row,tr,P,shape)
            pred=predict(rgb,xy,target,model,fixed,learned,models)
            band=np.searchsorted([6.,12.,24.],dist).astype(np.uint8)
            save(f,pred=pred,truth=label,xy=xy.astype(np.int16),band=band,id=row["id"],day=row["day"],weather=row["weather"],
                 arms=ARMS,seeds=SEEDS,offsets=SHIFTS,shape=shape,lock=sig)
            if i==0 or (i+1)%12==0 or i+1==len(rows):print("Projected fog",target,i+1,"/",len(rows),flush=True)
        del model;torch.cuda.empty_cache()


def packet(f):
    lab=f["truth"];band=f["band"];pred=f["pred"]
    assert len(lab)==len(band)==pred.shape[-1] and set(lab)<=set(TO_CITY.values())
    assert pred.shape==(5,len(ARMS),len(SEEDS),len(lab)) and (pred<19).all()
    hist=np.empty((4,5,len(ARMS),len(SEEDS),19,19),np.int32)
    base=(band.astype(np.int32)*19+lab)*19
    for ix in np.ndindex(pred.shape[:3]):
        q=np.bincount(base+pred[ix],minlength=4*19*19).reshape(4,19,19)
        hist[(slice(None),)+ix]=np.stack((q.sum(0),q[0],q[:2].sum(0),q[:3].sum(0)))
    assert (hist[1]<=hist[2]).all() and (hist[2]<=hist[3]).all() and (hist[3]<=hist[0]).all()
    assert (hist.sum(-1)==hist.sum(-1)[:,:1,:1,:1]).all()
    return hist


def statistics(bank,days):
    from boundary_refinement.evaluation.semantic.path_counts import metrics,scores
    dates=np.unique(days)
    data=np.stack([bank[days==day].sum(0,dtype=np.int64) for day in dates])
    truth=data.sum(-1);hit=data.diagonal(axis1=-1,axis2=-2);union=truth+data.sum(-2)-hit
    stats=np.take(np.stack((hit,union,truth),-1),sorted(TO_CITY.values()),axis=-2)
    output={}
    for k,region in enumerate(REGIONS):
        for j,offset in enumerate(SHIFTS):
            if j and region!="near12":continue
            name=region if j==0 else f"near12@{offset[0]},{offset[1]}"
            v=stats[:,k,j][:,None]
            d=metrics(v,dates.tolist(),list(ARMS),["network"],PAIRS,11324)
            if not d["groups"]:output[name]={"groups":0};continue
            r=d["targets"]["network"];mask=v[:,0,0,0,:,2].sum(-1)>0
            individual=scores(v[mask],np.eye(sum(mask)))[0].mean(-1)[:,0]
            for candidate,controls in PAIRS.items():
                for comp in controls:
                    r["contrasts"][candidate+" - "+comp]["equal_day_pp"]=float((individual[:,ARMS.index(candidate)]-individual[:,ARMS.index(comp)]).mean())
            output[name]=dict(**r,recording_days=d["groups"],class_point_support=d["class_support_pixels"],valid_points=sum(d["class_support_pixels"]))
    return output


def rescore():
    sig=digest(OUT/"stf-scoring.json");rows=read(OUT/"stf-manifest.json")["rows"];out={};fingerprints={}
    for target in TARGETS:
        bank=[];counts=[]
        for r in rows:
            with np.load(OUT/"measured-points"/target/(r["id"]+".npz"),allow_pickle=False) as f:
                assert f["lock"].item()==sig and f["id"].item()==r["id"] and f["arms"].tolist()==list(ARMS)
                hashed=sha256(f["xy"].tobytes()+f["truth"].tobytes()+f["band"].tobytes()).hexdigest()
                if target==TARGETS[0]:fingerprints[r["id"]]=hashed
                else:assert fingerprints[r["id"]]==hashed
                bank.append(packet(f));counts.append([len(f["truth"]),int((f["band"]<2).sum())])
        a=np.stack(bank);days=np.array([r["day"] for r in rows]);weather=np.array([r["weather"] for r in rows])
        panels={}
        for panel in ("all","light_fog","dense_fog"):
            pick=np.ones(len(rows),bool) if panel=="all" else weather==panel
            panels[panel]={"frames":int(pick.sum()),"days":len(set(days[pick])),"endpoints":statistics(a[pick],days[pick])}
        out[target]=panels
        print(target,"STF near12",panels["all"]["endpoints"]["near12"]["contrasts"]["paper - released"],flush=True)
    write(OUT/"stf-results.json",dict(results=out,classes=sorted(TO_CITY.values()),weather=dict(Counter(r["weather"] for r in rows)),
                                     frames=len(rows),lock=sig,points_per_image=counts,shape=list(calibrate()[2])))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("step",choices=("preview","run","rescore"));args=ap.parse_args()
    if args.step=="preview":preview()
    else:
        if args.step=="run":run()
        rescore()

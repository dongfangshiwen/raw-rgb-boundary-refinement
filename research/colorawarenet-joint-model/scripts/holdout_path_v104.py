"""Reserve disjoint BDD keyframes before inference. Weather screening only uses RGB."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from scripts import reserve_drivable_v103 as road
from scripts import screen_dense_clips_v103 as previous
from scripts import weatherproof_release_v103 as io

OUT=io.WORK/"experiment_exports/matched-color-decoding-v104"
DATA=io.WORK/"benchmark_data/matched-color-v104"
RULE={
    "question":"Does raw intervening RGB variation add transferable information beyond spatial bilinear weights and endpoint Gaussian RGB on the manuscript's exact virtual grid?",
    "operator":"Six equally spaced straight-line RGB samples from each child to each bilinear donor center. Penalize RMS RGB arc length in excess of endpoint distance, saturated with scale 0.08. The Gaussian uses the same child and donor block mean as the manuscript. Keep the raw guide, active 3x3 coarse-cell gate, positive bilinear support, zero padding of the correction outside the active cells, and bilinear projection of corrections onto the native base identical across fitted arms.",
    "training":"96 real-fog MUSES source frames, 9 LiDSeg fit recordings. Three seeds 101,211,307; each 320 AdamW updates with identical sampled labeled working-grid children per arm. Six arms: Gaussian with fitted bandwidth and spatial exponent, five effective saturating-range coefficients (four range amplitudes and spatial exponent), Gaussian plus an additional endpoint-only coefficient, Gaussian plus aligned path excess, Gaussian plus shifted-path excess and Gaussian plus a permutation of the four path values. Equalized three-scalar controls for the path method. No choice by target outcomes.",
    "reserve":"One semantic-training-partition BDD100K keyframe per unused prefix. Exclude all earlier reserve lists, including all 2048 V103 reserved RGBs. SHA256 path104/<id> order; first 128 form a disjoint, unconditional confirmatory all-weather test. Next at most 1536 are reserved for a fog-enriched screen. On the latter, rank by cosine similarities of frozen ImageNet MobileNetV3 embeddings to earlier RGB-only visually screened fog/non-fog examples (difference of normalized centroids). Inspect top 384 plus 48 drawn by SHA256 from the remaining rows. Enlarge shortlisted RGBs; lock visible distance haze independently of labels, logits and comparative results. No expansion after scoring. Keyframe prefix is a proxy for video, not location. Report selection and all outcomes.",
    "inference":"Primary Mask2Former, secondary SegFormer-B1. All frozen target backbones and fixed weights. Evaluate both new BDD samples and previously inspected IDD fog and 4 BDD fog clips separately. Primary scores: pooled fixed 19-class mIoU on 6-native-pixel semantic boundary band for BDD; historical IDD uses its original 17 classes. Report full-image mIoU, GT-present sensitivity, and paired 5000-draw group bootstrap (seed 10417), descriptive only below six eligible groups. No independent fog mechanism claim unless candidate improves both endpoint and both broken-path controls, with positive lower intervals on fresh fog primary; report unselected and second-backbone transfer independently. No adjustment for exploratory secondary comparisons.",
}

def reserve():
    OUT.mkdir(exist_ok=True,parents=True);DATA.mkdir(exist_ok=True,parents=True)
    p=OUT/"protocol.json"
    if p.exists():assert io.read(p)==RULE
    else:io.write(p,RULE)
    dest=OUT/"reserve.json"
    if dest.exists():return io.read(dest)["rows"]
    reg=io.read(io.WORK/"benchmark_data/bdd-fog-v73/registration.json")
    seen=set(k.split("-")[0] for k in reg["cases"])
    for path in [*[road.OLD/(f+".json") for f in ("bdd-screen-inventory","bdd-manifest","bddval-manifest","fogreserve-manifest")],road.LAST/"manifest.json",io.OUT/"bdd-manifest.json",previous.R/"reserve.json"]:
        seen.update(r["id"].split("-")[0] for r in io.read(path)["rows"] if path!=road.LAST/"manifest.json" or r["cohort"]=="bdd")
    revision=reg["revisions"]["semantic"];source=reg["source"]["semantic"]
    meta=json.loads(road.fetch(f"https://huggingface.co/api/datasets/{source}/revision/{revision}"))
    assert meta["sha"]==revision
    paths=[x["rfilename"] for x in meta["siblings"]]
    base=f"https://huggingface.co/datasets/{source}/resolve/{revision}/"
    def assets(folder,ext):
        return {Path(x).name.removesuffix(ext):base+x for x in paths if x.startswith(folder+"/") and x.endswith(ext)}
    images=assets("data/images_10k/10k/train",".jpg"); masks=assets("data/seg_maps/labels/train","_train_id.png")
    assert images.keys()==masks.keys() and len(images)==7000
    rows=[];excluded=len(seen)
    for key in sorted(images,key=lambda k:io.sha(("path104/"+k).encode())):
        group=key.split("-")[0]
        if group in seen:continue
        seen.add(group)
        rows.append(dict(id=key,group=group,cohort="bdd_dense",panel="random" if len(rows)<128 else "screen",rgb_url=images[key],mask_url=masks[key]))
    available=len(rows);rows=rows[:1664]
    io.write(dest,dict(rows=rows,available=available,excluded_prefixes=excluded,revision=revision,protocol=io.sha(p.read_bytes())))
    print("Disjoint reserve",len(rows),"available",available,"historical prefixes",excluded,flush=True)
    return rows

def picture(r):
    p=DATA/(r["id"]+".jpg")
    if not p.exists():
        blob=road.fetch(r["rgb_url"])
        with Image.open(BytesIO(blob)) as im:assert min(im.size)>64 and im.format=="JPEG",(r["id"],im.size,im.format)
        tmp=p.with_suffix(".part");tmp.write_bytes(blob);tmp.replace(p)
    return p

def stage(r):
    paths=[picture(r),DATA/(r["id"]+".png")]
    if not paths[1].exists():
        blob=road.fetch(r["mask_url"])
        with Image.open(paths[0]) as image,Image.open(BytesIO(blob)) as mask:
            assert mask.size==image.size and set(np.unique(np.array(mask)))<=set(range(19))|{255}
        tmp=paths[1].with_suffix(".maskpart");tmp.write_bytes(blob);tmp.replace(paths[1])
    return dict(r,paths=[p.relative_to(io.WORK).as_posix() for p in paths],sha256=[io.sha(p.read_bytes()) for p in paths])

def prepare():
    rows=reserve()
    with ThreadPoolExecutor(14) as pool:
        it={pool.submit(stage if r["panel"]=="random" else picture,r):r for r in rows}
        bank={}
        for i,f in enumerate(as_completed(it),1):
            val=f.result(); row=it[f]
            if row["panel"]=="random":bank[row["id"]]=val
            if len(bank)==128 and not (OUT/"random.json").exists():io.write(OUT/"random.json",{"rows":[bank[r["id"]] for r in rows if r["panel"]=="random"]})
            if i%200==0:print("Staged",i,len(rows),flush=True)
    assert len(bank)==128

def sheets(rows,large):
    pack,cols,cell=(12,3,(385,235)) if large else (36,6,(206,136))
    for start in range(0,len(rows),pack):
        sub=rows[start:start+pack]
        canvas=Image.new("RGB",(cols*cell[0],((len(sub)+cols-1)//cols)*cell[1]),"white")
        draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(sub):
            with Image.open(picture(r)) as im:
                im.thumbnail((cell[0]-2,cell[1]-17));x=j%cols*cell[0];y=j//cols*cell[1]
                canvas.paste(im,(x,y+15));draw.text((x+2,y),r["code"]+"  "+r["id"],fill="black")
        canvas.save(OUT/(("close" if large else "screen")+f"-{start//pack+1:02d}.jpg"),quality=90)

def rank():
    import torch
    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
    torch.set_num_threads(5)
    model=mobilenet_v3_small(weights=None).eval()
    weight=io.WORK/"colorawarenet-joint-model/pretrained/mobilenet-v3-small/model.pth"
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
    pre=MobileNet_V3_Small_Weights.DEFAULT.transforms()
    @torch.inference_mode()
    def features(paths):
        vectors=[]
        for i in range(0,len(paths),32):
            batch=[]
            for path in paths[i:i+32]:
                with Image.open(path) as im:batch.append(pre(im.convert("RGB")))
            vectors.append(torch.nn.functional.normalize(model.avgpool(model.features(torch.stack(batch))).flatten(1)))
        return torch.cat(vectors)
    dense=io.read(previous.R/"screen.json")["rows"]; approved=set(io.read(previous.R/"weather-lock.json")["codes"])
    yes,no=[],[]
    for r in dense:
        (yes if r["code"] in approved else no).append(previous.DATA/(r["id"]+"-thumb.jpg"))
    earlier=io.read(io.OUT/"bdd-manifest.json")["rows"];ok=set(io.read(io.OUT/"bdd-visual-lock.json")["ids"])
    yes.extend(io.WORK/r["paths"][0] for r in earlier if r["id"] in ok)
    unit=lambda x:torch.nn.functional.normalize(x.mean(0),dim=0)
    axis=unit(features(yes))-unit(features(no))
    rows=[r for r in reserve() if r["panel"]=="screen"]
    scores=(features([picture(r) for r in rows])@axis).tolist()
    order=sorted(range(len(rows)),key=lambda i:(-scores[i],rows[i]["id"]))
    ix=order[:384]+sorted(order[384:],key=lambda i:io.sha(("control104/"+rows[i]["id"]).encode()))[:48]
    selected=[dict(rows[i],code=f"H{j:03d}",rank_score=scores[i]) for j,i in enumerate(ix,1)]
    io.write(OUT/"screen.json",dict(rows=selected,positive=len(yes),negative=len(no),weights=io.sha(weight.read_bytes())))
    sheets(selected,False)

def select(final):
    rows=io.read(OUT/"screen.json")["rows"];codes=io.read(OUT/("weather-lock.json" if final else "shortlist.json"))["codes"]
    selected=[r for r in rows if r["code"] in codes];assert len(codes)==len(selected)
    if not final:sheets(selected,True);return
    assert set(codes)<=set(io.read(OUT/"shortlist.json")["codes"])
    with ThreadPoolExecutor(8) as pool:complete=list(pool.map(stage,selected))
    io.write(OUT/"fog-new.json",dict(rows=complete,lock=io.sha((OUT/"weather-lock.json").read_bytes())))

if __name__=="__main__":
    arg=argparse.ArgumentParser();arg.add_argument("step",choices=("prepare","rank","close","stage"));mode=arg.parse_args().step
    if mode=="prepare":prepare()
    elif mode=="rank":rank()
    else:select(mode=="stage")

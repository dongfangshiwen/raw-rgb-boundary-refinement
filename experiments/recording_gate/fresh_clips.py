"""Reserve and visually screen real BDD100K recordings before opening their labels."""
from boundary_refinement.paths import WORKSPACE
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import argparse
import json
import re
import sys

from PIL import Image, ImageDraw
import numpy as np

HERE = (WORKSPACE / "experiment_exports" / "recording-heldout-color-gate-2026-11-21")
ROOT = HERE.parents[1]
from boundary_refinement.data.cohorts import holdout_path as history

DATA = ROOT / "benchmark_data/bdd-heldout-2026-11-21"
key = lambda s: sha256(s.encode()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def put(p, obj, immutable=False):
    if immutable and p.exists():
        assert read(p) == obj, p
        return
    temp = p.with_suffix(".part")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(p)


def history_files():
    v103, v104 = history.io.OUT, history.OUT
    return [*[history.road.OLD/(x+".json") for x in ("bdd-screen-inventory", "bdd-manifest", "bddval-manifest", "fogreserve-manifest")],
            history.road.LAST/"manifest.json", v103/"bdd-selection.json", v103/"bdd-manifest.json",
            history.previous.R/"reserve.json", v104/"reserve.json", v104/"manuscript-extension/bdd-reserve.json"]


def old_rgb():
    pat = re.compile(r"^([0-9a-f]{8})-([0-9a-f]{8})")
    for folder in (ROOT/"benchmark_data").iterdir():
        if not folder.is_dir() or folder == DATA:
            continue
        for path in folder.rglob("*.jpg"):
            match = pat.match(path.name)
            if match:
                yield match.group(1), path


def reserve():
    DATA.mkdir(exist_ok=True)
    dest = HERE/"bdd-reserve.json"
    if dest.exists():
        return read(dest)["rows"]
    protocol = {
        "question":"Does adding a previously frozen raw-colour correction to released semantic scores improve native boundaries on previously unused recording prefixes? Does source-fitted colour weighting or new source-only acceptance add value over the same fixed correction?",
        "frozen_weights":sha256((HERE/"weights.json").read_bytes()).hexdigest(),
        "prior_score_protocol":sha256((HERE.parent/"decoder-resolution-2026-11-21/official-residual-protocol.json").read_bytes()).hexdigest(),
        "population":"All previously unused BDD100K semantic-train keyframes with native dense train-ID annotations, one per prefix. Exclude every earlier screened/reserved/evaluated prefix in the listed manifests and any additional locally staged BDD JPEG in benchmark_data. The dataset training partition is used as external evaluation for the Cityscapes released backbones and MUSES-fitted decoders. Unknown geographic/route overlap.",
        "allocation":"Sort by SHA256('v112-bdd/'+id). First 64 distinct video prefixes are a separate unconditional all-weather sample. Screen all remaining RGBs with the frozen previous MobileNetV3 ImageNet centroid direction. Visually inspect top 256 and 48 lowest hash('v112-screen-control/'+id) among the remainder. Enlarge plausible candidates. Image-only inclusion: a visible pale veil and distance-dependent contrast loss at multiple depths. Exclude ambiguous cloud, road spray, glass moisture, optical glare and shadows. One visual reader; not instrument- or author-confirmed fog.",
        "leakage":"Seal visual decisions before opening dense target maps or inferring segmentation. Check candidate RGB for near copies of locally available historical BDD RGBs or already retained new RGBs using normalized 32x18 RGB mean absolute error <0.035 AND 64-bit difference hash Hamming distance <=2. Retain earlier locked row. Full source and target model weights remain frozen.",
        "inference":"Official pretrained Mask2Former is primary, SegFormer-B1 secondary. Long edge 1024 is primary, 512 secondary. Use all three existing manuscript seeds 101/211/307, same aligned raw RGB, boundary radius six native pixels, fixed 19 classes, zero for zero-union. On the same scores compare released baseline, old paper placement, released + fitted, released + fixed four-range and source-only aligned acceptance on released + fixed. Include half-shifted fitted residual as negative control. Pool confusion per seed then average seed scores. Primary endpoint fitted residual vs released and matched fixed at 1024 on visually selected clips. Paired 5000 prefix-bootstrap draws, RNG 11223; no interval / inference with fewer than six prefixes. Also report unconditional panel separately, native full mIoU, and ground-truth-present sensitivity. No inference of general fog efficacy from a few or ambiguous keyframes; do not tune on these panels.",
        "history":[str(p.relative_to(ROOT)) for p in history_files()]
    }
    put(HERE/"bdd-protocol.json", protocol, True)
    reg = read(ROOT/"benchmark_data/bdd-fog-v73/registration.json")
    seen = {k.split("-")[0] for k in reg["cases"]}
    for f in history_files():
        seen.update(r["id"].split("-")[0] for r in read(f)["rows"] if f != history.road.LAST/"manifest.json" or r["cohort"] == "bdd")
    seen.update(g for g,p in old_rgb())
    previous = set(seen)
    rev, repo = reg["revisions"]["semantic"],reg["source"]["semantic"]
    meta=json.loads(history.road.fetch(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}"))
    assert meta["sha"] == rev
    files = [r["rfilename"] for r in meta["siblings"]]
    def ids(folder,suffix):
        return {Path(f).name.removesuffix(suffix):f for f in files if f.startswith(folder+"/") and f.endswith(suffix)}
    imgs=ids("data/images_10k/10k/train",".jpg")
    labels=ids("data/seg_maps/labels/train","_train_id.png")
    assert imgs.keys() == labels.keys() and len(imgs)==7000
    rows=[]
    for i in sorted(imgs,key=lambda k:key("v112-bdd/"+k)):
        group=i.split("-")[0]
        if group in seen:continue
        seen.add(group)
        rows.append(dict(id=i,group=group,panel="random" if len(rows)<64 else "screen",rgb=imgs[i],label=labels[i]))
    assert not previous.intersection(r["group"] for r in rows)
    put(dest,dict(rows=rows,prior_prefixes=len(previous),repository=repo,revision=rev,protocol=sha256((HERE/"bdd-protocol.json").read_bytes()).hexdigest()),True)
    print("Previously unused keyframes",len(rows),"prior prefixes",len(previous),flush=True)
    return rows


def base():
    reg=read(HERE/"bdd-reserve.json")
    return f"https://huggingface.co/datasets/{reg['repository']}/resolve/{reg['revision']}/"


def thumb(r, base_url):
    p=DATA/(r["id"]+"-thumb.jpg")
    if not p.exists():
        blob=history.road.fetch(base_url+r["rgb"])
        with Image.open(BytesIO(blob)) as im:
            assert sorted(im.size)==[720,1280]
            im.thumbnail((300,169))
            tmp=p.with_suffix(".part")
            im.convert("RGB").save(tmp,format="JPEG",quality=90)
            tmp.replace(p)
    return p


def prepare():
    rows=reserve()
    import concurrent.futures
    url=base()
    with ThreadPoolExecutor(16) as pool:
        fs=[pool.submit(thumb,r,url) for r in rows if r["panel"]=="screen"]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            f.result()
            if i%200==0:print("RGBs staged",i,"/",len(fs),flush=True)


def rank():
    import torch
    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
    torch.set_num_threads(4)
    net=mobilenet_v3_small(weights=None).cuda().eval()
    w=ROOT/"model_code/pretrained/mobilenet-v3-small/model.pth"
    net.load_state_dict(torch.load(w,map_location="cpu",weights_only=True))
    pre=MobileNet_V3_Small_Weights.DEFAULT.transforms()
    @torch.inference_mode()
    def features(paths):
        vectors=[]
        for start in range(0,len(paths),32):
            batch=[]
            for path in paths[start:start+32]:
                with Image.open(path) as img:batch.append(pre(img.convert("RGB")))
            v=net.avgpool(net.features(torch.stack(batch).cuda())).flatten(1)
            vectors.append(torch.nn.functional.normalize(v).cpu())
        return torch.cat(vectors)
    ref=history.previous
    ok=set(read(ref.R/"weather-lock.json")["codes"])
    yes,no=[],[]
    for r in read(ref.R/"screen.json")["rows"]:
        (yes if r["code"] in ok else no).append(ref.DATA/(r["id"]+"-thumb.jpg"))
    ok=set(read(history.io.OUT/"bdd-visual-lock.json")["ids"])
    yes.extend(ROOT/r["paths"][0] for r in read(history.io.OUT/"bdd-manifest.json")["rows"] if r["id"] in ok)
    unit=lambda a:torch.nn.functional.normalize(a.mean(0),dim=0)
    axis=unit(features(yes))-unit(features(no))
    rows=[r for r in reserve() if r["panel"]=="screen"]
    scores=(features([DATA/(r["id"]+"-thumb.jpg") for r in rows])@axis).tolist()
    order=sorted(range(len(rows)),key=lambda i:(-scores[i],rows[i]["id"]))
    ix=order[:256]+sorted(order[256:],key=lambda i:key("v112-screen-control/"+rows[i]["id"]))[:48]
    screened=[dict(rows[i],code=f"N{j:03d}",rank_score=scores[i]) for j,i in enumerate(ix,1)]
    put(HERE/"bdd-screen.json",dict(rows=screened,positive_templates=len(yes),negative_templates=len(no),weights_sha256=sha256(w.read_bytes()).hexdigest()),True)
    sheets(screened,False)


def full(r, url):
    p=DATA/(r["id"]+".jpg")
    if not p.exists():
        blob=history.road.fetch(url+r["rgb"])
        with Image.open(BytesIO(blob)) as im:assert sorted(im.size)==[720,1280]
        tmp=p.with_suffix(".part");tmp.write_bytes(blob);tmp.replace(p)
    return p


def sheets(rows,large):
    cols,n,(cw,ch)=(3,9,(498,303)) if large else (5,30,(258,164))
    url=base()
    for start in range(0,len(rows),n):
        sub=rows[start:start+n]
        canvas=Image.new("RGB",(cols*cw,((len(sub)+cols-1)//cols)*ch),"white")
        pen=ImageDraw.Draw(canvas)
        for i,r in enumerate(sub):
            p=full(r,url) if large else DATA/(r["id"]+"-thumb.jpg")
            with Image.open(p) as img:
                img.thumbnail((cw-3,ch-18));x=i%cols*cw;y=i//cols*ch;canvas.paste(img,(x,y+15))
            pen.text((x+2,y),r["code"], fill="black")
        canvas.save(HERE/(("bdd-enlarged" if large else "bdd-screen")+f"-{start//n+1:02d}.jpg"),quality=91)


def finalize():
    lock=read(HERE/"bdd-weather-decision.json")
    selected=[dict(r,panel="visible_fog") for r in read(HERE/"bdd-screen.json")["rows"] if r["code"] in lock["included"]]
    assert len(selected)==len(lock["included"]) and set(lock["included"])<=set(read(HERE/"bdd-shortlist.json")["codes"])
    selected += [r for r in reserve() if r["panel"]=="random"]
    url=base()
    with ThreadPoolExecutor(12) as pool:
        list(pool.map(lambda r:full(r,url),selected))
    def fingerprint(path):
        with Image.open(path) as im:
            rgb=np.asarray(im.convert("RGB").resize((32,18),Image.Resampling.BILINEAR),float)/255
            a=np.asarray(im.convert("L").resize((9,8),Image.Resampling.BILINEAR))
            return rgb, np.packbits(a[:,1:]>a[:,:-1]).tobytes()
    # The hash filter precedes the more expensive per-pixel comparison.
    known={}
    for _,p in old_rgb():
        try:
            rgb,dh=fingerprint(p)
        except OSError:
            continue
        known.setdefault(int.from_bytes(dh,"big"),[]).append((str(p),rgb))
    prior_files=sum(map(len,known.values()))
    kept,excluded=[],[]
    for r in selected:
        rgb,dh=fingerprint(DATA/(r["id"]+".jpg"));h=int.from_bytes(dh,"big")
        hits=[s for hh,items in known.items() if (hh^h).bit_count()<=2 for s,a in items if np.abs(a-rgb).mean()<.035]
        if hits:
            excluded.append(dict(id=r["id"],panel=r["panel"],near=hits));continue
        kept.append(r);known.setdefault(h,[]).append((r["id"],rgb))
    put(HERE/"bdd-eligible.json",dict(rows=kept,excluded=excluded,prior_rgb_files=prior_files,weather_sha256=sha256((HERE/"bdd-weather-decision.json").read_bytes()).hexdigest()),True)
    def stage(r):
        p=DATA/(r["id"]+".png")
        if not p.exists():
            blob=history.road.fetch(url+r["label"])
            with Image.open(BytesIO(blob)) as im, Image.open(DATA/(r["id"]+".jpg")) as source:
                assert im.size==source.size and set(np.unique(im))<=set(range(19))|{255}
            tmp=p.with_suffix(".maskpart");tmp.write_bytes(blob);tmp.replace(p)
        rgb=DATA/(r["id"]+".jpg")
        return dict(r,paths=[str(x.relative_to(ROOT)) for x in (rgb,p)],sha256=[sha256(x.read_bytes()).hexdigest() for x in (rgb,p)])
    with ThreadPoolExecutor(12) as pool:
        staged=list(pool.map(stage,kept))
    put(HERE/"bdd-evaluation.json",dict(rows=staged,eligible_sha256=sha256((HERE/"bdd-eligible.json").read_bytes()).hexdigest()),True)
    print("Eligible:",len(staged),"including fog",sum(r["panel"]=="visible_fog" for r in staged),"near copies", len(excluded),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("step",choices=("prepare","rank","enlarge","finalize"))
    mode=arg.parse_args().step
    if mode=="prepare":prepare()
    elif mode=="rank":rank()
    elif mode=="enlarge":
        codes=set(read(HERE/"bdd-shortlist.json")["codes"])
        sheets([r for r in read(HERE/"bdd-screen.json")["rows"] if r["code"] in codes],True)
    else:finalize()

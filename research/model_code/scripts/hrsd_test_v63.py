"""Pin a scene-balanced optical subset of the publisher's HRSD tests."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import re
from time import sleep

from PIL import Image

from scripts.fresh_real_v53 import groups
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor, long_edge
from scripts.satehaze_test_v59 import direct
from scripts.train_regional_guidance_v18 import atomic_write
from scripts.unreal_nh_v57 import retrieve


ROOT=REPO.parent/"benchmark_data/hrsd-test-v63"
MANIFEST=ROOT/"manifest.json"
SOURCE="https://github.com/xushouyi1/HSMD-Net/tree/caa69ab2c2098e2d2d9bef0a3885b98f07807f77"
FOLDERS={
    "dhid":("1P4gy7w9S0t8kXXMDINcvtIAICHwJ_f50","1Gsg6OSr8NTzBhxXGtKr0Q0soHWXU_aZa","1pS3kcnp0F63jSOcRqjAvrCruxl7mPc5j"),
    "lhid":("1JXIfoU2hEdh1RCkQ1JB8vJ7PVWCQ3Xq3","1Bkn--EYn-VCpo5s0ckyxLxiE6BBE6V7Z","1kfVTdCKS5Xt8aQdEKZG7Sqf8mYtMc7BA"),
}
N=40


def rank(label):
    return sha256(("hrsd-v63/"+label).encode()).digest()


def index(identity):
    page=retrieve("https://drive.google.com/embeddedfolderview?id="+identity).decode("utf-8")
    expression=r'<a[^>]+href="https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)/[^"]+"[^>]*>[\s\S]{0,700}?flip-entry-title">([^<]+)'
    found=[(name,key) for key,name in re.findall(expression,page) if re.fullmatch(r"\d[\w.]*\.jpg",name)]
    inventory=dict(found)
    if len(inventory)!=len(found):
        raise ValueError("A test-folder name appears more than once")
    return inventory


def take(branch, listed, hazy, target):
    text=retrieve(direct(listed))
    original=text.decode("utf-8-sig").split()
    hazy_idx,target_idx=index(hazy),index(target)
    if len(original)!=500 or len(set(original))!=500 or not set(original)<=hazy_idx.keys():
        raise ValueError(f"The look-up disagrees with the author {branch} test list")
    grouped=defaultdict(list)
    for path in original:
        grouped[path.split("_",1)[0]].append(path)
    chosen=sorted(grouped,key=lambda s:rank(branch+"/scene/"+s))[:N]
    if len(chosen)!=N:
        raise ValueError("Too few distinct author scene keys")
    out=[]
    for key in chosen:
        name=min(grouped[key],key=lambda s:rank(branch+"/variant/"+s))
        reference=key+".jpg"
        out.append(dict(id=branch+"/"+key,condition=branch,haze=(name,hazy_idx[name]),clean=(reference,target_idx[reference])))
    (ROOT/f"{branch}-list.txt").write_bytes(text)
    return out


def copy(task):
    row,role=task
    name,key=row[role]
    target=ROOT/row["condition"]/role/name
    if not target.exists():
        for i in range(3):
            try:
                data=retrieve(direct(key))
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                break
            except (OSError,ValueError):
                if i==2:
                    raise
                sleep(i+1)
        tmp=target.with_suffix(".pending")
        tmp.write_bytes(data)
        tmp.replace(target)
    with Image.open(target) as image:
        shape=list(image.size)
    return dict(path=target.relative_to(ROOT).as_posix(),identifier=key,SHA256=digest(target),size=shape)


def stage():
    if MANIFEST.exists():
        result=read(MANIFEST)
        if any(digest(ROOT/row[k]["path"])!=row[k]["SHA256"] for row in result["records"] for k in ("haze","clean")):
            raise ValueError("An HRSD test image changed")
        return result
    ROOT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as download:
        lists=[download.submit(take,key,*groups) for key,groups in FOLDERS.items()]
        cases=[case for task in lists for case in task.result()]
        for branch in FOLDERS:
            for role in ("haze","clean"):
                (ROOT/branch/role).mkdir(parents=True,exist_ok=True)
        jobs=[(case,k) for case in cases for k in ("haze","clean")]
        files={}
        for n,((row,k),meta) in enumerate(zip(jobs,download.map(copy,jobs),strict=True),1):
            files[row["id"],k]=meta
            if n%40==0:
                print(dict(files=n,total=len(jobs)),flush=True)
    records, snapshots=[],[]
    for item in cases:
        record=dict(item,**{k:files[item["id"],k] for k in ("haze","clean")})
        if record["haze"]["size"]!=record["clean"]["size"]:
            raise ValueError("The spatial target shape differs from its paired input")
        records.append(record)
        with Image.open(ROOT/record["clean"]["path"]) as handle:
            snapshots.append(long_edge(handle.convert("RGB"),256))
    clusters=groups(records,snapshots)
    for row in records:
        row["group"]=clusters[row["id"]]
    result=dict(source=SOURCE,counts=dict(Counter(r["condition"] for r in records)),records=records,
        groups=len(set(clusters.values())),lists={c:digest(ROOT/f"{c}-list.txt") for c in FOLDERS}, locked_utc=now(),
        rule="Sample 40 reference IDs per HRSD test subdomain by filename hash. In DHID choose one of the "
             "synthetic haze variants per reference by a second hash. Use the paired RGB published grids "
             "and merge near-duplicate clean references. Evaluation only.")
    atomic_write(MANIFEST,result)
    print(dict(counts=result["counts"],visual_groups=result["groups"]),flush=True)
    return result


def pairs(info):
    for row in info["records"]:
        items=[]
        for role in ("haze","clean"):
            with Image.open(ROOT/row[role]["path"]) as handle:
                items.append(image_tensor(handle.convert("RGB")))
        yield "hrsd_"+row["condition"],row["id"],row["group"],*items,None,row["condition"]


if __name__=="__main__":
    stage()

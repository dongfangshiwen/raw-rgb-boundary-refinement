"""Register the SateHaze1k RGB test lists published with HSMD-Net."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import re
from time import sleep

from PIL import Image

from scripts.fresh_real_v53 import groups
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.real_detail_data_v24 import image_tensor, long_edge
from scripts.train_regional_guidance_v18 import atomic_write
from scripts.unreal_nh_v57 import retrieve


ROOT = REPO.parent / "benchmark_data/satehaze-test-v59"
MANIFEST = ROOT / "manifest.json"
PUBLISHER = "https://github.com/xushouyi1/HSMD-Net/tree/caa69ab2c2098e2d2d9bef0a3885b98f07807f77"
BRANCHES = {
    "thin": ("1UKWbOA1uCSRH8VsE0Y5MOXNngTo-xL8e", "1zu6-IubQslpafrMJpiaKylZF9JcUoDVc", "1CPq7d0IApfXeufJn9d7-OLof8yac89B-"),
    "moderate": ("18z74H4DSuN1b0k2vtZJdNGky-DUSPRz7", "1EmI14Xw54PK_BTGDrQEsJdYXda1zfU5z", "1CU0JBP_iOSx7w7P064mXscv54nLT6Ck7"),
    "thick": ("16DBdAG8MabkCMJbO88-8kEHcNSHytIQg", "1S0zjptuoZdpJxuwq46P2tn0WisiLg0tJ", "18NJm_LC9xKkd-v9AfyGC-D8OVXon8RvN"),
}


def direct(identity):
    return f"https://drive.usercontent.google.com/download?id={identity}&export=download&confirm=t"


def directory(identity):
    base = 'https://drive.google.com/embeddedfolderview?id='
    html = retrieve(base+identity).decode("utf-8")
    expression = r'<a[^>]+href="https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)/[^"]+"[^>]*>[\s\S]{0,700}?flip-entry-title">([^<]+)'
    listing = {}
    for key, name in re.findall(expression, html):
        case = re.fullmatch(r"(\d+)(?:[-_][A-Za-z]+)?\.(?:png|jpg|jpeg|bmp|tif)", name, flags=re.I)
        if case:
            number = int(case[1])
            if number in listing:
                raise ValueError(f"Duplicate release tile number {name}")
            listing[number] = (name, key)
    return listing


def load_list(entry):
    condition, (list_id, haze_dir, clean_dir) = entry
    content = retrieve(direct(list_id))
    names = content.decode("utf-8-sig").split()
    ids = [int(re.fullmatch(r"(\d+)-inputs\.png", label)[1]) for label in names]
    if len(set(ids)) != len(ids) or len(ids) != 45:
        raise ValueError(f"Unexpected {condition} test list from the published repository")
    haze, clear = directory(haze_dir), directory(clean_dir)
    if any(i not in haze or i not in clear for i in ids):
        raise ValueError(f"Unpaired release images in {condition}")
    (ROOT / f"{condition}-list.txt").write_bytes(content)
    return condition, [dict(id=f"{condition}/{i}", number=i, condition=condition,
                            haze=haze[i], clean=clear[i]) for i in ids]


def fetch(task):
    row, role = task
    label, file_id = row[role]
    final = ROOT / row["condition"] / role / label
    if final.exists():
        data = final.read_bytes()
    else:
        for k in range(3):
            try:
                data = retrieve(direct(file_id))
                with Image.open(BytesIO(data)) as frame:
                    frame.verify()
                break
            except (OSError, ValueError):
                if k == 2:
                    raise
                sleep(1. + k)
        tmp = final.with_suffix(".pending")
        tmp.write_bytes(data)
        tmp.replace(final)
    with Image.open(BytesIO(data)) as frame:
        size = list(frame.size)
    return dict(path=final.relative_to(ROOT).as_posix(), file_id=file_id,
                SHA256=sha256(data).hexdigest(), bytes=len(data), size=size)


def stage():
    if MANIFEST.exists():
        result=read(MANIFEST)
        for row in result["records"]:
            for role in ("haze","clean"):
                if digest(ROOT/row[role]["path"])!=row[role]["SHA256"]:
                    raise ValueError("A SateHaze1k release file changed")
        return result
    ROOT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        cohorts = dict(executor.map(load_list,BRANCHES.items()))
        rows = [r for key in BRANCHES for r in cohorts[key]]
        for c in BRANCHES:
            for role in ("haze","clean"):
                (ROOT/c/role).mkdir(parents=True,exist_ok=True)
        tasks = [(r,role) for r in rows for role in ("haze","clean")]
        downloaded = {}
        for index,(task,value) in enumerate(zip(tasks,executor.map(fetch,tasks),strict=True),1):
            downloaded[task[0]["id"],task[1]]=value
            if index%45==0:
                print(dict(files=index,total=len(tasks)),flush=True)
    records, photos = [], []
    for row in rows:
        record = dict(row, **{r:downloaded[row["id"],r] for r in ("haze","clean")})
        if record["haze"]["size"]!=record["clean"]["size"]:
            raise ValueError(f"Image pair {row['id']} has different grids")
        records.append(record)
        with Image.open(ROOT/record["clean"]["path"]) as frame:
            photos.append(long_edge(frame.convert("RGB"),256))
    cohort=groups(records,photos)
    for row in records:
        row["group"]=cohort[row["id"]]
    result=dict(publication=PUBLISHER, ids=BRANCHES, counts=dict(Counter(r["condition"] for r in records)),
                grouped=len(set(cohort.values())), lists_sha256={c:digest(ROOT/f"{c}-list.txt") for c in BRANCHES},
                records=records,locked_utc=now(),
                rule="SateHaze1k paired RGB test files released by HSMD-Net's authors. Synthetic haze over optical imagery. "
                     "Use the published 45-file list per haze strength and group near-duplicate clear references across "
                     "the strengths. Evaluate only. Do not use the SAR or any development files.")
    atomic_write(MANIFEST,result)
    print(dict(pairs=len(records),groups=result["grouped"]),flush=True)
    return result


def samples(registration):
    for row in registration["records"]:
        crops = []
        for role in ("haze","clean"):
            with Image.open(ROOT/row[role]["path"]) as frame:
                crops.append(image_tensor(frame.convert("RGB")))
        yield "satehaze",row["id"],row["group"],*crops,None,row["condition"]


if __name__=="__main__":
    stage()

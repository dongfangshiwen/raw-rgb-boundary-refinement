"""Install verified public model files, or link verified copies from the research workspace."""
import argparse
from hashlib import file_digest
from pathlib import Path
import os
import shutil
import time
import urllib.request

MCT="https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/8e0b26c7732e2b0dc104f69c5ea30f913c878a89/"
MASK="https://huggingface.co/facebook/mask2former-swin-tiny-cityscapes-semantic/resolve/01a04783d1e2981b13cf5aace8cd87774012de27/"
SEG="https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024/resolve/ec86afeba68e656629ccf47e0c8d2902f964917b/"
FILES={
 "mct":(MCT,"official-dehazeformer-research/hf-author-demo",{
    "models/dehazeformer.py":"56712bf3bb949d5c4c5a068d8a301b8057179c828a5a1acfb813af373e8bf207",
    "saved_models/dehazeformer.pth":"225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304",
    "LICENSE":"86baa11328570d9f18851a9cbdd3ed03d9e9640a7f4365036aadde7fda170575"}),
 "mask2former":(MASK,"mask2former-road-v100",{
    "model.safetensors":"489e8dcd9b46f991ba0daa00a9dc3d780094168937dad3ff3a1c0dd884664754",
    "config.json":"3fec5bdc33a941fe7a9a06d8c8cd2e8cb64a09ff0f2b9f47de395274cfdbf80a",
    "preprocessor_config.json":"98ca5faae0c936507c0e6888d85a08ae8ed2e96a853787275e125c63a212e9b0"}),
 "segformer":(SEG,"segformer-b1-v42",{
    "pytorch_model.bin":"9a7c40538e41d92f93ad6501d54e6333833a1d96bda7b0906c7255240e927832",
    "config.json":"967a4a40a37783615f4e6767e6c2669ce8fae67411e6a8eda2426b0d9c536e23",
    "preprocessor_config.json":"697d92c83640ee381065f575ebdaef42d00d44483a827b92265026c0173dcbbf"})}
def verify(path,sha):
    if not path.is_file():return False
    with path.open("rb") as f:return file_digest(f,"sha256").hexdigest()==sha

def main():
    q=argparse.ArgumentParser();q.add_argument("--from-cache",type=Path)
    q.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"assets")
    q.add_argument("--backbone",choices=["mask2former","segformer","both"],default="mask2former")
    a=q.parse_args()
    for kind,(prefix,old,files) in FILES.items():
        if kind!="mct" and a.backbone not in ("both",kind):continue
        for name,sha in files.items():
            path=a.output/kind/name;path.parent.mkdir(parents=True,exist_ok=True)
            if verify(path,sha):continue
            if path.exists():raise ValueError(f"Existing file has a different hash: {path}")
            temp=path.with_suffix(path.suffix+".part")
            if a.from_cache:
                original=a.from_cache/old/name
                if not verify(original,sha):raise ValueError(f"Unverified source: {original}")
                try:os.link(original,temp)
                except OSError:shutil.copyfile(original,temp)
            else:
                for i in range(5):
                    try:
                        with urllib.request.urlopen(prefix+name+f"?download=true&try={i}",timeout=90) as inp,temp.open("wb") as out:
                            shutil.copyfileobj(inp,out)
                        if not verify(temp,sha):raise OSError("Incomplete published asset")
                        break
                    except OSError:
                        if i==4:raise
                        time.sleep(1+i)
            if not verify(temp,sha):raise ValueError(f"Hash failed: {name}")
            temp.replace(path)
    print("Verified public assets in",a.output)
if __name__=="__main__":main()

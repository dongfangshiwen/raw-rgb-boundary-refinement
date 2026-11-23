"""Assemble local research artifacts, then independently re-score the extracted package."""
import ast
from hashlib import sha256
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from scripts import score_path_v104 as utils

ROOT=utils.DEFAULT
REPO=Path(__file__).resolve().parents[1]

def local_sources():
    queue=[*REPO.glob("scripts/*v104.py"),*REPO.glob("scripts/*v105.py")];keep=set()
    if (REPO/"scripts/__init__.py").exists():queue.append(REPO/"scripts/__init__.py")
    def resolve(name):
        if name.split(".")[0] not in ("scripts","dehaze_seg"):return
        file=REPO/Path(*name.split("."))
        for p in (file.with_suffix(".py"),file/"__init__.py"):
            if p.is_file() and p not in keep:queue.append(p)
    while queue:
        path=queue.pop()
        if path in keep:continue
        keep.add(path);tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):
                for a in n.names:resolve(a.name)
            elif isinstance(n,ast.ImportFrom):
                name=n.module or ""
                if n.level:
                    package=path.relative_to(REPO).parts[:-1]
                    name=".".join((*package[:len(package)-n.level+1],name)).rstrip(".")
                resolve(name)
                for a in n.names:resolve(name+"."+a.name)
    return sorted(keep)

def package():
    ext=ROOT/"manuscript-extension";artifacts={}
    for d in (ROOT,ext):
        for path in [*d.glob("*.json"),*d.glob("*.pth"),*d.glob("counts/*/*.npz"),*d.glob("road-recheck/*/*.npz"),*d.glob("examples/**/*"),*d.glob("inference/*.pth")]:
            if path.is_file() and path.name!="package-verification.json":artifacts[path.relative_to(ROOT).as_posix()]=path
    for path in [ROOT/"boundary-effects.png",ROOT/"native-prediction.png",ext/"overlap-01.jpg"]:artifacts[path.relative_to(ROOT).as_posix()]=path
    for path in local_sources():artifacts[path.relative_to(REPO).as_posix()]=path
    readme="""# Frozen color-guided boundary decoding

The two experiments use MUSES fog source supervision and frozen external segmenters. Stage one fits the Gaussian-family controls and stage two adds one coefficient to the exact archival manuscript decoder. Ground-truth masks and third-party backbone weights are kept in their original datasets and workspace locations.

## Recalculate the results

Extract this archive, change into the extracted directory and use Python with NumPy. The following commands rebuild all raw and view-audited statistics from 559 per-frame evaluation packets. Seventy packets contain additional method outputs for already counted driving frames. Each packet contains integer confusion matrices indexed by target, method, native region and source seed. The JSON files specify their ordering. The archive also contains the native reference example, class-level effects, split manifests, source and weight hashes, and three fitted inference checkpoints.

```bash
python -m scripts.score_path_v104 --root .
python -m scripts.score_anchor_v105 --root manuscript-extension
python -m scripts.score_anchor_v105 --root manuscript-extension --audit-views
python -m scripts.class_influence_v105 --root manuscript-extension
python -m scripts.score_road_recheck_v105 --root manuscript-extension
```

## Apply the frozen decoder

Use a CUDA-enabled PyTorch installation. The image must be the aligned float RGB used to construct the frozen scores. Supply it before channel normalization. The coarse scores are native backbone scores on its stride-four working lattice. Select the original source seed explicitly. The function returns native integer class IDs for one image.

```python
from scripts.apply_path_v105 import predict

labels = predict(coarse_logits, working_rgb, (native_height, native_width),
                 "manuscript-extension/inference/path-101.pth",
                 road_manuscript=True)
```

Use the road manuscript route for the two Cityscapes models. Use road_manuscript=False for the ADE20K 150-class probe. The three source seeds are 101, 211 and 307. The final audited tables are in manuscript-extension/results-view-audited.json. Historical preprocessing and external checkpoints are dependencies for reproducing the complete source training or frozen target backbone inference. The code references their existing project paths and original releases. The verified standalone operations are native decoding from provided scores and CPU reconstruction of the reported metrics.

## Rerun in the original project workspace

The experiment scripts live under research/model_code/scripts in the public repository. Stage benchmark_data alongside model_code and experiment_exports under research/. Use the CUDA interpreter in the project's .venv. The source sampler and archival manuscript weights depend on earlier project assets. Check training.json and manuscript-extension/training.json for their source hashes, and the population manifests for the pinned dataset provenance.

Stage one training uses scripts.train_path_v104 and target inference uses scripts.eval_path_v104. Stage two uses scripts.train_anchor_v105, scripts.eval_anchor_v105, and scripts.check_anchor_v105. The coverage report uses every scored frame. Percent intervals resample acquisition groups and average the three seed scores within each draw. Unknown labels are ignored and absent fixed classes contribute zero; GT-supported sensitivity is included separately.
"""
    runtime=dict(python=sys.version,os=platform.platform(),packages={k:version(k) for k in ("torch","torchvision","transformers","numpy","pillow","scipy","matplotlib")},
                 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip())
    tmp=ROOT/"replication.zip.part";dest=ROOT/"replication.zip"
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name,path in sorted(artifacts.items()):z.write(path,name)
        if "scripts/__init__.py" not in artifacts:z.writestr("scripts/__init__.py","")
        z.writestr("README.md",readme);z.writestr("runtime.json",json.dumps(runtime,indent=2))
        html=(ROOT/"index.html").read_text(encoding="utf-8").replace('<a href="replication.zip">实验复算包与推理权重</a>　',"")
        z.writestr("index.html",html)
    with zipfile.ZipFile(tmp) as z:
        hashes="".join(sha256(z.read(n)).hexdigest()+"  "+n+"\n" for n in z.namelist())
    with zipfile.ZipFile(tmp,"a",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("SHA256SUMS.txt",hashes)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None;count=len(z.namelist());allnames=set(z.namelist())
        for line in z.read("SHA256SUMS.txt").decode().splitlines():
            expected,n=line.split("  ",1);assert sha256(z.read(n)).hexdigest()==expected
        with tempfile.TemporaryDirectory(prefix="audit-",dir=ROOT) as folder:
            target=Path(folder).resolve();assert target.is_relative_to(ROOT.resolve())
            z.extractall(target)
            env={**os.environ,"OPENBLAS_NUM_THREADS":"2","PYTHONPATH":""}
            commands=[("score_path_v104",".",()),("score_anchor_v105","manuscript-extension",()),
                      ("score_anchor_v105","manuscript-extension",("--audit-views",)),("class_influence_v105","manuscript-extension",()),
                      ("score_road_recheck_v105","manuscript-extension",())]
            for script,root,options in commands:
                subprocess.run([sys.executable,"-m","scripts."+script,"--root",root,*options],cwd=target,env=env,check=True)
            for f in ("results.json","manuscript-extension/results.json","manuscript-extension/results-view-audited.json","manuscript-extension/class-influence.json","manuscript-extension/road-recheck-results.json"):
                assert utils.read(ROOT/f)==utils.read(target/f),f
            from html.parser import HTMLParser
            class Links(HTMLParser):
                def handle_starttag(self,tag,attrs):
                    for k,v in attrs:
                        if k in ("href","src") and not v.startswith("https://"):
                            assert v in allnames,v
            Links().feed((target/"index.html").read_text(encoding="utf-8"))
    tmp.replace(dest);utils.write(ROOT/"package-verification.json",dict(files=count,bytes=dest.stat().st_size,sha256=sha256(dest.read_bytes()).hexdigest(),
              statistical_outputs_equal=5,packets=sum(k.endswith(".npz") for k in allnames),hashes_and_local_report_links_verified=True))
    print("Verified",dest,count,dest.stat().st_size,flush=True)

if __name__=="__main__":package()

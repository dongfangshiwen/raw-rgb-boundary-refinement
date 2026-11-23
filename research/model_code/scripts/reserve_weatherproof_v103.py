"""Reserve WeatherProof camera groups and prepare blind input-only screening sheets."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path, PurePosixPath
from hashlib import sha256
import argparse
import re
import struct
import zlib
import numpy as np
from PIL import Image, ImageDraw
from scripts import weatherproof_release_v103 as archive

OUT,DATA=archive.OUT,archive.DATA
def camera(name):
    return re.sub(r"(?:[_-]\d+)+$","",name)
def rank(name):
    return sha256(("wp103/"+name).encode()).hexdigest()
def reservation():
    path=OUT/"wp-selection.json"
    if path.exists():return archive.read(path)
    src=archive.index()
    scenes=defaultdict(list)
    for key in src:
        if key.endswith("_degraded.png"):
            split,scene,file=key.split("/")
            stem=file.removesuffix("_degraded.png")
            if all(f"{split}/{scene}/{stem}_{s}.png" in src for s in ("gt-clean","gt-intern")):
                scenes[split,scene].append(stem)
    val=sorted(s for split,s in scenes if split=="test_scenes")
    assert len(val)==38
    forbidden={camera(s) for s in val}
    train=defaultdict(list)
    for split,s in scenes:
        if split=="train_scenes" and camera(s) not in forbidden:train[camera(s)].append(s)
    chosen=[("test_scenes",s) for s in val]+[("train_scenes",min(train[g],key=rank)) for g in sorted(train,key=rank)[:72]]
    rows=[]
    for i,(sp,sc) in enumerate(chosen,1):
        seq=sorted(scenes[sp,sc],key=int)
        ids=[seq[(2*j+1)*len(seq)//6] for j in range(3)]
        assert len(set(ids))==3
        rows.append(dict(code=f"W{i:03d}",scene=sc,group=camera(sc),split=sp,triplets=[f"{sp}/{sc}/{k}" for k in ids]))
    note=dict(source="https://github.com/UCLA-VMG/WeatherProof",archive_id=archive.ID,archive_size=archive.SIZE,
              design=archive.sha((OUT/"protocol.json").read_bytes()),rows=rows)
    archive.write(path,note)
    return note

def payload(prefix,types,src):
    keys=[prefix+"_"+k+".png" for k in types]
    paths=[DATA/"members"/Path(*PurePosixPath(k).parts) for k in keys]
    if not all(p.exists() for p in paths):
        a=min(src[k][0] for k in keys)
        b=max(src[k][0]+src[k][1]+30+len(k.encode())+100 for k in keys)
        if b-a>5000000:
            for k in keys:archive.member(k,src[k])
        else:
            block=archive.get_range(a,min(b,archive.SIZE)-1)
            for k,p in zip(keys,paths):
                off,size,length,crc,method=src[k];o=off-a
                assert block[o:o+4]==b"PK\x03\x04"
                n,x=struct.unpack_from("<HH",block,o+26)
                assert block[o+30:o+30+n].decode()==k
                packed=block[o+30+n+x:o+30+n+x+size]
                raw=zlib.decompress(packed,-15) if method==8 else packed
                assert len(raw)==length and zlib.crc32(raw)==crc,k
                p.parent.mkdir(exist_ok=True,parents=True)
                temp=p.with_suffix(".part");temp.write_bytes(raw);temp.replace(p)
    for k,p in zip(keys,paths):
        assert zlib.crc32(p.read_bytes())==src[k][3]
    return paths

def prepare(action):
    note=reservation()
    src=archive.index()
    rows=note["rows"]
    if action=="screen":
        tasks=[r["triplets"][1] for r in rows]
        work=lambda pre:payload(pre,("degraded","gt-clean"),src)
    else:
        ratings=archive.read(OUT/"wp-weather-lock.json")["ratings"]
        assert set(ratings)=={r["code"] for r in rows}
        rows=[r for r in rows if r["split"]=="test_scenes" or ratings[r["code"]] in ("fog","mixed")]
        tasks=[pre for r in rows for pre in r["triplets"]]
        work=lambda pre:payload(pre,("degraded","gt-clean","gt-intern"),src)
    bank={}
    with ThreadPoolExecutor(6) as pool:
        for i,(pre,files) in enumerate(zip(tasks,pool.map(work,tasks)),1):
            bank[pre]=files
            if i%25==0:print(action,i,len(tasks),flush=True)
    if action=="screen":
        target=OUT/"screen";target.mkdir(exist_ok=True)
        for at in range(0,len(rows),12):
            sheet=Image.new("RGB",(1110,688),"white");d=ImageDraw.Draw(sheet)
            for j,r in enumerate(rows[at:at+12]):
                x=j%3*370;y=j//3*172
                for k,p in enumerate(bank[r["triplets"][1]]):
                    with Image.open(p) as im:
                        im=im.convert("RGB");im.thumbnail((180,139))
                        sheet.paste(im,(x+k*183,y+17))
                d.text((x+3,y+1),r["code"]+"   adverse  |  reference",fill="black")
            sheet.save(target/f"pairs-{at//12+1:02d}.jpg",quality=90)
        print("Blind screening ready",len(rows),len(set(r['group'] for r in rows)),flush=True)
    else:
        manifest=[]
        for r in rows:
            for pre in r["triplets"]:
                files=bank[pre]
                with Image.open(files[0]) as a,Image.open(files[1]) as clear,Image.open(files[2]) as label:
                    assert a.size==clear.size==label.size
                    ids=np.asarray(label)
                    if ids.ndim==3:
                        assert np.all(ids[:,:,0]==ids[:,:,1]) and np.all(ids[:,:,0]==ids[:,:,2])
                        ids=ids[:,:,0]
                    assert ids.ndim==2 and set(np.unique(ids))<=set(range(10))|{255},(pre,np.unique(ids))
                manifest.append(dict(id=pre,cohort="wp",group=r["group"],scene=r["scene"],code=r["code"],split=r["split"],
                        weather=ratings[r["code"]],paths=[p.relative_to(archive.WORK).as_posix() for p in files],
                        sha256=[archive.sha(p.read_bytes()) for p in files]))
        archive.write(OUT/"wp-manifest.json",dict(rows=manifest,selection=archive.sha((OUT/"wp-selection.json").read_bytes()),
                     weather=archive.sha((OUT/"wp-weather-lock.json").read_bytes())))
        print("WeatherProof triplets staged",len(manifest),flush=True)
if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("action",choices=["screen","stage"])
    prepare(cli.parse_args().action)

"""Read selected official WeatherProof ZIP members by validated HTTP byte ranges."""
from boundary_refinement.paths import WORKSPACE
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, RawIOBase
from pathlib import Path, PurePosixPath
from threading import local
from hashlib import sha256
import argparse
import json
import struct
import time
import zlib
import zipfile
import requests
from PIL import Image

WORK=WORKSPACE
DATA=WORK/"benchmark_data/weatherproof-v103"
OUT=WORK/"experiment_exports/fog-boundary-validation-v103"
URL="https://drive.usercontent.google.com/download"
ID="1tAZE8uo5wYvJYPZxKc7Mo-ye3D3-3Ld-"
SIZE=60857898073
LOCAL=local()
def write(path, data):
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix(path.suffix+".part")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def sha(data):
    return sha256(data).hexdigest()
def get_range(a,b):
    if not hasattr(LOCAL,"network"):
        LOCAL.network=requests.Session()
    for attempt in range(7):
        try:
            params={"export":"download","confirm":"t","id":ID,"range_part":str(a)+"-"+str(b)}
            with LOCAL.network.get(URL,params=params,headers={"Range":f"bytes={a}-{b}","Accept-Encoding":"identity"},
                                   timeout=(9,48),stream=True) as response:
                if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {a}-{b}/{SIZE}":
                    raise IOError(f"Unexpected range {response.status_code}, {response.headers.get('Content-Range')}")
                data=response.content
                if len(data)!=b-a+1:
                    raise IOError("Truncated archive range")
                return data
        except (requests.RequestException,IOError):
            if attempt==6:
                raise
            time.sleep(min(1+attempt,4))

class Remote(RawIOBase):
    def __init__(self):
        self.pos=0
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset+(0 if whence==0 else self.pos if whence==1 else SIZE)
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else SIZE-self.pos,SIZE-self.pos)
        if n<=0:return b""
        start=self.pos
        self.pos+=n
        if n<=2**20:return get_range(start,self.pos-1)
        pieces=[(a,min(a+2**20,self.pos)-1) for a in range(start,self.pos,2**20)]
        with ThreadPoolExecutor(4) as pool:
            return b"".join(pool.map(lambda ab:get_range(*ab),pieces))

def index():
    dest=DATA/"zip-index.json"
    if dest.exists():return read(dest)
    with zipfile.ZipFile(Remote()) as src:
        files={i.filename:[i.header_offset,i.compress_size,i.file_size,i.CRC,i.compress_type]
               for i in src.infolist() if not i.is_dir() and "__MACOSX/" not in i.filename}
    assert len(files)>1000
    write(dest,files)
    print("Directory entries",len(files),"sample",list(files)[:10],flush=True)
    return files

def member(name,card):
    offset,compressed,length,crc,method=card
    dest=DATA/"members"/Path(*PurePosixPath(name).parts)
    assert dest.resolve().is_relative_to((DATA/"members").resolve())
    def valid(raw):
        return len(raw)==length and zlib.crc32(raw)==crc
    if dest.is_file():
        raw=dest.read_bytes()
        if not valid(raw):raise ValueError(f"Local CRC mismatch {name}")
        return raw
    # Include enough spare bytes for the local filename and ordinary ZIP extra fields.
    block=get_range(offset,min(SIZE,offset+30+len(name.encode())+compressed+1024)-1)
    assert block[:4]==b"PK\x03\x04"
    n,x=struct.unpack_from("<HH",block,26)
    header=30+n+x
    assert block[30:30+n].decode()==name
    if len(block)<header+compressed:
        packed=get_range(offset+header,offset+header+compressed-1)
    else:
        packed=block[header:header+compressed]
    assert method in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)
    raw=zlib.decompress(packed,-15) if method==zipfile.ZIP_DEFLATED else packed
    assert valid(raw),name
    dest.parent.mkdir(exist_ok=True,parents=True)
    temp=dest.with_suffix(dest.suffix+".part")
    temp.write_bytes(raw);temp.replace(dest)
    return raw

if __name__=="__main__":
    files=index()
    from collections import Counter
    print(Counter(k.split("/")[0] for k in files).most_common(25))
    meta=[k for k in files if PurePosixPath(k).suffix.lower() in (".txt",".csv",".json",".md",".yaml")]
    print("Metadata",len(meta),meta[:40])
    for k in meta:
        if "/" not in k and files[k][2]<500000:
            print(k,member(k,files[k])[:1500],flush=True)

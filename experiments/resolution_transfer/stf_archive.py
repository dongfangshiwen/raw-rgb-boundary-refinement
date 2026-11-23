"""Read selected members from the public SemanticSTF release linked by its authors."""
from concurrent.futures import ThreadPoolExecutor
from io import RawIOBase
from pathlib import Path, PurePosixPath
from threading import local
import argparse
import struct
import time
import zipfile
import zlib
import requests
from experiments.resolution_transfer.cohort import ROOT, OUT, read, write

REV="e6d79cb635464e6d98b269f9cf9649f5fc7096a4"
SIZE=6957328238
URL=f"https://huggingface.co/datasets/AR-X/SemanticSTF/resolve/{REV}/SemanticSTF.zip"
ARCHIVE_OID="636a3c36c25fafc1a3d84e871c21c48c35336bd6fddeea1099ae346badc40d6a"
DATA=ROOT/"benchmark_data/semanticstf-transfer-2026-11-21"
THREAD=local()


def chunk(start,end):
    if not hasattr(THREAD,"net"):THREAD.net=requests.Session()
    for attempt in range(6):
        try:
            with THREAD.net.get(URL,params={"download":"true","range_part":f"{start}-{end}"},
                headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"},timeout=(7,35),stream=True) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get("Content-Range")==f"bytes {start}-{end}/{SIZE}"
                data=r.content
                if len(data)!=end-start+1:raise IOError("Range size mismatch")
                return data
        except (OSError,requests.RequestException,AssertionError):
            if attempt==5:raise
            time.sleep(min(4,attempt+1))


class Directory(RawIOBase):
    def __init__(self):self.offset=0
    def tell(self):return self.offset
    def seekable(self):return True
    def seek(self,off,whence=0):
        self.offset=off+(0 if whence==0 else self.offset if whence==1 else SIZE)
        return self.offset
    def read(self,count=-1):
        count=min(count if count>=0 else SIZE,SIZE-self.offset)
        if count<=0:return b""
        assert count<8*1024*1024
        a=self.offset;self.offset+=count
        return chunk(a,self.offset-1)


def index():
    DATA.mkdir(exist_ok=True)
    out=DATA/"index.json"
    if not out.is_file():
        with zipfile.ZipFile(Directory()) as arc:
            contents={i.filename:[i.header_offset,i.compress_size,i.file_size,i.CRC,i.compress_type]
                      for i in arc.infolist() if not i.is_dir() and "__MACOSX/" not in i.filename}
        write(out,dict(repository="AR-X/SemanticSTF",revision=REV,reported_archive_sha256=ARCHIVE_OID,
                       directory_only=True,members=contents),True)
    return read(out)["members"]


def member(name,book):
    off,compressed,size,crc,method=book[name]
    path=DATA/"members"/Path(*PurePosixPath(name).parts)
    assert path.resolve().is_relative_to((DATA/"members").resolve())
    if path.is_file():
        raw=path.read_bytes()
    else:
        blob=chunk(off,min(SIZE,off+30+len(name.encode())+compressed+1024)-1)
        assert blob[:4]==b"PK\x03\x04"
        n,m=struct.unpack_from("<HH",blob,26);header=30+n+m
        assert blob[30:30+n].decode()==name
        data=blob[header:header+compressed]
        if len(data)<compressed:data=chunk(off+header,off+header+compressed-1)
        if method==zipfile.ZIP_STORED:raw=data
        elif method==zipfile.ZIP_DEFLATED:raw=zlib.decompress(data,-15)
        else:raise ValueError("Unsupported ZIP method")
        assert len(raw)==size and zlib.crc32(raw)==crc
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".partial")
        tmp.write_bytes(raw);tmp.replace(path)
    assert len(raw)==size and zlib.crc32(raw)==crc
    return path


if __name__=="__main__":
    bk=index()
    metadata=[p for p in bk if p.lower().endswith((".txt",".yaml",".json",".md",".yml"))]
    print("Archive count",len(bk),"metadata",len(metadata),metadata[:40],flush=True)
    small=[n for n in metadata if bk[n][2]<500000]
    with ThreadPoolExecutor(8) as t:
        for path in t.map(lambda n:member(n,bk),small):
            print(path.name,path.stat().st_size,flush=True)

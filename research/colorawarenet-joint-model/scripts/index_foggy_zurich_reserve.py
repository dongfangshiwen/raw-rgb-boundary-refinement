"""Index the official Foggy Zurich ZIP with byte ranges, without opening images."""
from collections import Counter
from io import SEEK_CUR, SEEK_END, SEEK_SET
from pathlib import PurePosixPath
import time
import zipfile

import requests
from urllib3.exceptions import HTTPError as RangeHTTPError

from scripts.projective_ledger_v12 import REPO, digest, now, write


URL = 'https://data.vision.ee.ethz.ch/csakarid/shared/Model_adaptation_SFSU_dense/Downloads/Foggy_Zurich.zip'
SOURCE = 'https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/'
OFFICIAL_MD5 = '4e711868f6b6fbf30331e7986048022a'
OUT = REPO/'results/foggy-zurich-reservation'
SIZE = 8985298224
BLOCK = 1024*1024


class ZipByRange:
    def __init__(self, session, url=URL, size=SIZE):
        self.session = session
        self.url = url
        self.size = size
        self.pos = 0
        self.etag = None
        self.received = 0
        self.cache = {}

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def seek(self, offset, whence=SEEK_SET):
        bases = {SEEK_SET:0,SEEK_CUR:self.pos,SEEK_END:self.size}
        new = bases[whence]+offset
        if new<0:raise OSError('Negative stream offset')
        self.pos = new
        return new

    def block(self, start):
        for attempt in range(4):
            try:
                return self.request_block(start)
            except (requests.exceptions.RequestException, RangeHTTPError) as error:
                if attempt==3:raise
                print(f'Retrying official range {start}: {type(error).__name__}',flush=True)
                time.sleep(1+attempt)

    def request_block(self, start):
        end = min(self.size,start+BLOCK)-1
        headers = {'Accept-Encoding':'identity','Range':f'bytes={start}-{end}'}
        if self.etag:headers['If-Range']=self.etag
        with self.session.get(self.url, headers=headers, stream=True, timeout=(10,22)) as response:
            if response.status_code in (408,429,502,503,504):response.raise_for_status()
            expected = f'bytes {start}-{end}/{self.size}'
            if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                raise OSError(f'Expected {expected}, got status {response.status_code} and {response.headers.get("Content-Range")}')
            tag = response.headers.get('ETag')
            if not tag or (self.etag and self.etag!=tag):raise OSError('Official ZIP revision changed')
            self.etag = tag
            raw = response.raw.read(end-start+2)
            if len(raw)!=end-start+1:raise OSError('Incomplete official ZIP block')
        self.received += len(raw)
        if len(self.cache)>=8:self.cache.pop(next(iter(self.cache)))
        self.cache[start]=raw
        return raw

    def read(self, size=-1):
        if self.pos>=self.size:return b''
        count=min(self.size-self.pos,size) if size>=0 else self.size-self.pos
        if count>16*BLOCK:raise OSError('Refuse bulk ZIP read')
        output = bytearray()
        while count:
            start=(self.pos//BLOCK)*BLOCK
            raw=self.cache.get(start)
            if raw is None:raw=self.block(start)
            local=self.pos-start;step=min(count,len(raw)-local)
            output.extend(raw[local:local+step]);self.pos+=step;count-=step
        return bytes(output)


def contents(archive):
    return [p for p in archive.infolist() if not p.is_dir() and
            not any(s.startswith(('.', '__MACOSX')) for s in PurePosixPath(p.filename).parts)]


def main():
    out=OUT/'index.json'
    if out.exists():raise ValueError('Preserve inspected revision and index')
    with requests.Session() as sess:
        file=ZipByRange(sess)
        with zipfile.ZipFile(file) as archive:
            rows=contents(archive)
            directories=Counter(PurePosixPath(r.filename).parent.as_posix() for r in rows)
            manifest=[dict(path=p.filename,bytes=p.file_size,compressed_bytes=p.compress_size,CRC32=f'{p.CRC:08x}') for p in rows]
        write(out,dict(source=SOURCE,official_ZIP=URL,official_bytes=SIZE,official_MD5_documented_not_recomputed=OFFICIAL_MD5,
            ETag=file.etag,network_bytes=file.received,members=manifest,directory_counts=directories,
            decoded_images=0,new_test_predictions=0,scope='Central directory only; no member bytes or image evidence checked.',finished_utc=now()))
    print(dict(entries=len(rows),top_path_counts=directories.most_common(22),index_SHA256=digest(out),network_bytes=file.received))


if __name__=='__main__':main()

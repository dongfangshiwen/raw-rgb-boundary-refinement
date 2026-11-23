"""Replay the archived numerical checks for a table in the revised manuscript."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from manuscript_tables import show_verified_cells, verify_bdd_selection

ROOT = Path(__file__).resolve().parent
JOBS = {
    'timing': ('S4', 'inference-costs/replay.py'),
    'drives': ('S1', 'recompute_tables.py'),
    'kernels': ('S4', 'kernel-ablation/replay.py'),
    'four-way': ('S4', 'four-decoder-fog/replay.py'),
    'bdd': ('S2', 'rescore.py'),
    'transfer': ('S3', 'recalculate.py'),
    'resolution': ('S4', 'replay.py'),
    'source': ('S1', 'check_supplement.py'),
    'photos': ('S2', 'rescore_photographs.py'),
}
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def unpack(archive, destination):
    destination.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive) as z:
        for name in z.namelist():
            resolved=(destination/name).resolve()
            if destination.resolve() not in resolved.parents and resolved!=destination.resolve():
                raise ValueError('Invalid archive member '+name)
        z.extractall(destination)

def run_job(job,folder):
    group, filename=JOBS[job]
    args=[]
    if job=='source':
        nested=folder/'source-diagnostics'/'Original supplementary archive.zip'
        with ZipFile(nested) as z:
            program=folder/'check_supplement.py'
            program.write_bytes(z.read(filename))
        args=[str(nested)]
    else:
        program=folder/filename
    # rescore.py writes a fresh result next to its inputs in the scratch directory.
    original=None
    if job=='bdd':
        original=json.loads((folder/'semantic-results.json').read_text('utf8'))
    # The original numerical checks use assertions even when the caller enables optimization.
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONOPTIMIZE='0')
    result=subprocess.run([sys.executable,str(program),*args],cwd=program.parent,env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf8',errors='replace')
    if result.returncode:
        raise RuntimeError(f'{group}/{filename} failed\n'+result.stdout[-1500:]+'\n'+result.stderr[-2500:])
    if original is not None:
        current=json.loads((folder/'semantic-results.json').read_text('utf8'))
        if current!=original:
            raise ValueError('The regenerated S2 segmentation results differ from the released file.')
    print(f'  Verified {group}/{filename}',flush=True)

def main():
    with (ROOT/'Table guide.csv').open(encoding='utf-8-sig',newline='') as f:
        guide={int(row['Table']):row for row in csv.DictReader(f)}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supplement', type=Path, default=ROOT,
                        help='Directory containing archives/ from the extracted publication supplement')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--table',nargs='+',type=int,choices=sorted(guide),metavar='N',
                       help='Current manuscript table number(s), for example --table 4 6 9 10 14')
    group.add_argument('--all',action='store_true',help='Run all registered replay groups once')
    group.add_argument('--list',action='store_true',help='List the current manuscript tables')
    args=parser.parse_args()
    if args.list:
        for i,row in guide.items():print(f'{i:2d}  {row["Subject"]}')
        return
    selected=list(guide) if args.all else list(dict.fromkeys(args.table))
    jobs=list(dict.fromkeys(j for i in selected for j in guide[i]['Checks'].split() if j))
    for i in selected:
        row=guide[i]
        print(f'Table {i}. {row["Subject"]}\n  {row["Model or weights"]}\n  Records: {row["Evidence records"]}')
    if not jobs:
        print('This table lists model roles. It has no numerical endpoint.')
        return
    hashes=json.loads((ROOT/'Archive checksums.json').read_text('utf8'))
    with tempfile.TemporaryDirectory(prefix='boundary-table-replay-') as td:
        work=Path(td)
        needed=dict.fromkeys(JOBS[j][0] for j in jobs)
        for g in needed:
            name=f'Supplementary Data {g}.zip'
            src=args.supplement.resolve()/'archives'/name
            if not src.is_file():
                parser.error(f'Missing {name}. Extract the outer supplement and pass its directory with --supplement.')
            if file_hash(src)!=hashes[name]:raise ValueError('Archive checksum mismatch: '+name)
            unpack(src,work/g)
        for job in jobs:run_job(job,work/JOBS[job][0])
        if 6 in selected:
            verify_bdd_selection(work/'S2')
        show_verified_cells(work, ROOT/'Published table cells.csv', selected)
    print('Completed the archived numerical checks for the selected tables.')

if __name__=='__main__':
    main()

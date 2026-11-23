"""CPU-only evidence arithmetic and file integrity, independent of CUDA training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'results/projective-joint-v12'
CORE=('dense','free_query','projective','visibility','full','mask2former','dehazeformer_pair','dehazeseg')
CONDITIONS=('original','fog055','fog030','occluded_fog055')


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def cm_metrics(matrix):
    a=np.asarray(matrix,dtype=np.float64)
    if a.shape!=(2,2) or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid binary confusion matrix')
    den=a.sum(0)+a.sum(1)-np.diag(a);iou=np.divide(np.diag(a),den,out=np.zeros(2),where=den>0)
    return dict(miou=float(iou.mean()),foreground_iou=float(iou[1]),fpr=float(a[0,1]/max(a[0].sum(),1)),
        precision=float(a[1,1]/max(a[:,1].sum(),1)),recall=float(a[1,1]/max(a[1].sum(),1)))


def pooled(rows):
    result=dict(images=len(rows))
    for field,label in (('confusion_matrix','canonical'),('native_confusion_matrix','native')):
        matrix=np.asarray([r[field] for r in rows],dtype=np.int64).sum(0)
        result[label]=dict(confusion_matrix=matrix.tolist(),**cm_metrics(matrix))
    b=np.asarray([r['boundary_counts'] for r in rows],dtype=np.int64).sum(0)
    p=b[0]/max(b[1],1);r=b[2]/max(b[3],1)
    result.update(boundary_counts=b.tolist(),boundary_f1=float(2*p*r/max(p+r,1e-12)))
    if 'occlusion_pixels' in rows[0]:
        pixels=sum(r['occlusion_pixels'] for r in rows);fp=sum(r['occlusion_predicted_foreground'] for r in rows)
        result.update(occlusion_pixels=pixels,occlusion_predicted_foreground=fp,occlusion_fpr=fp/max(pixels,1))
    return result


def historical_inventory():
    """Read-only checks of 243 prior selected files, core code, data, archives."""
    inventory={};weights={}
    def add(path,sha):
        path=str(Path(path).resolve())
        if path in inventory and inventory[path]!=sha:raise ValueError('Conflicting historic hash:'+path)
        inventory[path]=sha
    core=read(REPO/'results/architecture-v3b/architecture_plan.json')['code_sha256']
    for p,sha in core.items():add(REPO/p,sha)
    v7=read(REPO/'results/density-shape-v7b/combined_training_lock.json')
    for p,sha in v7['frozen_source_hashes'].items():add(REPO/p,sha)
    for r in v7['trained_checkpoints'].values():weights[r['path']]=r['sha256']
    for name in ('parallel-joint-v8','partial-sharing-v8b','coefficient-joint-v9','background-followup-v9'):
        p=read(REPO/'results'/name/'training_lock.json')
        for r in p.get('jobs',p.get('models')):weights[r.get('checkpoint',r.get('path'))]=r.get('sha256',r.get('checkpoint_sha256'))
    for name in ('corridor-visibility-v10','geometry-controls-v10b','task-controls-v10c'):
        root=REPO/'results'/name
        for p,sha in read(root/'training_plan.json')['source_sha256'].items():add(REPO/p,sha)
        for path in (root/'models').rglob('completed.json'):weights[str(path.with_name('best.pth').resolve())]=read(path)['checkpoint_sha256']
    if len(weights)!=184:raise ValueError('Expected184preV11weights')
    for r in read(REPO/'results/task-compatible-v11/training_lock.json')['jobs']:weights[r['checkpoint']]=r['sha256']
    for r in read(REPO/'results/task-compatible-v11/combined_weight_lock.json')['supplement_jobs']:weights[r['checkpoint']]=r['sha256']
    if len(weights)!=243:raise ValueError('Expected243priorselectedweights')
    for p,sha in weights.items():add(p,sha)
    for name,sha in (('parallel-joint-v8','abad8120080f555e99876489b3028903f4b35e487a929aeb7654f19ebadbb610'),
        ('background-followup-v9','0a9496c04d8964f7b4160a071cc9077fce2007e03c10a841f9040139cc3bf3f2'),
        ('corridor-visibility-v10','f464d1f8f9c88303a6859e673a97b6456ba67c78f3ef128bd70c03997d76b2a0'),
        ('task-compatible-v11','bc827b3a588e310e54c5112619d58551944216bd3d1a96edc5f72e0dbd613f6b')):add(REPO/'results'/name/'evidence.zip',sha)
    for name in ('task-compatible-v11','pretrained-photo-v11b'):
        for p,sha in read(REPO/'results'/name/'training_plan.json')['source_sha256'].items():add(REPO/p,sha)
    for r in read(REPO/'protocols/crosswalk-partial-background-v8/manifest.json')['records']:
        add(r['original_mask'],r['original_sha256']);add(r['draft_mask'],r['draft_sha256'])
    plan=read(ROOT/'training_plan.json')
    for p,sha in plan['source_SHA256'].items():add(REPO/p,sha)
    for p,sha in plan['library_SHA256'].items():add(REPO/'.venv/Lib/site-packages/transformers'/p,sha)
    old=read(REPO/'results/task-compatible-v11/training_plan.json')
    for name,sha in old['pretrained']['files'].items():add(REPO/'pretrained/mit-b0-imagenet'/name,sha)
    for path,sha in inventory.items():
        if digest(path)!=sha:raise ValueError('Historical source/weight/archive/data changed:'+path)
    return dict(prior_selected_files=243,old_core_files=len(core),verified_files=len(inventory),SHA256=inventory,checked_utc=now())

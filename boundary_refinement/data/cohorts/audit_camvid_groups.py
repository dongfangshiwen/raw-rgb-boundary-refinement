"""Source-only CamVid sequence grouping, without interpreting official test masks."""
from collections import defaultdict

import numpy as np
from PIL import Image

from boundary_refinement.data.camvid_data import DATA
from boundary_refinement.data.provenance import REPO, digest, now, read, write


OUT=REPO/'results/camvid-sequence-audit-v17'
CLASSES=('Sky','Building','Pole','Road','Pavement','Tree','SignSymbol','Fence','Car','Pedestrian','Bicyclist')


def main():
    rows=[r for r in read(DATA)['semantic_records'] if r['source']=='camvid']
    groups=defaultdict(lambda: defaultdict(list));masks=0
    for row in rows:
        if row['video']!=row['id'].rsplit('_',1)[0]:
            raise ValueError('Author sequence ID mismatch')
        groups[row['video']][row['partition']].append(row['id'])
    if {k:sum(len(v.get(k,[])) for v in groups.values()) for k in ('train','val','test')}!={'train':367,'val':101,'test':233}:
        raise ValueError('Author split cardinality changed')
    development=[r for r in rows if r['partition'] in ('train','val')]
    for row in development:
        if digest(row['mask'])!=row['mask_SHA256']:
            raise ValueError('Source mask changed')
        with Image.open(row['mask']) as im:
            mask=np.asarray(im)
            if mask.ndim!=2 or np.any((mask<0)|(mask>11)) or im.size!=tuple(row['size']):raise ValueError('Bad original annotation')
            row['label_counts']=np.bincount(mask.ravel(),minlength=12).tolist()
        masks+=1
    dev_groups=sorted({r['video'] for r in development})
    folds=[]
    for video in dev_groups:
        train=[r for r in development if r['video']!=video]
        val=[r for r in development if r['video']==video]
        def describe(part):
            pixels=np.sum([r['label_counts'] for r in part],axis=0).tolist()
            return dict(count=len(part),image_ids=[r['id'] for r in part],class_pixels_11=pixels[:11],void=pixels[11],
                        missing_classes=[n for n,c in zip(CLASSES,pixels) if c==0])
        folds.append(dict(held_video=video,train=describe(train),validation=describe(val)))
    group_sizes={k:{p:len(names) for p,names in parts.items()} for k,parts in groups.items()}
    never_fit={k:len(parts['test']) for k,parts in groups.items() if set(parts)=={'test'}}
    value=dict(all_passed=True,classes=CLASSES,manifest_SHA256=digest(DATA),standard_split_video_counts=group_sizes,
        training_or_validation_unseen_test_videos=never_fit,source_only_folds=folds,decoded_source_masks=masks,
        test_masks_opened=0,new_test_predictions=0,training_updates=0,
        scope='Video-disjoint CV feasibility only. V15–V17 use the original standard split. Same site could recur in separate video IDs. Folds reuse development data; independent location claims require a separate dataset.',finished_utc=now())
    write(OUT/'audit.json',value)
    print(dict(standard_split=group_sizes,test_only_videos=never_fit,decoded_source_masks=masks,
               feasible_folds=[(f['held_video'],f['train']['count'],f['validation']['count'],
                                f['train']['missing_classes'],f['validation']['missing_classes']) for f in folds],test_predictions=0))


if __name__=='__main__':main()

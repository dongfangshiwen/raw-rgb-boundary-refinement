"""Descriptive class-wise and leave-one-class-out effects after the view audit."""
from pathlib import Path
import argparse
import numpy as np
from boundary_refinement.evaluation.semantic import path_counts as io

def run(root):
    audit=io.read(root/"view-audit-decisions.json")
    city=["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]
    rows={}
    for tag,suite,names in (("bdd","mask2former",city),("wp","mask2former",["building","road","sky"]),("wp","ade5",["building","road","sky","tree","grass"])):
        arms=io.read(root/(tag+"-evaluation.json"))["suites"][suite];groups={}
        for p in sorted((root/"counts"/tag).glob("*.npz")):
            with np.load(p,allow_pickle=False) as f:
                g=f["group"].item()
                if tag=="wp":
                    if f["weather"].item()!="fog" or g in audit["exclude_new_groups"]:continue
                    g=audit["merge_into"].get(g,g)
                x=f[suite][1].astype(np.int64);groups[g]=groups.get(g,0)+x
        sub=np.stack(list(groups.values()));gt=sub.sum(-1);hits=sub.diagonal(axis1=-2,axis2=-1);union=gt+sub.sum(-2)-hits
        hit=hits.sum(0);un=union.sum(0);iou=np.divide(hit,un,out=np.zeros_like(hit,dtype=float),where=un>0)*100
        deltas=(iou[arms.index("path")]-iou[arms.index("paper")]).mean(0)[:len(names)]
        take=[]
        for i,name in enumerate(names):
            take.append(dict(label=name,pixels=int(gt[:,0,0,i].sum()),support_groups=int((gt[:,0,0,i]>0).sum()),class_iou_delta_pp=deltas[i],macro_omitting_class_pp=float(np.delete(deltas,i).mean())))
        rows[tag+"_"+suite]=dict(classes=take,macro_pp=float(deltas.mean()),positive_classes=int((deltas>0).sum()),negative_classes=int((deltas<0).sum()),
             class_deletion_range_pp=[min(r["macro_omitting_class_pp"] for r in take),max(r["macro_omitting_class_pp"] for r in take)])
    io.write(root/"class-influence.json",rows)
if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--root",type=Path,default=io.DEFAULT/"manuscript-extension");run(cli.parse_args().root)

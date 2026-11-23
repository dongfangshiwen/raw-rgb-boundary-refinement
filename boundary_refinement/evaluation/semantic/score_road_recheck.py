"""CPU-only recalculation on previously inspected driving-in-fog cohorts."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from boundary_refinement.evaluation.semantic import path_counts as io
from boundary_refinement.evaluation.semantic.anchor_counts import PAIRS

def run(root):
    card=io.read(root/"road-recheck-evaluation.json");res={}
    for tag,s in card["panels"].items():
        groups=defaultdict(lambda:0);seen=set()
        for p in sorted((root/"road-recheck"/tag).glob("*.npz")):
            with np.load(p,allow_pickle=False) as f:
                assert f["lock"].item()==s["lock"] and f["id"].item() not in seen;seen.add(f["id"].item())
                cm=f["counts"];g=f["group"].item()
                assert cm.shape==(2,2,len(card["arms"]),3,19,19) and (cm[:,1]<=cm[:,0]).all() and (cm>=0).all()
                gt=cm.sum(-1);hits=cm.diagonal(axis1=-2,axis2=-1);assert (gt==gt[:1,:,:1,:1]).all()
                groups[g]+=np.stack((hits,gt+cm.sum(-2)-hits,gt),-1)
        assert len(groups)==s["groups"] and len(seen)==s["frames"]
        names=sorted(groups);data=np.stack([groups[g] for g in names])
        classes=[c for c in range(19) if tag!="idd" or c not in (9,16)]
        sub=np.take(data,classes,axis=-2)
        res[tag]={reg:io.metrics(sub[:,:,j],names,card["arms"],card["targets"],PAIRS,10517) for j,reg in enumerate(("full","boundary"))}
        print("Recomputed secondary road fog",tag,len(seen),len(groups),flush=True)
    io.write(root/"road-recheck-results.json",res)

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--root",type=Path,default=io.DEFAULT/"manuscript-extension");run(cli.parse_args().root)

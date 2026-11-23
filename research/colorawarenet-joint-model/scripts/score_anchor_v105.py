"""Rebuild fixed-class IoUs and paired intervals without CUDA or original data."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
try:
    from scripts import score_path_v104 as utility
except ImportError:
    import score_path_v104 as utility

PAIRS={"paper":("released","fixed_grid","gaussian"),
       "gaussian_path":("gaussian",),
       "path":("paper","endpoint","shifted","permuted","gaussian","gaussian_path"),
       "paper_segfix":("paper","segfix","fixed_grid_segfix","gaussian_segfix"),
       "path_segfix":("paper_segfix","path","segfix","gaussian_segfix"),
       "gaussian_segfix":("gaussian","segfix")}

def run(root,audited=False):
    results={}
    audit=utility.read(root/"view-audit-decisions.json") if audited else None
    for panel in ("bdd","wp"):
        card=utility.read(root/f"{panel}-evaluation.json")
        seen=set();bank=defaultdict(dict);weathers={}
        for file in sorted((root/"counts"/panel).glob("*.npz")):
            with np.load(file,allow_pickle=False) as f:
                assert f["lock"].item()==card["lock"] and f["seeds"].tolist()==[101,211,307]
                key,group=f["id"].item(),f["group"].item()
                assert key not in seen and group not in weathers
                weathers[group]=f["weather"].item();seen.add(key)
                for suite,arms in card["suites"].items():
                    cm=f[suite];assert cm.dtype==np.int32 and cm.shape==(2,len(arms),3,19,19)
                    gt=cm.sum(-1);assert (gt==gt[:,:1,:1]).all() and (cm[1]<=cm[0]).all() and (cm>=0).all()
                    hit=cm.diagonal(axis1=-2,axis2=-1)
                    bank[suite][group]=np.stack((hit,gt+cm.sum(-2)-hit,gt),-1)
        assert len(seen)==len(weathers)==card["frames"]==card["groups"]
        removed=set(audit["exclude_new_groups"]) if audited and panel=="wp" else set()
        renames=audit["merge_into"] if audited and panel=="wp" else {}
        eligible=sorted(set(weathers)-removed)
        subsets={"unrestricted":eligible} if panel=="bdd" else {
            "fog":[g for g in eligible if weathers[g]=="fog"],"fog_mixed":eligible}
        for subset,names in subsets.items():
            out={}
            clustered=sorted({renames.get(g,g) for g in names})
            for suite,arms in card["suites"].items():
                classes=list(range(19 if panel=="bdd" else int(suite[3:]) if suite.startswith("ade") else 3))
                pooled=defaultdict(lambda:0)
                for g in names:pooled[renames.get(g,g)]+=bank[suite][g]
                data=np.take(np.stack([pooled[g] for g in clustered]),classes,axis=-2)
                used={k:tuple(x for x in cs if x in arms) for k,cs in PAIRS.items() if k in arms}
                out[suite]={reg:utility.metrics(data[:,j][:,None],clustered,arms,[suite],used,10517) for j,reg in enumerate(("full","boundary"))}
                out[suite]["classes"]=classes
            results[panel+"_"+subset]=dict(cameras=len(clustered),images=len(names),suites=out)
            print("Recomputed",panel,subset,len(names),len(clustered),flush=True)
    utility.write(root/("results-view-audited.json" if audited else "results.json"),dict(panels=results,bootstrap_draws=5000,seed=10517))

if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("--root",type=Path,default=utility.DEFAULT/"manuscript-extension")
    cli.add_argument("--audit-views",action="store_true");args=cli.parse_args();run(args.root,args.audit_views)

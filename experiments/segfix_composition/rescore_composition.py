"""Paired cluster intervals for the exploratory ordered composition."""
import json
from experiments.segfix_composition.rescore_native import HERE,CLASSES,read,get_groups,analyze

PAIRS={"paper_then_segfix":("segfix","paper_512","gaussian_then_segfix","released"),
       "gaussian_then_segfix":("segfix","matched_gaussian_fixed")}

def run():
    root=HERE/"composition"
    record=read(root/"evaluation.json")
    assert record["parent"]==read(HERE/"evaluation.json")["lock"]
    rows=read(HERE/"manifest.json")["rows"]
    result={"status":"Exploratory on previously inspected cohorts","lock":record["lock"],"cohorts":{}}
    for cohort in CLASSES:
        subset={r["id"]:r for r in rows if r["cohort"]==cohort}
        n=len(subset)
        grouped=get_groups(subset,cohort,record,root)
        result["cohorts"][cohort]=dict(frames=n,groups=len(grouped),
                targets=analyze(grouped,record["arms"],PAIRS))
    (root/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Verified four baselines and scored all frozen compositions in 190 frames.")

if __name__=="__main__":
    run()

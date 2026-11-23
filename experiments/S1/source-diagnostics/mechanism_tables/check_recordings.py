"""Delete one recording at a time and recalculate the paired boundary effect in the existing stress test."""
import argparse
from pathlib import Path
import gzip
import json
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE.parent.parent/"calibrated-tone-v99"
def score(mats):
    diag=mats.diagonal(axis1=-2,axis2=-1)
    union=mats.sum(-1)+mats.sum(-2)-diag
    per=np.divide(100.*diag,union,out=np.zeros_like(diag,dtype=float),where=union>0)
    return per.mean(axis=(-1,-2))

def compute(rows,official):
    result={}
    arms=["mean","fixed"];seeds=[101,211,307]
    for cohort in ("MUSES_t70","MUSES_t40"):
        selected=[r for r in rows if r["ds"]==cohort and r["rgb"]=="curve" and r["sem"] in arms]
        groups=sorted({r["group"] for r in selected})
        assert len(groups)==13 and len(selected)==33*3*2
        h=np.zeros((len(groups),2,3,19,19),dtype=np.int64)
        for r in selected:
            h[groups.index(r["group"]),arms.index(r["sem"]),seeds.index(r["seed"])]+=np.asarray(r["edge"])
        def delta(matrices):
            uniform,weighted=score(matrices)
            return float(weighted-uniform)
        total=h.sum(0)
        all_effect=delta(total)
        np.testing.assert_allclose(all_effect,official["edge"][cohort]["effects"]["curve-fixed_minus_mean"]["diff"],atol=1e-8)
        omit={name:delta(total-h[i]) for i,name in enumerate(groups)}
        result[cohort]={"pooled":all_effect,"omitted_recording_effects":omit,
                        "range":[min(omit.values()),max(omit.values())],
                        "positive_omissions":sum(e>0 for e in omit.values())}
    return result

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="S1 source-diagnostics/original/data/joint directory")
    parser.add_argument("--output", type=Path, default=HERE/"recording-sensitivity.json")
    args = parser.parse_args()
    with gzip.open(args.data/"joint-seg.json.gz", "rt", encoding="utf8") as stream:
        rows = json.load(stream)
    res = compute(rows, json.loads((args.data/"joint-test.json").read_text(encoding="utf8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res,indent=2), encoding="utf8")
    print([(c,a["positive_omissions"],a["range"]) for c,a in res.items()])

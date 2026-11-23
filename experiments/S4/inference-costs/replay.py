"""Verify the descriptive timing table from individual batch-one observations."""
from hashlib import sha256
from pathlib import Path
from statistics import mean
import csv
import json
import math

ROOT=Path(__file__).resolve().parent


def verify():
    plan=ROOT/"timing-protocol.json"
    protocol=json.loads(plan.read_text("utf8"))
    data=json.loads((ROOT/"timing-results.json").read_text("utf8"))
    assert data["protocol_sha256"]==sha256(plan.read_bytes()).hexdigest()
    assert protocol["script_sha256"]==sha256((ROOT/"profile_mask.py").read_bytes()).hexdigest()
    assert len({r["group"] for r in protocol["frames"]})==24 and protocol["repeats_per_frame"]==6
    with (ROOT/"summary.csv").open(encoding="utf8",newline="") as f:
        csvrows={(v["backbone"],v["decoder"]):v for v in csv.DictReader(f)}
    assert len(csvrows)==6
    for target,d in data["targets"].items():
        assert d["validated_native_masks"]==72
        base=d["measurements"]["bilinear"]
        for method,measure in d["measurements"].items():
            sample=measure["elapsed_ms"]
            peak=measure["peak_bytes"]
            assert len(sample)==len(peak)==144 and all(x>0 for x in sample)
            ms=mean(sample)
            values={"mean_ms":ms,"serial_fps":1000/ms,"peak_mib":max(peak)/2**20,
                    "additional_ms_vs_bilinear":ms-mean(base["elapsed_ms"]),
                    "additional_peak_mib_vs_bilinear":(max(peak)-max(base["peak_bytes"]))/2**20}
            for key,value in values.items():
                assert math.isclose(d["summary"][method][key],value,rel_tol=1e-11,abs_tol=1e-10)
                assert abs(float(csvrows[target,method][key])-value)<.00051
    return len(csvrows)


if __name__=="__main__":
    print("Verified",verify(),"timing rows from individual frames.")

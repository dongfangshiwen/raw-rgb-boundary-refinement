"""Audit dataset weather labels by exact published image ID, after locking RGB selection."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from experiments.recording_gate.fresh_clips import HERE, ROOT, history, put, read


def run():
    regs=read(ROOT/"benchmark_data/bdd-fog-v73/registration.json")
    def read_attributes(meta):
        raw=history.road.fetch(meta["url"])
        assert sha256(raw).hexdigest()==meta["SHA256"]
        rows=[json.loads(line) for line in raw.splitlines()]
        assert len(rows)==meta["rows"]
        return [(r["file_name"].removesuffix(".jpg"),r["label"]) for r in rows]
    with ThreadPoolExecutor(8) as pool:
        rows=[pair for batch in pool.map(read_attributes,regs["metadata"]) for pair in batch]
    attrs=dict(rows)
    selected=read(HERE/"bdd-evaluation.json")["rows"]
    checked=[dict(id=r["id"],code=r.get("code"),panel=r["panel"],published=attrs.get(r["id"])) for r in selected]
    put(HERE/"bdd-weather-audit.json",dict(shards=len(regs["metadata"]),rows=len(rows),
        source_revision=regs["revisions"],match="Exact image ID only. Missing entries are unknown. Frozen inclusion and outputs never changed.",
        counts={k:dict(Counter(x["published"] or "unmatched" for x in checked if x["panel"]==k)) for k in ("visible_fog","random")},
        cases=checked),True)
    print([(r["code"],r["published"]) for r in checked if r["code"]],flush=True)


if __name__=="__main__":
    run()

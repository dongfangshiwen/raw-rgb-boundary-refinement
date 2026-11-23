"""Reserve previously unscored capture codes in the paired RW2AH release."""
from collections import defaultdict
import hashlib

from PIL import Image
import torch

from boundary_refinement.evaluation.photographic.evaluate_conditional_fusion import evaluation_pairs
from boundary_refinement.data.staging.prepare_rw2ah import ROOT as V25, STORE as RW_STORE, archive_split, obtain
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.data.real_haze_data import overlap_check
from boundary_refinement.data.staging.stage_fusion_holdouts import prior_refs, witness
from boundary_refinement.training.train_regional_guidance import atomic_write, torch_save


ROOT = REPO/"results/post-fusion-v30"
STORE = REPO.parent/"benchmark_data/post-fusion-holdout-v30"
RW_CARD = ROOT/"rw_reserve.json"
RW_SIZE = 48


def observed_refs():
    yield from prior_refs()
    sources = (("rw2ah",read(V25/"data_manifest.json")["test"]),)
    for name,record in sources:
        if digest(record["file"])!=record["SHA256"]:
            raise ValueError("An earlier paired dataset changed")
        for row in torch.load(record["file"],map_location="cpu",weights_only=True):
            image = Image.fromarray(row["clean"].permute(1,2,0).numpy())
            yield witness(name+"/"+row["id"],image,None)
    for cohort in ("rwhaze","ntire24_complement"):
        seen = set()
        for dataset,scene,group,_,clear,_ in evaluation_pairs(cohort):
            if group not in seen:
                image = Image.fromarray(clear.permute(1,2,0).numpy())
                yield witness(dataset+"/"+scene,image,None)
                seen.add(group)


def key(value):
    return hashlib.sha256(("post-fusion-v30/"+value).encode()).digest()


def camera_code(scene):
    return scene.split("-",1)[0]


def rw_reserve():
    ROOT.mkdir(parents=True,exist_ok=True)
    STORE.mkdir(parents=True,exist_ok=True)
    if RW_CARD.exists():
        record = read(RW_CARD)
        if digest(record["file"])!=record["SHA256"]:
            raise ValueError("The registered RW2AH extension changed")
        return record
    indexed = RW_STORE/"index.json"
    source = read(indexed)
    known = read(V25/"data_manifest.json")
    if digest(indexed)!=known["index_SHA256"]:
        raise ValueError("The registered RW2AH archive directory changed")
    seen = {r["id"].rsplit("-",1)[0] for role in ("train","test") for r in known[role]["files"]}
    members,_ = archive_split(source,"test")
    by_view = defaultdict(lambda: defaultdict(list))
    for scene in sorted(members):
        capture = scene.rsplit("-",1)[0]
        if capture not in seen:
            by_view[camera_code(scene)][capture].append(scene)
    candidates = []
    queues = {c:sorted(groups,key=key) for c,groups in by_view.items()}
    for round_id in range(max(map(len,queues.values()))):
        for camera in sorted(queues,key=key):
            if round_id < len(queues[camera]):
                family = queues[camera][round_id]
                candidates.append(min(by_view[camera][family],key=key))
    picked = candidates[:RW_SIZE]
    if len(picked)!=RW_SIZE:
        raise ValueError("Not enough new paired capture codes")
    staged = obtain(source,{"test":[(i,members[i]) for i in picked]})["test"]
    flags,nearest = overlap_check(list(observed_refs()),[card for card,row in staged if row is not None])
    excluded = {r["id"] for r in flags}
    rows = [row for card,row in staged if card["id"] not in excluded and row is not None]
    if not rows:
        raise ValueError("No new RW2AH pairs passed validation")
    target = STORE/"rw_reserve.pth"
    torch_save(target,rows)
    record = dict(count=len(rows),ids=[r["id"] for r in rows],intended=picked,file=str(target),
                  SHA256=digest(target),source_url=source["url"],revision=source["revision"],
                  paper="https://ojs.aaai.org/index.php/AAAI/article/view/32297",
                  groups=sorted({camera_code(r["id"]) for r in rows}),excluded=flags,nearest=nearest,
                  source_directory_SHA256=digest(indexed),
                  scope="Unused capture codes from the public redistribution's RW2AH test split. Camera prefixes can overlap older cohorts. These are new local evaluation pairs, not a different acquisition dataset.",
                  selection="Exclude all first-two-part codes previously used in V25, sample round-robin across the first part, and order deterministically within each group without scores.",
                  finished_utc=now())
    atomic_write(RW_CARD,record)
    print(dict(cohort="RW2AH capture reserve",prepared=len(rows),requested=len(picked),
               excluded=len(excluded),prefixes=record["groups"]),flush=True)
    return record


if __name__=="__main__":
    rw_reserve()

"""Reserve untouched RW²AH capture codes for the source-retention study."""
from collections import defaultdict
from hashlib import sha256

from PIL import Image
import torch

from scripts.prepare_rw2ah_v25 import ROOT as V25, STORE, archive_split, obtain
from scripts.projective_ledger_v12 import digest, now, read
from scripts.real_haze_data_v22 import overlap_check
from scripts.stage_fusion_holdouts_v29 import witness
from scripts.stage_postfusion_holdout_v30 import RW_CARD, camera_code, observed_refs
from scripts.structure_transfer_v32 import ROOT
from scripts.train_regional_guidance_v18 import atomic_write, torch_save


CARD = ROOT / "rw_confirmation.json"
SAMPLE_SIZE = 24


def capture(id):
    return id.rsplit("-",1)[0]


def order(id):
    return sha256(("source-replay-v32/"+id).encode()).digest()


def stage():
    ROOT.mkdir(parents=True,exist_ok=True)
    if CARD.exists():
        card = read(CARD)
        if digest(card["file"])!=card["SHA256"]:
            raise ValueError("RW²AH confirmation pixels changed")
        return card
    old = read(V25/"data_manifest.json")
    previous = read(RW_CARD)
    observed = {capture(r["id"]) for role in ("train","test") for r in old[role]["files"]}
    observed.update(capture(id) for id in previous["intended"])
    archive_file = STORE/"index.json"
    if digest(archive_file)!=old["index_SHA256"]:
        raise ValueError("The author-linked RW²AH archive index changed")
    archive = read(archive_file)
    candidates,_ = archive_split(archive,"test")
    groups = defaultdict(lambda:defaultdict(list))
    for scene in candidates:
        if capture(scene) not in observed:
            groups[camera_code(scene)][capture(scene)].append(scene)
    selected = []
    sorted_groups = {k:sorted(population,key=order) for k,population in groups.items()}
    for j in range(max(map(len,sorted_groups.values()))):
        for group in sorted(sorted_groups,key=order):
            population = sorted_groups[group]
            if j<len(population):
                selected.append(min(groups[group][population[j]],key=order))
    selected = selected[:SAMPLE_SIZE]
    if len(selected)!=SAMPLE_SIZE:
        raise ValueError("Insufficient unscored capture codes")
    scored = obtain(archive,{"test":[(id,candidates[id]) for id in selected]})["test"]
    past = list(observed_refs())
    if digest(previous["file"])!=previous["SHA256"]:
        raise ValueError("Earlier RW²AH reserve changed")
    for row in torch.load(previous["file"],map_location="cpu",weights_only=True):
        image = Image.fromarray(row["clean"].permute(1,2,0).numpy())
        past.append(witness("RW2AH prior/"+row["id"],image,"prior:"+row["id"]))
    flags,nearest = overlap_check(past,[card for card,frame in scored if frame is not None])
    omitted = {r["id"] for r in flags}
    valid = [r for card,r in scored if r is not None and card["id"] not in omitted]
    if not valid:
        raise ValueError("No eligible new captures")
    target = ROOT/"rw_confirmation_1024.pth"
    torch_save(target,valid)
    ledger = dict(index_SHA256=digest(archive_file),source_revision=archive["revision"],selected=selected,
                  count=len(valid),ids=[r["id"] for r in valid],file=str(target),SHA256=digest(target),
                  exclusions=flags,nearest=nearest,scope="Unused first-two-part capture codes, same author-linked RW²AH test archive and camera prefixes. Author checkpoint coverage unknown. Cluster estimates by camera prefix.",finished_utc=now())
    atomic_write(CARD,ledger)
    print(dict(paired=len(valid),excluded=list(omitted)),flush=True)
    return ledger


if __name__ == "__main__":
    stage()

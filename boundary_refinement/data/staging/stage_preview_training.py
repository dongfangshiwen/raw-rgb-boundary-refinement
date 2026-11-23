"""Recover the original RGB pixels for the already registered V25 training scenes."""
from boundary_refinement.data.staging.prepare_rw2ah import ROOT as V25, STORE, archive_split, obtain
from boundary_refinement.data.provenance import REPO, digest, now, read
from boundary_refinement.training.train_regional_guidance import atomic_write, torch_save


ROOT = REPO / "results/preview-routing-v28"
STAGED = ROOT / "native_train.json"


def stage():
    ROOT.mkdir(parents=True, exist_ok=True)
    if STAGED.exists():
        card = read(STAGED)
        if digest(card["file"]) != card["SHA256"]:
            raise ValueError("Registered preview training pixels changed")
        return card
    upstream = read(V25 / "data_manifest.json")
    original_index = STORE / "index.json"
    if digest(original_index) != upstream["index_SHA256"]:
        raise ValueError("RW2AH central directory changed")
    index = read(original_index)
    train = upstream["train"]
    old = {r["id"]: r for r in train["files"] if "clean" in r}
    if len(old) != 96 or list(old) != sorted(old):
        raise ValueError("Expected the 96 already registered training pairs")
    members, _ = archive_split(index, "train")
    # Obtain the evaluator-size view of the training archive members.
    recovered = obtain(index, {"test": [(k, members[k]) for k in old]})["test"]
    payload = []
    for check, pixels in recovered:
        row = old[check["id"]]
        if pixels is None or any(check[k]["SHA256"] != row[k]["SHA256"] for k in ("hazy", "clean")):
            raise ValueError("Recovered RGB pair does not match V25 source " + row["id"])
        payload.append(pixels)
    target = STORE / "preview_native_train_v28.pth"
    torch_save(target, payload)
    card = dict(count=len(payload), ids=list(old), file=str(target), SHA256=digest(target),
                upstream_SHA256=digest(V25 / "data_manifest.json"), archive_SHA256=digest(original_index),
                view="Native through long edge 1024, RGB only. The paired archive members serve training only.",
                finished_utc=now())
    atomic_write(STAGED, card)
    print(dict(staged=card["count"], bytes=target.stat().st_size), flush=True)
    return card


if __name__ == "__main__":
    stage()

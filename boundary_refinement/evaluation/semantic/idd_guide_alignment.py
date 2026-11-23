"""Check the deployed raw guide and its spatial controls on two previously scored IDD-AW cohorts."""
import numpy as np
from PIL import Image
import torch

from boundary_refinement.evaluation.semantic import idd_fog_evaluate as evaluation
from boundary_refinement.inference import idd_fog_models as fit
from boundary_refinement.evaluation.paired import muses_transmission as joint
from boundary_refinement.inference import parallel_fog as deployed
from boundary_refinement.data.cohorts.idd_fog_data import DATA, MAP, OUT, fixed
from boundary_refinement.evaluation.semantic.idd_fog_reference import iou_counts
from boundary_refinement.evaluation.semantic.idd_peer_rescore import compute_summary
from boundary_refinement.data.provenance import digest, read, write


DEST = OUT / "raw-guide-spatial-check"
MIXED = OUT / "new-drive-guide-controls"
ARMS = ("raw", "flat", "shift", "dual", "author")


def register():
    fog = [r for r in read(DATA / "evaluation.json")["rows"] if "fog" in r["cohorts"]]
    mixed = read(MIXED / "source.json")["rows"]
    if {r["group"] for r in fog} & {r["group"] for r in mixed}:
        raise ValueError("Overlapping drive IDs")
    fixed(DEST / "protocol.json", {
        "status": "Post hoc. Both IDD cohorts were previously scored; all existing weights remain frozen.",
        "fog_source": digest(DATA / "evaluation.json"), "mixed_source": digest(MIXED / "source.json"),
        "deployed_code": digest(deployed.__file__), "semantic_weights": digest(joint.weighted.ROOT / "completed.json"),
        "original_fog_transfer": digest(OUT / "decoder-transfer-protocol.json"),
        "original_mixed_transfer": digest(MIXED / "protocol.json"), "seeds": list(fit.SEEDS),
        "arms": list(ARMS), "input": "512 long side, native GT, 17 shared classes, radius 6 boundary band.",
        "primary": "Exact deployed raw guide minus the published Mask2Former inference on the 24 screened fog drives.",
        "mechanism": "Replace both raw guides with a spatial constant (zero RGB cost, learned temperature retained), "
                     "or use the same raw guide rolled by half the coarse grid rounded down in both axes. "
                     "Hold the semantic logits, active-cell selection, geometry, weights and 16-child uniform pooling fixed.",
        "statistics": "Same source-seed average and 5000 paired drive bootstrap draws as the original evaluation. "
                      "All intervals pointwise. Report both endpoints on both cohorts, including unfavorable results."
    })
    return {"fog": fog, "mixed": mixed}


@torch.inference_mode()
def evaluate_frame(model, row, cohort, check_dual):
    path = DATA / row["rgb"]
    if digest(path) != row["rgb_sha"]:
        raise ValueError(f"RGB checksum changed for {row['id']}")
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    truth = evaluation.native_truth(row)
    counts = evaluation.Counts(truth)
    logits, guides, _, work = model.features(rgb)
    h, w = logits.shape[-2:]
    weights = logits.new_full((h*w, 16), 1./16)
    template = joint.base.guide_stencil(guides[:0], logits.argmax(1)[0])

    def score(guide_pair):
        stencil = joint.weighted.prepare(guide_pair, logits, "dual_core", template, weights)
        per_seed = []
        for seed in fit.SEEDS:
            delta = joint.base.prediction("dual_free", model.decoders["dual_core", seed], logits, stencil)
            mask = model.native_mask(logits, delta, work, truth.shape)
            per_seed.append(np.stack(counts.score(mask)))
        return np.stack(per_seed, axis=1)

    raw_guide = guides[[0, 0]]
    raw = score(raw_guide)
    flat = score(torch.zeros_like(raw_guide))
    shift = score(raw_guide.roll((4*(h//2), 4*(w//2)), dims=(-2, -1)))
    filename = row["id"].replace("/", "_") + ".npz"
    if cohort == "fog":
        with np.load(OUT / "mask2former-counts" / filename) as f:
            author = np.repeat(f["counts"][:, None], len(fit.SEEDS), axis=1)
        with np.load(OUT / "backbone-transfer" / filename) as f:
            dual = f["uniform"]
    else:
        with np.load(MIXED / "counts" / filename) as f:
            author, dual = f["author"], f["uniform_dual"]
            if not np.array_equal(f["uniform_raw"], raw):
                raise ValueError(f"Raw-guide predictions differ from original counts on {row['id']}")
    if check_dual and not np.array_equal(score(guides), dual):
        raise ValueError(f"Dual-guide reproduction failed on {row['id']}")
    all_counts = np.stack((raw, flat, shift, dual, author), axis=1)
    support = all_counts.sum(-1)
    if not np.all(support == support[:, :1, :1]):
        raise ValueError(f"Ground truth support differs on {row['id']}")
    return all_counts


def summarize(rows, matrices):
    compact = iou_counts(matrices, sorted(MAP.values()))
    contrasts = {}
    for arm in ARMS[1:]:
        pair = compact[:, :, [0, 0, ARMS.index(arm)]]
        packets = [{"group": r["group"], "v": v} for r, v in zip(rows, pair, strict=True)]
        scores = compute_summary(packets)["scores"]
        contrasts["raw_minus_" + arm] = {
            endpoint: {"raw": s["weighted"], "control": s["corrected_peer"],
                       "delta_pp": s["ours_minus_corrected"], "paired_ci95_pp": s["paired_ci95"]}
            for endpoint, s in scores.items()
        }
    return {"frames": len(rows), "drives": len({r["group"] for r in rows}), "contrasts": contrasts}


def run():
    cohorts = register()
    model = deployed.ParallelPredictor()
    protocol_sha = digest(DEST / "protocol.json")
    result = {"protocol_sha256": protocol_sha, "script_sha256": digest(__file__),
              "pytorch": str(torch.__version__), "gpu": torch.cuda.get_device_name(), "cohorts": {}}
    for name, rows in cohorts.items():
        cache = DEST / f"{name}-counts.npz"
        ids = [r["id"] for r in rows]
        if not cache.exists():
            matrices = []
            for i, row in enumerate(rows):
                matrices.append(evaluate_frame(model, row, name, i in (0, len(rows)-1)))
                if (i+1) % 40 == 0:
                    print(name, i+1, "of", len(rows), flush=True)
            temp = cache.with_suffix(".part")
            with temp.open("wb") as f:
                np.savez_compressed(f, ids=ids, groups=[r["group"] for r in rows], arms=ARMS,
                                    seeds=fit.SEEDS, protocol=protocol_sha, counts=np.stack(matrices))
            temp.replace(cache)
        with np.load(cache) as f:
            if f["ids"].tolist() != ids or f["arms"].tolist() != list(ARMS) or f["protocol"].item() != protocol_sha:
                raise ValueError(f"Stale cached counts for {name}")
            result["cohorts"][name] = summarize(rows, f["counts"])
        result["cohorts"][name]["counts_sha256"] = digest(cache)
        print(name, result["cohorts"][name]["contrasts"]["raw_minus_author"], flush=True)
    write(DEST / "summary.json", result)


if __name__ == "__main__":
    run()

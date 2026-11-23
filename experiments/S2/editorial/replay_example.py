"""Recreate Fig. 11c from the frozen original, using the first reserved validation frame."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Path to colorawarenet-joint-model")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    from scripts import eval_range_v101 as ev

    ev.setup.configure_cuda()
    row = ev.rows("bddval")[0]
    assert row["id"] == "8a56deb7-6cf433c2"
    image, truth = ev.scene(row, "bddval")
    scoring = ev.ev.Counts(truth)
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    results = {}
    with torch.inference_mode():
        original = ev.original.load()["dual_core", 101]
        inp, (h, w) = ev.setup.peer_input(image)
        rgb = inp[..., :h, :w].contiguous()
        guide = rgb.clamp(0, 1).mul(255).round().byte()
        for backbone in ("mask2former", "segformer"):
            net = ev.Target(backbone)
            logits, released = net(rgb, tuple(truth.shape))
            local = ev.legacy.guide_stencil(guide[[0, 0]], logits.argmax(1)[0])
            delta = ev.legacy.prediction("dual_free", original, logits, local)
            guided = ev.resize(logits, tuple(truth.shape)) + ev.resize(delta, tuple(truth.shape))
            packet = ev.OUT / f"counts-{backbone}-bddval" / (row["id"] + ".npz")
            with np.load(packet, allow_pickle=False) as archived:
                assert archived["id"].item() == row["id"] and archived["seeds"][0] == 101
                for arm, score in (("released", released), ("paper_fit", guided)):
                    mask = score.argmax(1)[0].byte()
                    col = archived["arms"].tolist().index(arm)
                    np.testing.assert_array_equal(np.stack(scoring.score(mask)), archived["counts"][:, col, 0])
                    path = output / f"{backbone}-{arm}.png"
                    Image.fromarray(mask.cpu().numpy()).save(path)
                    with Image.open(path) as im:
                        np.testing.assert_array_equal(np.asarray(im), mask.cpu().numpy())
                    results[path.name] = ev.digest(path)
            del net, logits, released, guided, delta, local
    record = {"frame": row["id"], "backbones": ["mask2former", "segformer"], "seed": 101,
              "arm": "paper_fit", "data": row, "origin": "First frame in the locked BDD100K validation manifest",
              "native_shapes_hw": list(truth.shape), "mask_files_sha256": results,
              "checks": "Both full-image and boundary confusion matrices equal the archived entries for both arms on both backbones."}
    (output / "frame.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(row["id"], "four native masks verified against both historical confusion matrices")


if __name__ == "__main__":
    main()

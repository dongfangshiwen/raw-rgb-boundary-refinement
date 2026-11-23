"""Evaluate the shared-field factors on drive-disjoint and real-fog images."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from dehaze_seg.metrics import confusion_matrix, segmentation_metrics
from scripts.author_pgl_v25 import pgl_rgb
from scripts.coarse_semantics_v39 import boundary_ids, blend, native_labels, paint, rgb_checkpoint
from scripts.evaluate_curve_transfer_v25 import as_float
from scripts.evaluate_real_semantics_v22 import DRIVING, ZURICH, load_pairs, load_segmenter, resize_image
from scripts.evaluate_scene_utility_v31 import bootstrap
from scripts.evaluate_selective_adaptation_v33 import ROOT as V33
from scripts.evaluate_structure_transfer_v32 import data_split
from scripts.evaluate_selective_adaptation_v33 import samples as old_samples
from scripts.kitti_fog_v40 import MANIFEST, ROOT, load, render
from scripts.parallel_utility_v32 import capture_group
from scripts.projective_ledger_v12 import digest, now, read
from scripts.shared_haze_field_v40 import ARMS, MAIN, SEEDS, encode, load_models, prediction
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda


RGB_COMPARISONS = (("inverse_only", "direct"), (MAIN, "visibility_only"), (MAIN, "anchor"))
SEG_COMPARISONS = (("visibility_only", "direct"), (MAIN, "inverse_only"), (MAIN, "shifted"))


def to_rgb(image):
    return (image[0].clamp(0, 1) * 255.).round().byte().permute(1, 2, 0).cpu().numpy()


def predictions(raw, models, anchors):
    packet = encode(raw, *anchors)
    outputs = {("anchor", 0): (packet[1], packet[-1], None)}
    for key, model in models.items():
        restored, scores, field, _ = prediction(model, packet, key[0])
        outputs[key] = (restored, scores, field)
    return outputs


def quality(image, clean):
    err = image - clean
    psnr = -10. * err.square().mean().clamp_min(1e-12).log10()
    edge = 127.5 * (err.diff(dim=-1).abs().mean() + err.diff(dim=-2).abs().mean())
    return dict(psnr=psnr.item(), edge_mae=edge.item())


def add_one(out, ds, view, name, group, model, seed, rgb, logits, field, clean=None, labels=None, edges=None, t=None):
    row = dict(dataset=ds, view=view, scene=name, group=group, arm=model, seed=seed)
    if clean is not None:
        row.update(quality(rgb, clean))
    if labels is not None:
        mask = native_labels(logits, labels.shape)
        row["confusion"] = confusion_matrix(mask, labels, 19).tolist()
        row["boundary"] = confusion_matrix(mask, edges, 19).tolist()
    if field is not None and t is not None:
        row["transmission_MAE"] = (field - F.adaptive_avg_pool2d(t, field.shape[-2:])).abs().mean().item()
    out.append(row)


def sample_panel(folder, title, images, caption, chosen_mask):
    folder.mkdir(parents=True, exist_ok=True)
    cw, ch = 365, 241
    figure = Image.new("RGB", (3 * cw, 2 * ch + 35), "#f3f6f9")
    draw = ImageDraw.Draw(figure)
    titlefont = ImageFont.load_default(size=18)
    font = ImageFont.load_default(size=16)
    draw.text((10, 8), title, font=titlefont, fill="#233444")
    for i, (array, text) in enumerate(zip(images, caption, strict=True)):
        x, y = (i % 3) * cw, 32 + (i // 3) * ch
        draw.text((x + 10, y + 5), text, font=font, fill="#233444")
        display = Image.fromarray(array)
        display.thumbnail((cw - 20, ch - 36), Image.Resampling.LANCZOS)
        figure.paste(display, (x + (cw - display.width) // 2, y + 28 + (ch - 36 - display.height) // 2))
    figure.save(folder / "comparison.png")
    raw, base, rgb, _, overlay, annotation = images
    for key, img in (("input", raw), ("restored", rgb), ("mask", chosen_mask), ("color_mask", paint(chosen_mask)),
                     ("overlay", overlay), ("anchor_rgb", base), ("label", annotation)):
        Image.fromarray(img).save(folder / (key + ".png"))
    return dict(size=list(chosen_mask.shape[::-1]), hashes={f.name: digest(f) for f in folder.glob("*.png")})


def save_sample(name, raw, outputs, label, original=None, anchor_model=None):
    base, anchor_logits, _ = outputs["anchor", 0]
    rgb, logits, _ = outputs[MAIN, SEEDS[0]]
    if original is not None:
        size = original.shape[:2]
        x = torch.from_numpy(original.copy()).permute(2, 0, 1)[None].cuda().float() / 255.
        anchor_native = pgl_rgb(anchor_model, x).clamp(0, 1)
        rgb = anchor_native + F.interpolate(rgb - base, size, mode="bilinear", align_corners=False)
        base, raw = anchor_native, x
    shape = label.shape
    frozen = native_labels(anchor_logits, shape).byte().cpu().numpy()
    mask = native_labels(logits, shape).byte().cpu().numpy()
    native = to_rgb(raw)
    images = (native, to_rgb(base), to_rgb(rgb), blend(native, frozen), blend(native, mask), paint(label))
    return sample_panel(ROOT / "visuals" / name, name, images,
                        ("Hazy input", "Anchor RGB", "Shared-field RGB", "Frozen SegFormer",
                         "Shared-field segmentation", "Fine annotation"), mask)


def overall(rows):
    classified = defaultdict(list)
    regression = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["view"], row["arm"], row["seed"])
        if "confusion" in row:
            classified[key].append(row)
        if "psnr" in row:
            regression[key].append(row)
    details = defaultdict(dict)
    for key, values in classified.items():
        for column, name in (("confusion", "miou"), ("boundary", "edge_miou")):
            matrix = np.sum([r[column] for r in values], axis=0, dtype=np.int64)
            details[key][name] = 100. * segmentation_metrics(torch.from_numpy(matrix))["miou"]
    for key, values in regression.items():
        for name in ("psnr", "edge_mae", "transmission_MAE"):
            if name in values[0]:
                details[key][name] = float(np.mean([r[name] for r in values]))
    averaged = defaultdict(list)
    for (d, v, arm, _), values in details.items():
        averaged[d, v, arm].append(values)
    summary = {"/".join(k): {m: float(np.mean([v[m] for v in values])) for m in values[0]}
               for k, values in averaged.items()}
    return summary, {"/".join(map(str, k)): v for k, v in details.items()}


def intervals(rows):
    photos, semantics = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for item in rows:
        ds, view, group, scene, arm = (item[k] for k in ("dataset", "view", "group", "scene", "arm"))
        if "psnr" in item:
            photos[ds, view, group, scene][arm].append(item["psnr"])
        if "confusion" in item:
            semantics[ds, view, group][arm].append(item["confusion"])
    rgb, seg = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for (ds, view, group, _), values in photos.items():
        for left, right in RGB_COMPARISONS:
            rgb[ds, view, left, right][group].append(np.mean(values[left]) - np.mean(values[right]))
    for (ds, view, group), values in semantics.items():
        miou = {m: 100. * segmentation_metrics(torch.from_numpy(np.sum(v, axis=0, dtype=np.int64)))["miou"]
                for m, v in values.items()}
        for left, right in SEG_COMPARISONS:
            seg[ds, view, left, right][group] = miou[left] - miou[right]
    return dict(
        rgb={"/".join(k): bootstrap({g: np.mean(v) for g, v in value.items()}) for k, value in rgb.items()},
        semantic={"/".join(k): bootstrap(value) for k, value in seg.items()},
        scope="RGB bootstrap pairs photographs within scene, and semantic bootstrap pairs capture-group mIoU after pooling group confusion matrices. Equal group summary can differ from aggregate corpus mIoU.",
    )


@torch.inference_mode()
def evaluate():
    configure_cuda()
    models = load_models()
    restorer, rgb_hash = rgb_checkpoint()
    seg, mean, std, seg_card = load_segmenter()
    anchors = (restorer, seg, mean, std)
    protocol = ROOT / "evaluation_protocol.json"
    plan = dict(
        focus="Primary checks: compare inverse-only with direct for real paired RGB and joint with visibility-only as a second factorial contrast. Compare joint with inverse-only and shifted for semantics. All weights fixed from training.",
        parents={str(p): digest(p) for p in (ROOT / "completed.json", MANIFEST, DRIVING, ZURICH, V33 / "rw_data.json", Path(__file__))},
        predeclared_margins=dict(rgb_target_gain=.10, source_loss_limit=.25, semantic_external_gain_pp=.50),
        populations=dict(kitti_calibrate=19, kitti_held=36, lmhaze=130, rw_new=46,
                         foggy_zurich_testv2=40, foggy_driving_fine=33),
        known_scope="KITTI evaluation uses local held raw driving sequences and artificial haze. LMHaze/RW and real fog semantic records appeared in earlier project checks. No new semantic ground truth is inferred for RGB-only scenes.",
    )
    if protocol.exists() and read(protocol) != plan:
        raise ValueError("The evaluation protocol changed")
    if not protocol.exists():
        atomic_write(protocol, plan)
    rows, counts, pictures = [], defaultdict(int), {}
    for r in load():
        if r["role"] == "fit":
            continue
        ds = "kitti_" + r["role"]
        scene, group = r["id"], r["drive"]
        clear = r["rgb"][None].cuda().float() / 255.
        target = r["label"].cuda().long()
        edges = boundary_ids(target)
        raw, transmission, _ = render(clear, r["distance"][None].cuda().float(), r["role"], scene)
        one = predictions(raw, models, anchors)
        for i, condition in enumerate(("clear", "light", "dense")):
            packet = {k: tuple(v[i:i + 1] if v is not None else None for v in value) for k, value in one.items()}
            for (arm, seed), (rgb, logits, field) in packet.items():
                add_one(rows, ds, condition, scene, group, arm, seed, rgb, logits, field,
                        clean=clear, labels=target, edges=edges, t=transmission[i:i + 1])
            if ds == "kitti_held" and counts[ds] == 0 and condition == "dense":
                pictures["kitti_held_dense"] = save_sample(f"kitti_held_dense_{scene}", raw[i:i + 1], packet, r["label"].numpy())
        counts[ds] += 1
    print(dict(kitti=dict(counts)), flush=True)
    for corpus in ("foggy_zurich_testv2", "foggy_driving_fine"):
        for scene, group, im, label in load_pairs(corpus):
            raw, _ = resize_image(im)
            y = torch.from_numpy(label.astype(np.int64, copy=False)).cuda()
            edges = boundary_ids(y)
            example = predictions(raw, models, anchors)
            for (arm, seed), (rgb, logits, t) in example.items():
                add_one(rows, corpus, "real", scene, group or capture_group(scene), arm, seed, rgb, logits, t,
                        labels=y, edges=edges)
            if counts[corpus] == 0:
                pictures[corpus] = save_sample(corpus, raw, example, label, im, restorer)
            counts[corpus] += 1
        print(dict(real_segmentation=corpus, images=counts[corpus]), flush=True)
    for records in (data_split("lmhaze"), old_samples("fresh_rw")):
        for ds, scene, group, hazy, clear in records:
            raw, target = as_float(hazy), as_float(clear)
            for (arm, seed), (rgb, logits, t) in predictions(raw, models, anchors).items():
                add_one(rows, ds, "real", scene, group, arm, seed, rgb, logits, t, clean=target)
            counts[ds] += 1
            if counts[ds] % 40 == 0:
                print(dict(real_rgb=ds, images=counts[ds]), flush=True)
    tested = dict(counts)
    if tested != plan["populations"]:
        raise ValueError(f"Incomplete datasets {tested}")
    summary, per_seed = overall(rows)
    table = ROOT / "individual.json"
    atomic_write(table, rows)
    atomic_write(ROOT / "test.json", dict(
        summary=summary, per_seed=per_seed, individual_SHA256=digest(table), protocol_SHA256=digest(protocol),
        paired_intervals=intervals(rows), images=tested, count=len(rows), visuals=pictures,
        anchors=dict(rgb_SHA256=rgb_hash, segmentation=seg_card), finished_utc=now(),
    ))
    for ds in ("kitti_held/dense", "lmhaze/real", "rw_new/real", "foggy_zurich_testv2/real", "foggy_driving_fine/real"):
        print(ds, {m: summary[f"{ds}/{m}"] for m in ("anchor", *ARMS)}, flush=True)


if __name__ == "__main__":
    evaluate()

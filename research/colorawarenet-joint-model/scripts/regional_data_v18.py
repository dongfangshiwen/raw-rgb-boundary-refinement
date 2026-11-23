"""CamVid development frames matched on the same 4:3 RGB and label grid."""
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.audit_camvid_groups_v17 import OUT as VIDEO_AUDIT
from scripts.camvid_data_v15 import DATA, fixed_fog, fog, make_batch
from scripts.data_alignment_v15 import decode_mask
from scripts.projective_ledger_v12 import digest, read


WIDTH, HEIGHT = 256, 192
VIEWS = ("camera", "uniform035", "spatial", "spatial_alt")


def prepare_grouped_data():
    audit = read(VIDEO_AUDIT / "audit.json")
    if not audit["all_passed"] or audit["manifest_SHA256"] != digest(DATA):
        raise ValueError("The audited manifest and video assignments differ")
    rows = [
        row for row in read(DATA)["semantic_records"]
        if row["source"] == "camvid" and row["partition"] in ("train", "val")
    ]
    ids, images, masks, videos = [], [], [], defaultdict(list)
    for row in rows:
        rgb_path, mask_path = row["image"], row["mask"]
        if digest(rgb_path) != row["image_SHA256"] or digest(mask_path) != row["mask_SHA256"]:
            raise ValueError("A development image or label changed: " + row["id"])
        with Image.open(rgb_path) as rgb, Image.open(mask_path) as mask:
            if rgb.size != mask.size or rgb.width * HEIGHT != rgb.height * WIDTH:
                raise ValueError("Expected matching 4:3 CamVid arrays: " + row["id"])
            images.append(np.array(rgb.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)))
            native = decode_mask(np.asarray(mask), "camvid11")
            resized = Image.fromarray(native).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
            masks.append(np.array(resized))
        ids.append(row["id"])
        videos[row["video"]].append(row["id"])
    if len(ids) != 468 or len(set(ids)) != len(ids):
        raise ValueError("Expected 468 unique source development frames")
    data = dict(raw=torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous(),
                mask=torch.from_numpy(np.stack(masks)), ids=ids)
    data["valid"] = torch.ones_like(data["mask"], dtype=torch.bool)
    index = {name: i for i, name in enumerate(ids)}

    folds = []
    for audited in audit["source_only_folds"]:
        held = audited["held_video"]
        validation = sorted(videos[held])
        if set(validation) != set(audited["validation"]["image_ids"]):
            raise ValueError("Held-out video differs from the audited list")
        core, calibration, buffer = [], [], []
        for other in sorted(set(videos) - {held}):
            frames = sorted(videos[other], key=lambda name: int(name.rsplit("_", 1)[1].removeprefix("f")))
            size = max(1, len(frames) // 5)
            lo = (len(frames) - size) // 2
            hi = lo + size
            core.extend(frames[:lo - 1] + frames[hi + 1:])
            calibration.extend(frames[lo:hi])
            buffer.extend([frames[lo - 1], frames[hi]])
        parts = dict(core=core, calibration=calibration, buffer=buffer, validation=validation)
        if sum(map(len, parts.values())) != 468 or set().union(*map(set, parts.values())) != set(ids):
            raise ValueError("Video fold must assign every frame once")
        folds.append(dict(held_video=held, ids=parts,
                          indices={name: [index[i] for i in names] for name, names in parts.items()}))
    return data, folds


def calibration_batch(data, indices, generator):
    batch = make_batch(data, indices, generator)
    size = batch["scenes"]
    raw = batch["reference"][:size]
    airlight = (0.75 + 0.25 * torch.rand(size, 3, 1, 1, generator=generator)).cuda()
    density = (0.25 + 0.65 * torch.rand(size, 1, 1, 1, generator=generator)).cuda()
    field = F.interpolate(torch.rand(size, 1, 5, 7, generator=generator).cuda(), raw.shape[-2:],
                          mode="bicubic", align_corners=False)
    other_spatial = (density + 0.45 * (field - 0.5)).clamp(0.15, 0.95)
    images = torch.cat((batch["image"], fog(raw, other_spatial, airlight)))
    return images, raw.repeat(4, 1, 1, 1)


def validation_view(reference, name):
    if name != "spatial_alt":
        return fixed_fog(reference, name)
    h, w = reference.shape[-2:]
    y = torch.linspace(0, 1, h, device=reference.device)[None, None, :, None]
    x = torch.linspace(0, 1, w, device=reference.device)[None, None, None, :]
    band_a = torch.exp(-((x - 0.30 - 0.16 * y) / 0.15).square() - ((y - 0.58) / 0.34).square())
    band_b = torch.exp(-((x - 0.79 + 0.08 * y) / 0.22).square() - ((y - 0.26) / 0.18).square())
    transmission = (0.65 - 0.47 * band_a - 0.31 * band_b).clamp(0.15, 0.85)
    airlight = reference.new_tensor([0.99, 0.91, 0.83])[None, :, None, None]
    return fog(reference, transmission, airlight)


def boundary_band(target):
    known = target != 255
    horizontal = (target[..., 1:] != target[..., :-1]) & known[..., 1:] & known[..., :-1]
    vertical = (target[..., 1:, :] != target[..., :-1, :]) & known[..., 1:, :] & known[..., :-1, :]
    boundary = (F.pad(horizontal, (0, 1)) | F.pad(horizontal, (1, 0))
                | F.pad(vertical, (0, 0, 0, 1)) | F.pad(vertical, (0, 0, 1, 0)))
    return (F.max_pool2d(boundary[:, None].float(), 7, stride=1, padding=3)[:, 0] > 0) & known

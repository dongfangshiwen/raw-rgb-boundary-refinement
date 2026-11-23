"""Build paired labelled fog views from drive-disjoint KITTI training frames."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from scripts.audit_real_fog_v17 import NAMES, city_train_ids
from scripts.depth_correlated_fog_v20 import DEPTH, DEPTH_SHA, DEPTH_SOURCE
from scripts.haze_response_v35 import linear_rgb, srgb
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ARCHIVES = REPO.parent / "benchmark_data/kitti-semantics-v40"
ROOT = REPO / "results/shared-haze-field-v40"
DATA = ARCHIVES / "drive_grouped_640.pth"
MANIFEST = ROOT / "kitti_data.json"
BENCHMARK = "https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015"
DOWNLOAD = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
GRID = (192, 640)


def rows():
    path = ARCHIVES / "devkit_scene_flow.zip"
    with ZipFile(path) as archive:
        mapping = archive.read("devkit/mapping/train_mapping.txt").decode("ascii").splitlines()
    groups = sorted({text.split()[1] for text in mapping if text},
                    key=lambda key: sha256(("v40/" + key).encode()).digest())
    if len(mapping) != 200 or len(groups) != 29:
        raise ValueError("KITTI registered mapping size changed")
    roles = dict(calibrate=set(groups[:4]), held=set(groups[4:10]), fit=set(groups[10:]))
    role_by_drive = {drive: role for role, drives in roles.items() for drive in drives}
    output = [dict(id=f"{index:06d}_10", drive=text.split()[1],
                   role=role_by_drive[text.split()[1]], recording=text)
              for index, text in enumerate(mapping) if text]
    if Counter(row["role"] for row in output) != dict(fit=87, calibrate=19, held=36):
        raise ValueError("Drive-disjoint selection changed")
    return output


def image_member(z, name):
    with Image.open(BytesIO(z.read(name))) as image:
        return np.array(image)


@torch.inference_mode()
def stage():
    configure_cuda()
    if MANIFEST.exists():
        card = read(MANIFEST)
        if card["SHA256"] != digest(DATA):
            raise ValueError("Prepared KITTI data changed")
        return card
    if digest(DEPTH / "model.safetensors") != DEPTH_SHA:
        raise ValueError("Depth proxy checkpoint changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    chosen = rows()
    archive_path = ARCHIVES / "data_semantics.zip"
    hashes = set()
    with ZipFile(archive_path) as archive:
        for record in chosen:
            name = record["id"]
            rgb = image_member(archive, f"training/image_2/{name}.png")
            label = image_member(archive, f"training/semantic/{name}.png")
            if rgb.shape[:-1] != label.shape or rgb.shape[-1] != 3:
                raise ValueError("KITTI RGB / label geometry differs")
            signature = sha256(rgb.tobytes()).hexdigest()
            if signature in hashes:
                raise ValueError("Duplicate input photograph crosses planned draws")
            hashes.add(signature)
            pixels = Image.fromarray(rgb).resize(GRID[::-1], Image.Resampling.BILINEAR)
            mask = Image.fromarray(city_train_ids(label)).resize(GRID[::-1], Image.Resampling.NEAREST)
            record.update(rgb=torch.from_numpy(np.array(pixels)).permute(2, 0, 1),
                          label=torch.from_numpy(np.array(mask)), original_SHA256=signature, size=list(label.shape))
    processor = AutoImageProcessor.from_pretrained(DEPTH, use_fast=False, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH, local_files_only=True,
                                                       use_safetensors=True).cuda().eval()
    for offset in range(0, len(chosen), 8):
        block = chosen[offset:offset + 8]
        images = [r["rgb"].permute(1, 2, 0).numpy() for r in block]
        inputs = processor(images, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            depth = model(pixel_values=inputs.pixel_values.cuda()).predicted_depth.float()[:, None]
        depth = F.interpolate(depth, GRID, mode="bicubic", align_corners=False)
        low, high = torch.quantile(depth.flatten(1), depth.new_tensor([.02, .98]), dim=1)
        relative = 1. - ((depth - low[:, None, None, None]) /
                         (high - low).clamp_min(1e-6)[:, None, None, None]).clamp(0., 1.)
        if not torch.isfinite(relative).all() or bool((high - low < 1e-6).any()):
            raise ValueError("Invalid KITTI depth proxy")
        for r, distance in zip(block, relative.cpu().half(), strict=True):
            r["distance"] = distance
        if offset % 40 == 0 or offset + 8 >= len(chosen):
            print(dict(staged=min(offset + 8, len(chosen)), total=len(chosen)), flush=True)
    torch_save(DATA, chosen)
    card = dict(
        source=BENCHMARK, license="KITTI CC BY-NC-SA 3.0 (Geiger et al.)", official_download=DOWNLOAD,
        source_SHA256=digest(archive_path), mapping_SHA256=digest(ARCHIVES / "devkit_scene_flow.zip"),
        SHA256=digest(DATA), data=str(DATA), grid=list(GRID), classes=list(NAMES), member_ids=len(chosen),
        unresolved_drive_omitted=58, exact_duplicates=0, officially_hidden_test_loaded=0,
        groups={role: sorted({r["drive"] for r in chosen if r["role"] == role})
                for role in ("fit", "calibrate", "held")},
        split_frames=dict(Counter(r["role"] for r in chosen)),
        depth_source=DEPTH_SOURCE, depth_checkpoint_SHA256=DEPTH_SHA,
        character="KITTI official training images split by raw drive. Dense distance is a relative monocular estimate. Synthetic pixel-aligned attenuation uses it to form fog image pairs; it is not ground-truth depth.",
        stage_SHA256=digest(__file__), finished_utc=now(),
    )
    atomic_write(MANIFEST, card)
    print(card["split_frames"], flush=True)
    return card


def load():
    meta = read(MANIFEST)
    if meta["SHA256"] != digest(DATA):
        raise ValueError("The prepared KITTI tensor file differs")
    return torch.load(DATA, map_location="cpu", weights_only=True)


def render(reference, distance, role, key):
    """Render one clear and two aligned linear-RGB fog conditions per scene."""
    rng = torch.Generator().manual_seed(int.from_bytes(sha256(("fog40/" + key).encode()).digest()[:4], "big"))
    air = torch.empty((3, 3, 1, 1)).uniform_(.58, .95, generator=rng).to(reference)
    if role == "fit":
        beta = torch.tensor([0., .7 + .25 * float(torch.rand((), generator=rng)),
                             1.35 + .40 * float(torch.rand((), generator=rng))]).to(reference)
        field = torch.rand((2, 1, 4, 8), generator=rng).to(reference)
        field = F.interpolate(field, reference.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        modulation = torch.cat((torch.ones_like(field[:1]), .85 + .30 * field), 0)
    else:
        beta = reference.new_tensor([0., .90, 1.70])
        h, w = reference.shape[-2:]
        y = torch.linspace(0, 1, h, device=reference.device)[None, None, :, None]
        x = torch.linspace(0, 1, w, device=reference.device)[None, None, None, :]
        field = .90 + .20 * torch.sin(2 * torch.pi * x) * torch.cos(torch.pi * y)
        modulation = torch.cat((torch.ones_like(field), field, field.flip(-1)), 0)
    optical = beta[:, None, None, None] * (.17 + .83 * distance) * modulation
    t = (-optical).exp()
    inputs = srgb(t * linear_rgb(reference) + (1. - t) * air).clamp(0., 1.)
    return inputs, t, air


if __name__ == "__main__":
    stage()

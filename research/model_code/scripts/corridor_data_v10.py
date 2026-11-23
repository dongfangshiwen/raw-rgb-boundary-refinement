"""Versioned sequence folds and paired opaque-occlusion interventions for V10.

Crosswalk191 changes role from a previously inspected external screen to development
training/CV data. Its filled author corridors are NOT verified visible-occluder labels.
Synthetic opacity labels are exact for the pasted image intervention only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "protocols/corridor-visibility-v10"
PUBLIC = REPO / "results/architecture-v2/external-crosswalk191/manifest.json"
OWN = REPO / "protocols/crosswalk-partial-background-v8/manifest.json"
VISUAL_GROUPS = (("01", "02"), ("03",), ("04", "09"), ("05", "06"), ("07", "08"))
FOLDS = (("01", "02"), ("03", "05", "06"), ("04", "07", "08", "09"))
VAL = (("03",), ("01", "02"), ("05", "06"))
SIZE = 384


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def partition(sequence, fold):
    if sequence in FOLDS[fold]:
        return "test"
    return "val" if sequence in VAL[fold] else "train"


def register():
    public, own = read(PUBLIC), read(OWN)
    public_root = Path(public["root"])
    records = []
    for row in public["records"]:
        record = dict(row, image=str((public_root / row["image"]).resolve()), mask=str((public_root / row["mask"]).resolve()),
            folds=[partition(row["sequence"], f) for f in range(3)])
        if digest(record["image"]) != row["image_sha256"] or digest(record["mask"]) != row["mask_sha256"]:
            raise ValueError("Public data changed")
        records.append(record)
    value = dict(version=10, public_manifest_sha256=digest(PUBLIC), own_manifest_sha256=digest(OWN),
        public_test_groups=[list(x) for x in FOLDS], public_validation_groups=[list(x) for x in VAL],
        conservative_visual_groups=[list(x) for x in VISUAL_GROUPS],
        grouping_evidence="Pretraining-only inspection of9sequence first images: conservatively join01/02(reciprocaljunctionviews),04/09(office/railingviews),05/06(vineyard/stonewallroad),07/08(parkintersection);03separate. Visual similarity only, no GPS or claimed provenindependence. Over-grouping safer than splitting a possiblysharedcrossing. No labels/modelpredictions used to choose folds.",
        public_records=records,
        own_records=[dict(id=r["id"], partition=r["partition"], clear=r["clear"], hazy=r["hazy"],
            clear_sha256=digest(r["clear"]), hazy_sha256=digest(r["hazy"]), mask=r["draft_mask"], mask_sha256=r["draft_sha256"])
            for r in own["records"]],
        role_change="Crosswalk191 now DEVELOPMENT TRAIN/CV, previously inspected many times. Outer predictions are exploratory sequence-held-out CV, not a fresh external test. No frame-level random split; each sequence has one role per fold. Original185 split stays fixed.",
        ontology="Filled crosswalk corridor INCLUDING stripe gaps. Author whole-frame masks supply negatives, but may include people/cars or omit other corridors: NOT expert-confirmed all-visible occluder-excluding GT. SDF derives this annotation, not true 3D/amodal geometry.",
        occlusion="Train-only opaque rectangular crops from public TRAIN backgrounds (all-zero author-mask rectangles). Paste into clear/reference BEFORE same known fog. Visible labels become0 exactly on the opaque patch; original corridor labels retained for auxiliary support. Clear reference used only during training. No claim that artificial occlusion proves real person/car exclusion.",
        mask_decoding="Public storedPNG0/255 ->binary0/1 by>0; own partial0/1/255 staysunchanged (255unknown). Filesneveroverwritten.",
        labels_unchanged=True, weak_v9_not_used=True, extra_public_supervision_declared=True,
        fpv2025="Official metadata lists300real/3000synthetic, but current public files/folders endpoints return403 and cached roots empty; NOT downloaded or used. No results claimed.")
    path = ROOT / "manifest.json"
    if path.exists() and read(path) != value:
        raise ValueError("Frozen V10 data protocol changed")
    if not path.exists():
        write(path, value)
    return value


def signed_distance(mask, radius=24):
    """Exact truncated Manhattan distance, positive on author corridor.

    Iterated four-neighbour erosion avoids an extra scientific dependency.
    Image exterior is not treated as annotated background.
    """
    foreground = (mask == 1).astype(np.uint8)
    inside, outside = foreground.copy(), 1-foreground
    distance = np.zeros(mask.shape, np.float32)
    for _ in range(radius):
        distance += inside.astype(np.float32) - outside.astype(np.float32)
        for value in (inside, outside):
            padded = np.pad(value, 1, mode="edge")
            value[:] = np.minimum.reduce((padded[1:-1, 1:-1], padded[:-2, 1:-1],
                padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]))
    distance /= radius
    # Coordinates96x96, distance expressed in384-pixel units.
    return np.asarray(Image.fromarray(distance).resize((96, 96), Image.Resampling.BILINEAR)).copy()


def load_rows(rows, own=False):
    import torch
    raw, masks, distances, ids = [], [], [], []
    for row in rows:
        pair = []
        roles = ("clear", "hazy") if own else ("image", "image")
        for role in roles:
            with Image.open(row[role]) as image:
                array = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)).copy()
                pair.append(torch.from_numpy(array).permute(2, 0, 1))
        with Image.open(row["mask"]) as image:
            mask = np.asarray(image.resize((SIZE, SIZE), Image.Resampling.NEAREST)).copy()
        if not own:
            if not set(np.unique(mask)).issubset({0,255}):
                raise ValueError("Expected public stored binary0/255 mask")
            mask = (mask > 0).astype(np.uint8)
        if not set(np.unique(mask)).issubset({0, 1, 255} if own else {0, 1}):
            raise ValueError("Mask values changed")
        masks.append(torch.from_numpy(mask))
        distances.append(torch.zeros(1, 96, 96) if own else torch.from_numpy(signed_distance(mask))[None])
        raw.append(torch.stack(pair)); ids.append(row["id"])
    return dict(raw=torch.stack(raw), mask=torch.stack(masks), distance=torch.stack(distances), ids=ids,
                complete=torch.full((len(ids),), not own, dtype=torch.bool))


def prepare_data(fold, manifest):
    import torch
    # The outer test rows are deliberately NOT loaded by the training process.
    result = {}
    for part in ("train", "val"):
        result["own_" + part] = load_rows([r for r in manifest["own_records"] if r["partition"] == part], True)
        result["public_" + part] = load_rows([r for r in manifest["public_records"] if r["folds"][fold] == part])
    # A small deterministic bank of opaque background crops, made ONLY from this fold's train sequences.
    bank = []
    public = result["public_train"]
    for i, ident in enumerate(public["ids"]):
        mask = public["mask"][i].numpy()
        for y, x in ((0, 0), (0, 144), (48, 280), (240, 0), (240, 240)):
            if not (mask[y:y+96, x:x+96] != 0).any():
                patch = public["raw"][i, 0, :, y:y+96, x:x+96]
                if patch.shape == (3, 96, 96):
                    bank.append(patch)
    if len(bank) < 20:
        raise ValueError("Insufficient all-background train-only donor crops")
    result["donors"] = torch.stack(bank)
    return result


def synthesize(clear, random_generator, strata):
    """Sample a stratified transmission field and channel-wise airlight."""
    import torch
    import torch.nn.functional as F

    n = len(clear)
    device = clear.device
    draw = torch.rand(n, generator=random_generator)
    mean = (0.4 + 0.55 * (strata.float() + draw) / 6).to(device)[:, None, None, None]
    noise = torch.rand((n, 1, 4, 4), generator=random_generator).to(device) * 2 - 1
    noise = noise - noise.mean((2, 3), keepdim=True)
    noise = noise / noise.abs().amax((2, 3), keepdim=True).clamp_min(1e-6)
    amplitude = 0.5 * torch.minimum(mean - 0.25, 0.95 - mean)
    transmission = mean + amplitude * F.interpolate(
        noise, clear.shape[-2:], mode="bilinear", align_corners=False
    )
    airlight = (0.75 + 0.25 * torch.rand((n, 3, 1, 1), generator=random_generator)).to(device)
    return transmission * clear + (1 - transmission) * airlight, transmission, airlight


def make_batch(data, own_indices, public_indices, generator, strata):
    import torch
    import torch.nn.functional as F
    own, public = data["own_train"], data["public_train"]
    indices = (own_indices, public_indices)
    raw = torch.cat([d["raw"][i] for d, i in zip((own, public), indices)]).cuda().float() / 255
    mask = torch.cat([d["mask"][i] for d, i in zip((own, public), indices)]).cuda().long()
    distance = torch.cat([d["distance"][i] for d, i in zip((own, public), indices)]).cuda()
    complete = torch.cat([d["complete"][i] for d, i in zip((own, public), indices)]).cuda()
    n = len(raw)
    flips = (torch.rand(n, generator=generator) > .5).cuda()
    raw = torch.where(flips[:, None, None, None, None], raw.flip(-1), raw)
    mask = torch.where(flips[:, None, None], mask.flip(-1), mask)
    distance = torch.where(flips[:, None, None, None], distance.flip(-1), distance)
    clear, hazy = raw[:, 0], raw[:, 1].clone()
    # Public images have no real fog/clean pair. Their reference is original RGB,
    # not asserted clear or radiometrically calibrated. The second view is fixed mild fog.
    hazy[len(own_indices):] = .8 * clear[len(own_indices):] + .2 * .9
    synthetic, transmission, airlight = synthesize(clear, generator, strata)
    occlusion = torch.zeros_like(mask, dtype=torch.bool)
    pasted = clear.clone()
    for j in range(n):
        side = int(torch.randint(48, 97, (), generator=generator))
        foreground = (mask[j] == 1).nonzero()
        if len(foreground) and bool(torch.rand((), generator=generator) < .75):
            center = foreground[int(torch.randint(len(foreground), (), generator=generator))]
            y = max(0, min(SIZE-side, int(center[0])-side//2))
            x = max(0, min(SIZE-side, int(center[1])-side//2))
        else:
            y, x = [int(torch.randint(SIZE-side+1, (), generator=generator)) for _ in range(2)]
        donor = data["donors"][int(torch.randint(len(data["donors"]), (), generator=generator))].cuda().float()[None] / 255
        patch = F.interpolate(donor, (side, side), mode="bilinear", align_corners=False)[0]
        pasted[j, :, y:y+side, x:x+side] = patch
        occlusion[j, y:y+side, x:x+side] = True
    occluded_fog = transmission * pasted + (1-transmission) * airlight
    visible = mask.clone(); visible[occlusion] = 0
    return dict(image=torch.cat((clear, hazy, synthetic, occluded_fog)),
        reference=torch.cat((clear, clear, clear, pasted)), visible_mask=torch.cat((mask, mask, mask, visible)),
        support_mask=mask, distance=distance, complete=complete, occlusion=occlusion,
        transmission=transmission, airlight=airlight, scene_count=n)


def contact_sheet(manifest):
    # Fold0 train annotation examples only. Nine raw sequence views were inspected
    # separately for conservative grouping, before registration/training/predictions.
    rows = [r for r in manifest["public_records"] if r["folds"][0] == "train"]
    selected = rows[::max(1, len(rows)//18)][:18]
    page = Image.new("RGB", (1440, 1080), "white"); draw = ImageDraw.Draw(page)
    for i, r in enumerate(selected):
        with Image.open(r["image"]) as im:
            im = im.convert("RGB").resize((470, 240))
        with Image.open(r["mask"]) as m:
            a = np.asarray(m.resize((470, 240), Image.Resampling.NEAREST))
        rgb = np.asarray(im).copy(); rgb[a > 0] = (.55*rgb[a > 0] + .45*np.array([10, 230, 100])).astype(np.uint8)
        x, y = (i%3)*480, (i//3)*180
        page.paste(Image.fromarray(rgb).resize((470, 156)), (x, y+22)); draw.text((x+4, y+4), r["id"], fill="black")
    path = REPO / "results/corridor-visibility-v10/data-audit/fold0-train-labels.png"
    path.parent.mkdir(parents=True, exist_ok=True); page.save(path)


if __name__ == "__main__":
    manifest = register(); contact_sheet(manifest)
    print(json.dumps(dict(images=len(manifest["public_records"]), folds=[{p:sum(r["folds"][f]==p for r in manifest["public_records"]) for p in ("train", "val", "test")} for f in range(3)])))

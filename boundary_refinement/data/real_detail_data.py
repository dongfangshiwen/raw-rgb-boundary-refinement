"""Read verified native haze/clear pairs without extracting extra RGB copies."""
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import torch

from boundary_refinement.data.provenance import REPO, digest, read
from boundary_refinement.data.real_haze_data import ROOT as V22, align, member_index, read_member
from boundary_refinement.training.train_boundary_consistency import ROOT as V23


ROOT = REPO / "results/real-detail-v24"
DATASETS = ("nh_holdout", "dense", "ohaze", "i_haze")


class RealPairs:
    """Register native and letterbox views by the same pair IDs."""
    def __init__(self, datasets):
        self.catalogue = read(V22 / "data_manifest.json")
        self.held = read(V22 / "held_out_images.json")
        self.indoor = read(V23 / "i_haze_manifest.json")
        self.sources = {"nh_train": (self.catalogue["training_archive"], self.catalogue["training"]["rows"]),
                        "nh_holdout": (self.catalogue["training_archive"], None),
                        "dense": (self.catalogue["dense_archive"], self.catalogue["evaluation"]["dense"]),
                        "ohaze": (None, self.catalogue["evaluation"]["ohaze"]),
                        "i_haze": (dict(path=self.indoor["archive"], SHA256=self.indoor["archive_SHA256"]), self.indoor["pairs"])}
        self.archives = {}
        for source in datasets:
            file, _ = self.sources[source]
            if file is not None:
                path = file["path"]
                if digest(path) != file["SHA256"]:
                    raise ValueError("Source archive changed: " + source)
                if path not in self.archives:
                    self.archives[path] = ZipFile(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for archive in self.archives.values():
            archive.close()

    def ids(self, source):
        if source == "nh_train":
            return [row["id"] for row in self.catalogue["training"]["rows"]]
        if source == "i_haze":
            excluded = {row["id"] for row in self.indoor["flagged"]}
            return [row["id"] for row in self.indoor["pairs"] if row["id"] not in excluded]
        return self.held["subsets"][source]

    def read(self, source, scene):
        file, members = self.sources[source]
        if scene not in self.ids(source):
            raise ValueError("Pair does not belong to the registered data role")
        archive = self.archives[file["path"]] if file else None
        if members is None:
            by_role = member_index(archive)[scene]
            output = [read_member(archive, by_role[role])[0] for role in ("hazy", "clean")]
        else:
            row = next(r for r in members if r["id"] == scene)
            output = []
            for role in ("hazy", "clean"):
                evidence = row[role]
                if archive:
                    image, check = read_member(archive, archive.getinfo(evidence["member"]))
                    if check["SHA256"] != evidence["SHA256"]:
                        raise ValueError("An archived RGB file changed")
                else:
                    if digest(evidence["path"]) != evidence["SHA256"]:
                        raise ValueError("An O-HAZE member changed")
                    with Image.open(evidence["path"]) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                output.append(image)
        if output[0].size != output[1].size:
            raise ValueError("Native RGB pair shape differs")
        return tuple(output)


def long_edge(image, size):
    w, h = image.size
    return image.resize((round(w*size/max(w, h)), round(h*size/max(w, h))), Image.Resampling.BILINEAR)


def image_tensor(image):
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()


def letterbox_rgb(hazy, clear):
    image, roi = align(hazy)
    target, target_rect = align(clear)
    if target_rect != roi:
        raise ValueError("RGB regions do not align")
    return image, target, roi

"""Make depth-aligned and reflected fields with exactly matched transmission values."""
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from scripts.camvid_data_v15 import DATA, fog
from scripts.projective_ledger_v12 import REPO, digest, now, read
from scripts.regional_data_v18 import HEIGHT, WIDTH, prepare_grouped_data
from scripts.train_regional_guidance_v18 import atomic_write, configure_cuda, torch_save


ROOT = REPO / "results/physical-guidance-v20"
DEPTH = REPO / "pretrained/depth-anything-v2-small-hf"
DEPTH_SHA = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
DEPTH_SOURCE = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf"
TEST_VIEWS = ("camera", "aligned_light", "reflected_light", "aligned_dense", "reflected_dense")


@torch.inference_mode()
def infer_proxy():
    configure_cuda()
    if digest(DEPTH / "model.safetensors") != DEPTH_SHA:
        raise ValueError("Pinned depth weights changed")
    path = ROOT / "depth_proxy.pth"
    if path.exists() or (ROOT / "depth_proxy.json").exists():
        raise ValueError("Preserve the existing depth proxy and its provenance")
    ROOT.mkdir(parents=True, exist_ok=True)
    source, _ = prepare_grouped_data()
    processor = AutoImageProcessor.from_pretrained(DEPTH, local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH, local_files_only=True, use_safetensors=True
    ).cuda().eval()
    values = []
    for start in range(0, len(source["ids"]), 6):
        rgb = source["raw"][start:start + 6].permute(0, 2, 3, 1).numpy()
        inputs = processor(images=list(rgb), return_tensors="pt")
        depth = model(pixel_values=inputs.pixel_values.cuda()).predicted_depth[:, None]
        grid = F.interpolate(depth, (HEIGHT, WIDTH), mode="bicubic", align_corners=False)
        tails = torch.quantile(grid.flatten(1), grid.new_tensor([0.02, 0.98]), dim=1)
        floor, ceiling = tails[0, :, None, None, None], tails[1, :, None, None, None]
        distance = 1 - ((grid - floor) / (ceiling - floor).clamp_min(1e-6)).clamp(0, 1)
        if not torch.isfinite(distance).all() or bool(((ceiling - floor) < 1e-6).any()):
            raise ValueError("Invalid relative distance signal")
        values.append(distance.cpu().half())
        if start % 60 == 0:
            print(dict(depth_proxies=min(start + 6, len(source["ids"])), time=now()), flush=True)
    proxy = torch.cat(values)
    torch_save(path, dict(ids=source["ids"], proxy=proxy, manifest_SHA256=digest(DATA), revision=REVISION))
    metadata = dict(all_passed=True, file=str(path), SHA256=digest(path), frames=len(proxy),
                    source=DEPTH_SOURCE, revision=REVISION, weights_SHA256=DEPTH_SHA, license="Apache-2.0",
                    scope="Relative depth estimated only from the CamVid development camera images. Uses 2nd and 98th percentiles. Controls synthetic haze and is unavailable to restoration at inference.",
                    grid=[WIDTH, HEIGHT], independent_test_images=0, finished_utc=now())
    atomic_write(ROOT / "depth_proxy.json", metadata)
    print(metadata, flush=True)


def load_proxy(source):
    meta = read(ROOT / "depth_proxy.json")
    if not meta["all_passed"] or digest(meta["file"]) != meta["SHA256"]:
        raise ValueError("Depth tensor differs from the registered file")
    actual = torch.load(meta["file"], map_location="cpu", weights_only=True)
    if (actual["ids"] != source["ids"] or actual["manifest_SHA256"] != digest(DATA)
            or actual["revision"] != REVISION):
        raise ValueError("Depth proxies do not align to the development frames")
    source["proxy_distance"] = actual["proxy"]
    return source


def render(reference, optical_depth, airlight):
    transmission = optical_depth.neg().exp()
    image = (fog(reference, transmission, airlight) * 255).round() / 255
    return image, transmission


def augmented_batch(data, selected, generator):
    raw = data["raw"][selected].cuda().float() / 255
    depth = data["proxy_distance"][selected].cuda().float()
    size = len(raw)
    flip = (torch.rand(size, generator=generator) > 0.5).cuda()[:, None, None, None]
    raw = torch.where(flip, raw.flip(-1), raw)
    depth = torch.where(flip, depth.flip(-1), depth)
    field = F.interpolate(torch.rand(size, 1, 5, 7, generator=generator).cuda(), raw.shape[-2:],
                          mode="bicubic", align_corners=False).clamp(0, 1)
    beta = (0.40 + 1.40 * torch.rand(size, 1, 1, 1, generator=generator)).cuda()
    airlight = (0.74 + 0.24 * torch.rand(size, 3, 1, 1, generator=generator)).cuda()
    optical = beta * (0.15 + 0.85 * depth) * (0.87 + 0.26 * field)
    aligned, aligned_t = render(raw, optical, airlight)
    reflected, reflected_t = render(raw, optical.flip((-2, -1)), airlight)
    return dict(image=torch.cat((raw, aligned, reflected)), clean=raw.repeat(3, 1, 1, 1),
                t=torch.cat((torch.ones_like(aligned_t), aligned_t, reflected_t)),
                air=airlight.repeat(3, 1, 1, 1), n=size)


def fixed_view(reference, depth, name):
    if name not in TEST_VIEWS:
        raise ValueError(name)
    if name == "camera":
        return reference, torch.ones_like(depth)
    beta = 1.30 if name.endswith("light") else 1.80
    airlight = reference.new_tensor([0.83, 0.91, 0.96])[None, :, None, None]
    height, width = reference.shape[-2:]
    y = torch.linspace(0, 1, height, device=reference.device)[None, None, :, None]
    x = torch.linspace(0, 1, width, device=reference.device)[None, None, None, :]
    spatial = 0.93 + 0.14 * (0.5 + 0.5 * torch.sin(2 * torch.pi * x) * torch.cos(torch.pi * y))
    optical = beta * (0.15 + 0.85 * depth) * spatial
    if name.startswith("reflected"):
        optical = optical.flip((-2, -1))
    return render(reference, optical, airlight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    infer_proxy()

"""Frozen dual-output stress test under two exactly registered sRGB contrast losses."""
from pathlib import Path
from hashlib import sha256
import numpy as np
from PIL import Image
import torch

from boundary_refinement.inference import idd_fog_models as joint
from boundary_refinement.evaluation.photographic import photo_calibration as calibration
from boundary_refinement.evaluation.semantic import range_transfer as previous
from boundary_refinement.baselines.pgl_net import load_pgl, pgl_rgb
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.evaluation.semantic.eval_native_guide import load_image
from boundary_refinement.evaluation.semantic.idd_fog_evaluate import Counts
from boundary_refinement.models.operators.gaussian_guide import scores as gaussian_scores
from boundary_refinement.models.operators.native_guide import PixelGrid, interpolate
from boundary_refinement.data.cohorts.native_reserve import ROOT, SEEDS
from boundary_refinement.data.provenance import digest, read, now
from boundary_refinement.baselines.segfix_peer import SegFix
from boundary_refinement.training.train_real_transfer import author_rgb
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda

STRENGTH = (.65, .40)
PHOTO = ("input", "mct", "pgl", "scalar", "dehaze_seg_s_local")
SEM = ("released", "segfix", "fixed_native", "matched_gaussian_fixed", "dehaze_seg_s_local")

def save_example(directory, pictures, masks, reference):
    directory.mkdir(exist_ok=True, parents=True)
    for name, image in {**pictures, "reference":reference}.items():
        arr = image[0].permute(1,2,0).mul(255).round().clamp(0,255).byte().cpu().numpy()
        Image.fromarray(arr).save(directory / (name + "-rgb.png"))
    for name, labels in masks.items():
        Image.fromarray(labels[0].cpu().numpy()).save(directory / (name + "-labels.png"))

def native_masks(z, released, small, raw, offsets):
    shape = raw.shape[-2:]
    bilinear = resize(z, shape).argmax(1)[0].byte()
    orig = released.argmax(1)[0].byte()
    masks = {"released":[orig], "segfix":[orig.flatten()[offsets].reshape(shape)],
             "fixed_native":[bilinear.clone()], "matched_gaussian_fixed":[bilinear.clone()]}
    grid = PixelGrid(z, small, raw)
    for ids in grid.active().nonzero().flatten().split(32768):
        if not len(ids):
            continue
        f = grid.four(ids, include_distance=True)
        masks["fixed_native"][0].view(-1)[ids] = interpolate(f[0],f[1],f[3]).argmax(-1).byte()
        masks["matched_gaussian_fixed"][0].view(-1)[ids] = gaussian_scores(f).argmax(-1).byte()
    return masks

@torch.inference_mode()
def run():
    configure_cuda()
    plan = ROOT / "paired-veil-protocol.json"
    photo_weights = calibration.OUT / "photo-weights.json"
    fit = read(photo_weights)["models"]["all/scalar"]
    assert fit["slopes"] == [0.] * 3 and len(fit["training_cities"]) == 23
    alpha = 1 / (1 + np.exp(-fit["bias"]))
    deps = [Path(__file__), plan, ROOT / "manifest.json", photo_weights,
            *(joint.pdf_path("published", seed) for seed in SEEDS), joint.CAM]
    lock = sha256("".join(map(digest, deps)).encode()).hexdigest()
    rows = [r for r in read(ROOT / "manifest.json")["rows"] if r["cohort"] == "bdd"]
    assert len(rows) == len(set(r["group"] for r in rows)) == 128
    target, refiner = previous.Target("mask2former"), SegFix()
    mct, _ = calibration.ex.road.load_source()
    pgl, _ = load_pgl()
    peers = [joint.trained_peer(s) for s in SEEDS]
    first = peers[0]
    directory = ROOT / "paired-counts"
    directory.mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        path = directory / (row["id"] + ".npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                assert archive["lock"].item() == lock
            continue
        rgb, truth = load_image(row)
        scorer = Counts(truth)
        original, (h, w) = joint.peer_input(rgb)
        reference = original[...,:h,:w]
        metrics, confusions = [], []
        for t in STRENGTH:
            veiled = np.rint(t * rgb.astype(np.float32) + (1 - t) * .90 * 255).clip(0,255).astype(np.uint8)
            padded, size = joint.peer_input(veiled)
            assert size == (h,w)
            small = padded[...,:h,:w].contiguous()
            raw = torch.from_numpy(veiled).cuda().permute(2,0,1)[None].float().div(255)
            base = author_rgb(mct,small).clamp(0,1)
            images = {"input":small,"mct":base,"pgl":pgl_rgb(pgl,small).clamp(0,1),
                      "scalar":small + alpha*(base-small)}
            offsets = refiner.lookup(veiled)
            z, released = target(small, truth.shape)
            masks = native_masks(z, released, small, raw, offsets)
            del released,z
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image,*features = first.backbone((padded-first.mean)/first.std)
                images["dehaze_seg_s_local"] = (image.float()*first.std+first.mean)[...,:h,:w].clamp(0,1)
                masks["dehaze_seg_s_local"] = [
                    resize(joint.peer_forward(p,features,size).float(), truth.shape).argmax(1)[0].byte() for p in peers]
            metrics.append([float(-10*(images[k]-reference).square().mean().clamp_min(1e-12).log10()) for k in PHOTO])
            banks = []
            for method in SEM:
                labels = masks[method]
                c = np.stack([np.stack(scorer.score(m)) for m in labels], axis=1)
                banks.append(np.repeat(c,len(SEEDS),axis=1) if len(labels)==1 else c)
            confusions.append(np.stack(banks).astype(np.int32))
            if i==0:
                save_example(ROOT/"paired-examples"/str(t), images, masks, reference)
        out = np.stack(confusions)
        assert np.all(out[:,:,1]<=out[:,:,0]) and np.all(out.sum(-1)==out[:1,:1,:,:1].sum(-1))
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as f:
            np.savez_compressed(f, id=row["id"],group=row["group"],lock=lock,psnr=metrics,
                                counts=out,transmission=STRENGTH,photo=PHOTO,semantic=SEM,seeds=SEEDS)
        tmp.replace(path)
        if i==0 or (i+1)%32==0:
            print("Paired RGB and masks",i+1,len(rows),flush=True)
    atomic_write(ROOT/"paired-evaluation.json",dict(lock=lock,plan=digest(plan),selection=digest(ROOT/"manifest.json"),
                 photo_coefficients=digest(photo_weights),alpha=float(alpha),frames=len(rows),conditions=STRENGTH,
                 photo=PHOTO,semantic=SEM,seeds=SEEDS,example=rows[0]["id"],finished=now()))

if __name__=="__main__":
    run()

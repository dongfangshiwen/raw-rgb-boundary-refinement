"""Paired RGB and segmentation predictions from the same fog input, with both interventions crossed."""
from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

from boundary_refinement.metrics.segmentation import confusion_matrix
from boundary_refinement.models.restoration import bounded_tone as color
from boundary_refinement.training import luminance_blend as calib
from boundary_refinement.evaluation.semantic import evaluate_prototype_risk as diagnostic
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.training import prototype_risk as gate
from boundary_refinement.training import road_heads as road
from boundary_refinement.models.operators import donor_colors as weighted
from boundary_refinement.models.operators import subpixel_transport as base
from boundary_refinement.models.segmentation.coarse_semantics import paint
from boundary_refinement.models.segmentation.contrastive_response import raw_anchor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.models.restoration.edge_projection import draw
from boundary_refinement.evaluation.paired.evaluate_shared_haze_field import to_rgb
from boundary_refinement.training.haze_response import linear_rgb, srgb
from boundary_refinement.metrics.color import srgb_to_lab
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.models.segmentation.restoration_affinity import boundary_band, compress_scores
from boundary_refinement.training.train_regional_guidance import atomic_write, configure_cuda


ROOT = calib.ROOT
RGB = ("affine", "luma", *calib.GATES)
SEM = ("mean", "fixed")
SEEDS = base.SEEDS
ARMS = tuple((a,b) for a in RGB for b in SEM)


def aggregate(rows, metric):
    datasets = defaultdict(dict)
    for r in rows:
        groups = datasets[r["ds"]]
        if r["group"] not in groups:
            groups[r["group"]] = np.zeros((len(ARMS), 3, 19, 19), dtype=np.int64)
        groups[r["group"]][ARMS.index((r["rgb"],r["sem"])),SEEDS.index(r["seed"])] += r[metric]
    out = {}
    for ds, groups in datasets.items():
        n = len(groups)
        draw_bank = diagnostic.draws(n)
        matrices = np.stack(list(groups.values()))
        values = {}
        for j, arm in enumerate(ARMS):
            parts = []
            for draws in np.array_split(draw_bank, max(1,len(draw_bank)//80)):
                c = (draws @ matrices[:,j].reshape(n,-1)).reshape(-1,3,19,19)
                diag = c.diagonal(axis1=-2,axis2=-1)
                totals = c.sum(-2)+c.sum(-1)-diag
                iou = np.divide(100.*diag, totals, out=np.zeros_like(diag,dtype=float), where=totals>0)
                parts.append(iou.mean((1,2)))
            values[arm] = np.concatenate(parts)
        effects = {a+"-fixed_minus_mean":values[a,"fixed"]-values[a,"mean"] for a in RGB}
        effects["luma-affine_at_fixed"] = values["luma","fixed"]-values["affine","fixed"]
        effects["factor_interaction"] = effects["luma-fixed_minus_mean"]-effects["affine-fixed_minus_mean"]
        effects["curve-constant_at_fixed"] = values["curve","fixed"]-values["constant","fixed"]
        out[ds] = dict(groups=n, scores={"/".join(k):v[0] for k,v in values.items()},
                       effects={k:diagnostic.interval(v) for k,v in effects.items()})
    return out


@torch.inference_mode()
def run():
    configure_cuda()
    params = calib.load()
    path = ROOT / "joint-protocol.json"
    cfg = dict(script=digest(__file__), parent=calib.plan(), coefficients=digest(ROOT / "coefficients.json"),
               selection=digest(held.SCENES), frozen_decoder=digest(weighted.ROOT / "completed.json"),
               transmissions=[.70,.40], airlight_linear=.85, rgb_seed=road.SEEDS[0], segmentation_seeds=list(SEEDS),
               rgb_resolution="native renderer working grid against the same antialiased clear RGB grid",
               segmentation="native labels; paired bootstrap by recording. Neutral, uniform linear-light visibility "
                            "stress; no metric depth or real measured atmospheric transmission. Inputs rounded to uint8.")
    if path.exists() and read(path) != cfg:
        raise ValueError("Joint factorial recipe changed")
    if not path.exists():
        atomic_write(path, cfg)
    engine = base.DualScaleFeatures()
    controlled, decoders, raws = calib.heads(), weighted.load(), base.anchors()
    seg_rows, photo_rows, gallery, shown = [], [], [], set()
    seed_rgb = road.SEEDS[0]
    for ds, key, group, daytime, original, truth in held.scene_cases():
        if ds != "new_muses_clear":
            continue
        target = torch.from_numpy(original).cuda().permute(2,0,1)[None].float()/255.
        for t in cfg["transmissions"]:
            source = "MUSES_t" + str(round(100*t))
            fog = to_rgb(srgb(t*linear_rgb(target)+(1.-t)*.85))
            views, deployed, _ = engine(fog)
            grid = views.fine.shape[-2:]
            fine = resize(torch.from_numpy(fog).cuda().permute(2,0,1)[None].float()/255., deployed.shape[-2:], antialias=True)
            ref = resize(target, deployed.shape[-2:], antialias=True)
            g = road.geometry(fine, engine.core.rgb)
            anchor, a, w = calib.endpoints(g, engine.core.bank, controlled, seed_rgb)
            torch.testing.assert_close(w, deployed, atol=1e-5, rtol=1e-5)
            pictures = dict(affine=a,luma=w,**{m:calib.mixture(anchor,a,w,m,params[m,seed_rgb]) for m in calib.GATES})
            phase = [b%4 for b in sha256(key.encode()).digest()[:2]]
            ref_lab = srgb_to_lab(ref[...,phase[0]::4,phase[1]::4])
            lum = color.luminance(anchor)[...,phase[0]::4,phase[1]::4].flatten()
            bins = torch.bucketize(lum, lum.new_tensor(color.BINS[1:-1]), right=True)
            for m, p in dict(input=fine, **pictures).items():
                photo_rows.append(dict(ds=source,key=key,group=group,daytime=daytime,arm=m,seed=seed_rgb,
                                       **color.stats(p,ref,phase,bins,ref_lab)))
            guides = {m:base.render_guides(fog, p, grid) for m,p in pictures.items()}
            edge = truth.masked_fill(~boundary_band(truth),255)
            example = None
            for s in SEEDS:
                z = raw_anchor(raws["raw_detail",s],views)
                native = resize(z,truth.shape)
                template = base.guide_stencil(guides["luma"][:0],z.argmax(1)[0])
                fixed = weighted.color_weights(z,"dual_core")
                weights = dict(fixed=fixed,mean=torch.full_like(fixed,1./16))
                for m, sem in ARMS:
                    stencil = weighted.prepare(guides[m],z,"dual_core",template,weights[sem])
                    delta = base.prediction("dual_free",decoders["dual_core",s],z,stencil)
                    pred = (native+resize(delta,truth.shape)).argmax(1)[0]
                    seg_rows.append(dict(ds=source,id=key,group=group,daytime=daytime,rgb=m,sem=sem,seed=s,
                                         counts=confusion_matrix(pred,truth,19).tolist(),
                                         edge=confusion_matrix(pred,edge,19).tolist()))
                    if (s == SEEDS[0] and m=="luma" and sem=="fixed" and daytime not in shown and t==.40):
                        example = paint(pred.cpu().byte().numpy())
            if example is not None:
                gallery.append((f"{key} | {daytime} | transmission 0.40",
                               [fog, to_rgb(w), to_rgb(pictures["curve"]), original, example, paint(truth.cpu().byte().numpy())],
                               ["Veiled input", "Window RGB", "Calibrated RGB", "Clear target", "Predicted labels", "True labels"]))
                shown.add(daytime)
        if len(photo_rows) % 120 == 0:
            print("joint pairs",len(photo_rows)//6,flush=True)
    n = read(held.SCENES)["cohorts"]["clear"]["selected"]
    if len(seg_rows)!=n*2*len(SEEDS)*len(ARMS) or len(photo_rows)!=n*2*6:
        raise ValueError("Incomplete factorial")
    rgb_file, cm_file = ROOT/"joint-rgb.json", ROOT/"joint-seg.json.gz"
    atomic_write(rgb_file,photo_rows)
    compress_scores(cm_file,seg_rows)
    draw(ROOT/"paired-predictions.png",gallery)
    atomic_write(ROOT/"joint-test.json",dict(protocol=digest(path),rgb_table=digest(rgb_file),cm_table=digest(cm_file),
                 photos=calib.summary(photo_rows),edge=aggregate(seg_rows,"edge"),full=aggregate(seg_rows,"counts"),
                 clear_images=n,synthetic_views=2*n,examples=digest(ROOT/"paired-predictions.png"),utc=now()))
    print("paired factorial complete", n, flush=True)


if __name__=="__main__":
    run()

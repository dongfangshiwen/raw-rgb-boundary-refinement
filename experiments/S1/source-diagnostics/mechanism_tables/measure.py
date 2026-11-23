"""Collect aligned view patches and check the two worked examples used on Figure 3."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE.parents[2] / "research" / "model_code"))
from scripts import joint_tone_prototype_v99 as joint
from scripts import real_detail_data_v24 as paired
from scripts import relative_utility_v56 as utility
from scripts.train_regional_guidance_v18 import configure_cuda

def array(t):
    return t.detach().cpu().numpy()

@torch.inference_mode()
def run():
    configure_cuda()
    old = json.loads((SOURCE/"a4/worked-examples.json").read_text())
    cell = json.loads((SOURCE/"cell-example.json").read_text())
    seed = old["seed"]
    mct,_ = joint.road.load_source()
    heads = joint.road.load_trained("rgb")
    with paired.RealPairs(("ohaze",)) as pool:
        raw, target = [paired.image_tensor(paired.long_edge(im, min(576,max(im.size))))[None].cuda().float()/255
                       for im in pool.read("ohaze", "023")]
    geom = joint.road.geometry(raw, mct)
    u, a, w = joint.calib.endpoints(geom, heads, joint.calib.heads(), seed)
    coeff = joint.calib.load()["curve",seed]
    rgb = joint.calib.mixture(u,a,w,"curve",coeff)
    _,(est,_,rms) = joint.road.mix_weights(geom,heads["utility",seed],"utility")
    _,_,norm = utility.relative_geometry(geom.gram,geom.features.new_zeros((1,1,1,3)))
    proj = est*rms
    steps = (proj/norm.clamp_min(utility.EPSILON)).clamp(0,1)
    gain = 2*proj*steps-norm*steps.square()
    i,j = old["cell_row_col"]
    selected = int(gain[0,i,j].argmax())
    np.testing.assert_allclose(array(steps[0,i,j]),old["step"],atol=1e-6)
    np.testing.assert_allclose(array(gain[0,i,j]),old["gain"],atol=1e-7)
    assert selected == old["candidate_id"] == 3
    source = geom.base
    local = source + steps[0,i,j,selected]*geom.deltas[selected-1:selected]
    observed = float(geom.partition.average(((source-target).square()-(local-target).square()).mean(1))[0,i,j])
    np.testing.assert_allclose(observed,old["direct_local_mse_reduction"],atol=1e-7)
    # All four crops are centered on this overlapping cell's maximum basis weight.
    cy = int(geom.partition.rows[:,i].argmax())
    cx = int(geom.partition.cols[:,j].argmax())
    def crops(t):
        return array(t[:,:,cy-36:cy+36,cx-36:cx+36].permute(0,2,3,1).contiguous())
    patches = crops(torch.cat([source,source+geom.deltas[:1],source+geom.deltas[1:2],raw]))
    assert patches.shape == (4,72,72,3)
    py,px = old["pixel_row_col"]
    lum = joint.color.luminance(u)
    alpha = (joint.calib.hats(lum,"curve")*coeff[None,:,None,None]).sum(1)
    np.testing.assert_allclose(array(coeff),old["knots"],atol=1e-6)
    np.testing.assert_allclose([float(lum[0,0,py,px]),float(alpha[0,py,px])],
                               [old["luminance"],old["window_fraction"]],atol=1e-6)
    for name,t in (("affine",a),("windowed",w),("blended",rgb)):
        np.testing.assert_allclose(array(t[0,:,py,px]),old["pixel_rgb"][name],atol=1e-6)
    frame = joint.weighted.ROOT/"examples/foggy_driving_fine"
    image, fine = np.asarray(Image.open(frame/"input.png").convert("RGB")), np.asarray(Image.open(frame/"truth.png"))
    y,x = [4*e for e in cell["cell_row_col"]]
    is_class = fine[y:y+4,x:x+4] == cell["owner_id"]
    np.testing.assert_array_equal(is_class,cell["children"])
    np.testing.assert_array_equal(image[y:y+4,x:x+4],cell["native_grid_rgb_4x4"])
    mass = np.asarray(cell["percent_weights"])
    np.testing.assert_allclose([100*is_class.mean(), mass[is_class].sum()],cell["correct_mass_percent"],atol=1e-5)
    neighborhood = image[y-22:y+26,x-22:x+26].copy()
    nh = fine[y-22:y+26,x-22:x+26].copy()
    winning_steps = steps[0].gather(-1,gain[0].argmax(-1,keepdim=True)).squeeze(-1)
    arrays = dict(candidate_patches=patches, step_map=array(winning_steps),
                  context_rgb=neighborhood,context_labels=nh)
    np.savez_compressed(HERE/"figure3-examples.npz",**arrays)
    ledger = dict(photo=old["photo"],seed=seed,candidate_order=old["candidate_order"],
                  cell_row_col=[i,j],crop_center_yx=[cy,cx],crop_bounds_xyxy=[cx-36,cy-36,cx+36,cy+36],
                  candidate_steps=array(steps[0,i,j]).tolist(),predicted_gain=float(gain[0,i,j,selected]),
                  observed_gain=observed, gain_unit="reduction in local mean squared error in unit RGB",
                  pixel_row_col=[py,px],luminance=float(lum[0,0,py,px]),alpha=float(alpha[0,py,px]),
                  knots=array(coeff).tolist(),cell_semantic=cell["cell_row_col"],semantic_frame=old["semantic"]["id"],
                  true_pixels=int(is_class.sum()),donor_mass_percent=[float(100*is_class.mean()),float(mass[is_class].sum())],
                  raw_rgb_rmse=cell["raw_rmse"], patch_storage="float32 unit RGB, no independent contrast stretching",
                  local_semantic_box_xyxy=[22,22,26,26],torch=torch.__version__,device=torch.cuda.get_device_name(),
                  arrays_sha256={n:hashlib.sha256(v.tobytes(order="C")).hexdigest() for n,v in arrays.items()})
    (HERE/"figure3-review.json").write_text(json.dumps(ledger,indent=2),encoding="utf8")
    print(json.dumps({k:ledger[k] for k in ("candidate_steps","predicted_gain","observed_gain","luminance","alpha","donor_mass_percent","crop_center_yx","device")}))

if __name__=="__main__":
    run()

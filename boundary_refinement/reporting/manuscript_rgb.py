"""Supplement the registered scalar baseline with the manuscript's actual three RGB heads."""
import numpy as np
from PIL import Image
import torch
from boundary_refinement.training import luminance_blend as calibration
from boundary_refinement.training import road_heads as road
from boundary_refinement.evaluation.paired.paired_photo import tensor
from boundary_refinement.models.restoration.cross_scale_rgb import resize
from boundary_refinement.data.provenance import digest
from boundary_refinement.training.train_regional_guidance import configure_cuda
from boundary_refinement.data.staging.weatherproof_files import OUT,WORK,read,write

@torch.inference_mode()
def run():
    configure_cuda()
    inputs=read(OUT/"rgb-individual.json")
    prior={r["id"]:r for r in inputs["rows"]}
    registration=dict(status="Additional frozen manuscript check, registered after the primary scalar and segmentation outcomes.",
        sample="All 180 previously registered WeatherProof triplets, identical image geometry and no target fitting.",
        rgb="Run the original local-utility RGB selector and luminance-calibrated mixture with each of the three source weights. Compare its frame-level PSNR against input, public MCT, PGL-Net-T and the already frozen scalar. Match each recording's semantic evaluation to its photographic outcomes.",
        weights=digest(calibration.ROOT/"coefficients.json"),source_digest=digest(OUT/"rgb-individual.json"),code=digest(__file__))
    reg=OUT/"manuscript-rgb-protocol.json"
    if reg.exists():
        original_registration=read(reg)
        assert original_registration=={**registration,"code":original_registration["code"]}
    else:write(reg,registration)
    models=calibration.heads(); params=calibration.load(); bank=road.load_trained("rgb")
    author,_=road.load_source()
    assert list(calibration.SEEDS)==[32003,32017,32029]
    chosen=read(OUT/"evaluation.json")["example_ids"]["wp"]
    results=[]
    for row in read(OUT/"wp-manifest.json")["rows"]:
        ref=prior[row["id"]]
        src=[WORK/p for p in row["paths"][:2]]
        assert list(map(digest,src))==row["sha256"][:2]
        x,y=[resize(tensor(p),tuple(ref["size"]),antialias=True) for p in src]
        g=road.geometry(x,author)
        assert abs(float((g.base-y).double().square().mean())-ref["mse"]["mct"])<1e-8
        seeds={}
        for s in calibration.SEEDS:
            anchor,affine,window=calibration.endpoints(g,bank,models,s)
            image=calibration.mixture(anchor,affine,window,"curve",params["curve",s])
            mse=(image-y).double().square().mean().item()
            seeds[s]=dict(mse=mse,psnr=float(-10*np.log10(max(mse,1e-12))))
            if row["id"]==chosen and s==calibration.SEEDS[0]:
                a=image[0].permute(1,2,0).clamp(0,1).mul(255).round().byte().cpu().numpy()
                Image.fromarray(a).save(OUT/"examples/wp/photographs/manuscript.png")
        results.append(dict(id=row["id"],group=row["group"],seeds=seeds))
        if len(results)%60==0:print("Original RGB branch",len(results),flush=True)
    write(OUT/"manuscript-rgb-individual.json",dict(rows=results,protocol=digest(reg),seeds=list(calibration.SEEDS),
          implementation=digest(__file__),note="Original RGB and semantic modules have different source seed IDs. Average each endpoint separately before the strict per-camera co-improvement check."))
if __name__=="__main__":run()

"""Export self-contained decoder coefficients and verify native masks and archived counts on CUDA."""
from pathlib import Path
from hashlib import sha256
import numpy as np
from PIL import Image
import torch
from scripts import apply_path_v105 as portable
from scripts import eval_anchor_v105 as run
from scripts import path_decoder_v104 as op
from scripts.reserve_anchor_v105 import OUT
from scripts.weatherproof_release_v103 import WORK,read,write

@torch.inference_mode()
def check():
    run.prior.base.configure_cuda();old,anchors,extra=run.fit.load();nets=run.prior.train.load()
    nets["fixed_grid",0]=op.Decoder("gaussian").cuda().eval()
    folder=OUT/"inference";folder.mkdir(exist_ok=True)
    for s in op.SEEDS:
        run.prior.train.source.torch_save(folder/f"path-{s}.pth",dict(range_by_view=(3*old["dual_core",s].range_weights.sigmoid()).cpu(),
                      spatial=anchors[s].exponent.cpu(),strength=extra["path",s].cpu()))
    models={"mask2former":run.prior.base.prior.Target("mask2former"),"ade":run.ADE()};records=[]
    for panel in ("bdd","wp"):
        rows=read(OUT/(panel+".json"))["rows"]
        for i in (0,1,13):
            row=rows[i];rgb,truths=run.prior.old.open_frame(row)
            raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
            padded,(h,w)=run.prior.base.peer_input(rgb);small=padded[...,:h,:w].contiguous()
            if panel=="wp":
                small=run.prior.base.resize(raw,tuple(max(32,min(512,32*round(t/32))) for t in (h,w)),antialias=True)
                with Image.open(WORK/row["paths"][-1]) as im:y=np.array(im)
                if y.ndim==3:y=y[:,:,0]
            file=OUT/"counts"/panel/(row["id"].replace("/","_")+".npz")
            for target in (("mask2former","ade") if panel=="wp" else ("mask2former",)):
                net=models[target];z,released=net(small,rgb.shape[:2]);road=target!="ade"
                mq=run.predictions(z,released,small,old,anchors,extra,nets,road)
                gt,lut=run.maps(y,(2,0,10) if road else net.ids[:5]) if panel=="wp" else (truths[0],None)
                scorer=run.prior.old.SparseCounts(gt);errors=[]
                suite=target if road else "ade5";arms=run.ROAD if road else run.CORE
                with np.load(file,allow_pickle=False) as f:archived=f[suite]
                for j,s in enumerate(op.SEEDS):
                    m=portable.predict(z,small,rgb.shape[:2],folder/f"path-{s}.pth",road_manuscript=road).byte().cpu().numpy()
                    different=int(np.count_nonzero(m!=mq["path"][j]));errors.append(different)
                    cm=scorer.cm((m if lut is None else lut[m]).ravel())
                    np.testing.assert_array_equal(cm,archived[:,arms.index("path"),j])
                    assert different==0,(row["id"],target,s,different)
                    if i==0 and road and j==0:
                        with Image.open(OUT/"examples"/panel/"path.png") as saved:np.testing.assert_array_equal(np.asarray(saved),m)
                records.append(dict(panel=panel,id=row["id"],target=target,classes=z.shape[1],different_pixels_by_seed=errors,native=rgb.shape[:2]))
    a=read(run.prior.OUT/"random.json")["rows"];b=read(OUT/"bdd.json")["rows"];f=read(run.prior.OUT/"fog-new.json")["rows"]
    hashes=[r["sha256"][0] for r in a+b+f];groups=[r["group"] for r in a+b+f]
    assert len(set(hashes))==len(set(groups))==len(hashes)==260
    count=lambda p:read(p)["archived_six_arm_count_matches"]
    report=dict(native_mask_checks=records,exact_native_masks=3*len(records),cuda=torch.version.cuda,pytorch=str(torch.__version__),
                weights={p.name:sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.pth")},
                new_bdd_unique_prefixes=len(set(groups)),new_bdd_unique_image_sha256=len(set(hashes)),
                v104_prior_frames_six_methods_exact=sum(count(run.prior.OUT/(k+"-evaluation.json")) for k in ("idd","old_fog","weatherproof")))
    write(OUT/"inference-checks.json",report)
    print("Native masks identical",report["exact_native_masks"],"historical frames exact",report["v104_prior_frames_six_methods_exact"],flush=True)

if __name__=="__main__":check()

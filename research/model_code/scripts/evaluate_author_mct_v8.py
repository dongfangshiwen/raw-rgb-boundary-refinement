"""Author HF DehazeFormer-MCT reference, different mixed-data pretraining."""
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path

from scripts import evaluate_parallel_rgb_v8 as rgb
from scripts.evaluate_parallel_joint_v8 import read,write,digest,ROOT,REPO

OFFICIAL=(Path(__file__).resolve().parents[2] / "benchmark_data" / 'official-dehazeformer-research')


def load_author_reference():
    registration=read(OFFICIAL/"reference_registration.json");folder=OFFICIAL/"hf-author-demo"
    for relative,sha in registration["source_sha256"].items():
        if digest(folder/relative)!=sha:raise ValueError("Author checkpoint/source changed")
    specification=importlib.util.spec_from_file_location("author_dehazeformer_mct_research",folder/"models/dehazeformer.py")
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    class CUDACompatibleMCT(module.MCT):
        def get_coord(self,image):
            # The author's CPU demo creates these coordinate tensors on CPU.
            # Retain their exact values and only move them to the input device.
            x,y=super().get_coord(image)
            return x.to(image.device),y.to(image.device)
    model=CUDACompatibleMCT()
    packet=rgb.torch.load(folder/"saved_models/dehazeformer.pth",map_location="cpu",weights_only=True)
    model.load_state_dict(packet["state_dict"],strict=True)
    return model.cuda().eval(),registration


def main():
    from scripts.evaluate_partial_sharing_v8b import training_lock
    _,registered,_=training_lock()
    implementation=dict(code_sha256=digest(__file__),reference_registration_sha256=digest(OFFICIAL/"reference_registration.json"),
        scoring_sha256=digest(REPO/"scripts/evaluate_parallel_rgb_v8.py"),
        cuda_change="Only move author CPU coordinate tensors to image device, no learned tensor or mathematical change. Never run Gradio app.",
        normalized_inference="Commonreflectpad16, input*2-1, authorMCTforward, clamp[-1,1]*.5+.5, unpad; float metricsbeforePNGquantization.",
        role="Extra pretrainedRGB reference withdifferent/unspecifiedmixedtrainingdata, not a matched-budget or independent-test guarantee")
    path=ROOT/"author_mct_implementation.json"
    if path.exists() and read(path)!=implementation:raise ValueError("Author reference evaluator changed")
    if not path.exists():write(path,implementation)
    if digest(registered["rgb_benchmarks"]["manifest"])!=registered["rgb_benchmarks"]["manifest_sha256"]:raise ValueError("Reference benchmark changed")
    rgb.bind();torch=rgb.torch;F=rgb.F
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,registration=load_author_reference()
    # Verify numerical equivalence of the device bridge against the original CPU method.
    with torch.inference_mode():
        probe=torch.linspace(-1,1,3*64*80).reshape(1,3,64,80)
        cpu_model=model.cpu();expected=cpu_model(probe).clone()
        model=model.cuda();actual=model(probe.cuda()).cpu()
        difference=float((actual-expected).abs().max())
        if difference>5e-4:raise ValueError("CPU/CUDA author-model bridge differs excessively")
    manifest=read(registered["rgb_benchmarks"]["manifest"]);rows=[]
    with torch.inference_mode():
        for i,record in enumerate(manifest["pairs"]):
            image,clear=rgb.pair(record);h,w=image.shape[-2:];right,bottom=-w%16,-h%16
            padded=F.pad(image,(0,right,0,bottom),mode="reflect") if right or bottom else image
            out=(model(padded*2-1).clamp(-1,1)*.5+.5)[:,:,:h,:w]
            rows.append(dict(dataset=record["dataset"],id=record["id"],
                psnr=float(-10*(out-clear).square().mean().clamp_min(1e-10).log10()),ssim_gaussian11_valid=rgb.gaussian_ssim_valid(out,clear),
                l1=float((out-clear).abs().mean()),ciede2000=rgb.ciede2000_rgb(out,clear)))
            if i%100==0:write(ROOT/"author_mct_status.json",dict(images=i+1,total_images=1010,updated_utc=datetime.now(timezone.utc).isoformat()))
    summary=[]
    for dataset in manifest["datasets"]:
        chosen=[r for r in rows if r["dataset"]==dataset]
        summary.append(dict(dataset=dataset,count=len(chosen),**{k:sum(r[k] for r in chosen)/len(chosen) for k in ("psnr","ssim_gaussian11_valid","l1","ciede2000")}))
    write(ROOT/"author_mct_results.json",dict(method="author_DehazeFormer_MCT_mixed_HF",checkpoint_sha256=registration["checkpoint_sha256"],
        parameters=sum(p.numel() for p in model.parameters()),CPU_CUDA_probe_max_abs=difference,
        images=rows,summary=summary,training_differs=True,not_matched_or_claimed_paper_T_B_checkpoint=True,finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(reference="author_DehazeFormer_MCT_mixed_HF",images=len(rows),summary=summary)))


if __name__=="__main__":main()

"""Export full-size restored RGB and a separate raw-image Cityscapes19 mask."""
import argparse
from pathlib import Path

import torch

from boundary_refinement.training.calibrate_expert_choice import ROOT, switches
from boundary_refinement.evaluation.semantic.evaluate_real_semantics import load_segmenter
from boundary_refinement.evaluation.paired.evaluate_road_outputs import export
from boundary_refinement.models.restoration.expert_fusion import experts
from boundary_refinement.data.provenance import digest, read
from boundary_refinement.training.train_regional_guidance import configure_cuda


def load_router():
    params = ROOT/"calibration.pth"
    receipt = read(ROOT/"evaluation.json")
    if digest(params)!=receipt["calibration_SHA256"] or digest(ROOT/"plan.json")!=receipt["plan_SHA256"]:
        raise ValueError("The calibrated routing state changed")
    return torch.load(params,map_location="cuda",weights_only=True),receipt


@torch.inference_mode()
def run(path,destination):
    configure_cuda()
    model,receipt = load_router()
    forward,_ = experts()
    segmenter,mean,std,card = load_segmenter()
    info = dict(method="guarded",calibration_SHA256=receipt["calibration_SHA256"])

    def restore(image):
        mct,pgl = (f(image).clamp(0,1) for f in forward)
        selected = switches(model,image,mct,pgl)["guarded"]
        info["selected_expert"] = "PGL-Net-T" if selected else "MCT"
        return pgl if selected else mct

    return export(path,destination,restore,segmenter,mean,std,card,info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    result = run(args.input,args.output)
    print(dict(native_size=result["native_size"],expert=result["restoration"]["selected_expert"],
               files=list(result["output_SHA256"])),flush=True)


if __name__=="__main__":
    main()

"""Export full-size restored RGB and a separate raw-image Cityscapes19 mask."""
import argparse
from pathlib import Path

import torch

from scripts.calibrate_expert_choice_v27 import ROOT, switches
from scripts.evaluate_real_semantics_v22 import load_segmenter
from scripts.evaluate_road_outputs_v25 import export
from scripts.expert_fusion_v26 import experts
from scripts.projective_ledger_v12 import digest, read
from scripts.train_regional_guidance_v18 import configure_cuda


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

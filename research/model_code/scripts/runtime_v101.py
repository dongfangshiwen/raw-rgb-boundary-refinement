"""Capture the effective CUDA research runtime without local account information."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import sys

import torch
from scripts.fit_range_v101 import OUT
from scripts.projective_ledger_v12 import REPO,digest,read
from scripts.train_regional_guidance_v18 import configure_cuda,atomic_write

def main():
    configure_cuda()
    ps="$os=Get-CimInstance Win32_OperatingSystem; $cpu=Get-CimInstance Win32_Processor |Select-Object -First 1;" \
       "[PSCustomObject]@{os=$os.Caption;build=$os.BuildNumber;cpu=$cpu.Name;" \
       "installed_ram_bytes=(Get-CimInstance Win32_PhysicalMemory|Measure-Object Capacity -Sum).Sum}|ConvertTo-Json -Compress"
    windows=json.loads(subprocess.check_output(["powershell","-NoProfile","-Command",
       "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"+ps],encoding="utf-8-sig"))
    prop=torch.cuda.get_device_properties(0)
    prev={}
    for name in ("semantic-subpixel-v94","resolution-interaction-v84","subpixel-transport-v92"):
        source=REPO/"results"/name/"completed.json"
        card=read(source);prev[name]=dict(torch=card["torch"],gpu=card["gpu"],record_sha256=digest(source))
    card={"captured_utc":datetime.now(timezone.utc).isoformat(),"scope":"Measured in the environment used for the new CUDA fits and inference",
          "host":windows,"system":platform.platform(),"python":sys.version.split()[0],"packages":{n:version(n) for n in
             ("torch","torchvision","transformers","numpy","scipy","Pillow","safetensors")},
          "gpu":prop.name,"gpu_total_bytes_pytorch":prop.total_memory,"cuda_compute_capability":list(torch.cuda.get_device_capability()),
          "pytorch_cuda_build":torch.version.cuda,"cudnn_runtime":list(torch._C._cudnn.getRuntimeVersion()),
          "nvidia_driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),
          "cpu_threads_pytorch":torch.get_num_threads(),"allow_tf32_matmul":torch.backends.cuda.matmul.allow_tf32,
          "allow_tf32_cudnn":torch.backends.cudnn.allow_tf32,"strict_deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
          "seeds":[101,211,307],"bootstrap_seed":100481,"bootstrap_draws":5000,
          "precision":"Frozen target-backbone forward BF16 autocast. Query-to-class aggregation and decoder/resampling FP32. Source logits cached FP16, cast to FP32 for fitting. AnyUp FP32 with q_chunk_size 512. Photometric MCT inference FP32; scalar/conditional gate optimizers use FP64; confusion counts integer.",
          "historical_records":prev,"current_fit_sha256":{p:digest(OUT/p) for p in ("training.json","axis-weights.json","photo-weights.json")}}
    atomic_write(OUT/"environment.json",card)
    print(card["host"],card["packages"],card["nvidia_driver"],flush=True)
if __name__=="__main__":main()

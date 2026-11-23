"""Small runtime helpers shared by command-line workflows."""
import csv
import hashlib
import json
import platform
import random
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import torch


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=1)
def cuda_runtime_error():
    """A visible GPU may still lack kernels in the installed PyTorch build."""
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        probe = torch.ones(1, device="cuda")
        (probe + probe).sum().item()
    except RuntimeError as exc:
        return str(exc).splitlines()[0]
    return None


def select_device(name="auto"):
    if name == "cpu":
        return torch.device("cpu")
    error = cuda_runtime_error()
    if error:
        if name == "cuda":
            raise ValueError(f"CUDA runtime check failed: {error}. Use --device cpu or a PyTorch build supporting your GPU.")
        if torch.cuda.is_available():
            warnings.warn(f"CUDA runtime check failed; using CPU: {error}", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")
    return torch.device("cuda")


def write_json(path, value):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_metadata(device):
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return dict(python=platform.python_version(), platform=platform.platform(), torch=str(torch.__version__),
                packages={name: version(name) for name in ("torchvision", "numpy", "Pillow", "tqdm")},
                cuda=torch.version.cuda, device=str(device), cpu_threads=torch.get_num_threads(),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                compute_capability=list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
                package_source_sha256=digest.hexdigest())


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    ensure_dir(path.parent)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

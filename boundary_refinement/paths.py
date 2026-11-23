"""Filesystem locations for code, provider datasets and registered measurements."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
# Existing workspaces can point directly at their model data and fitted checkpoints.
MODEL_DATA = Path(os.environ.get("RGB_BOUNDARY_MODEL_DATA", SOURCE_ROOT / "work" / "model")).expanduser().resolve()
WORKSPACE = MODEL_DATA.parent


def record_path(path):
    """Return the original relative data key or the implementation path in this checkout."""
    path = Path(path)
    if path.is_relative_to(MODEL_DATA):
        return str(path.relative_to(MODEL_DATA))
    return path.relative_to(SOURCE_ROOT).as_posix()

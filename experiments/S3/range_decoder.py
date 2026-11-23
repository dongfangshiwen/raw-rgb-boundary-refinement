"""Shared range kernel for readers of the S3 supplement."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boundary_refinement.models.operators.range_kernel import FAMILIES, RANGES, RangeDecoder, prepare

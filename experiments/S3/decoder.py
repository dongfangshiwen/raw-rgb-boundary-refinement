"""Frozen score-grid decoder used by the S3 supplement."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from boundary_refinement.inference.score_grid import AdaptiveConv, RangeDecoder, bilateral, decode, prepare, resize

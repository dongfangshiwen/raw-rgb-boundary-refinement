"""Camera prediction entry point for the S2 supplement."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from boundary_refinement.inference.camera import PALETTE, RangeDecoder, verify_inputs, filehash, main, mct_from, native, predict, resize, semantic_scores, shorter


if __name__ == "__main__":
    main()

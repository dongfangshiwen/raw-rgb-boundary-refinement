"""Run the four frozen decoders from the S4 entry point."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.frozen_decoders.frozen_inference import main, run


if __name__ == "__main__":
    main()

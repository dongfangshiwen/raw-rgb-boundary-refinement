"""Recheck the four-decoder counts in an expanded S4 experiment."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.frozen_decoders.replay import compute, main, verify


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)

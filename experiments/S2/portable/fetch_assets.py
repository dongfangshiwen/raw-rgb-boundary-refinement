"""Asset preparation entry point for the S2 supplement."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from boundary_refinement.data.model_assets import FILES, install, main, verify


if __name__ == "__main__":
    main()

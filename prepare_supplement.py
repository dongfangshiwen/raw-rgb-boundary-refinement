"""Prepare a separate data workspace from the four numbered publication archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def extract(archive, destination):
    destination.mkdir(parents=True)
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            entry = (destination / name).resolve()
            if not entry.is_relative_to(destination):
                raise ValueError(f"Unexpected path in {archive.name}: {name}")
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("supplement", type=Path, help="Directory containing the extracted supplement's archives folder")
    parser.add_argument("destination", type=Path, help="New directory for the expanded records and weights")
    args = parser.parse_args()
    dst = args.destination.resolve()
    if dst.exists():
        parser.error(f"The destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    hashes = json.loads((ROOT / "Archive checksums.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=dst.parent) as temp:
        staged = Path(temp).resolve() / "records"
        for key in ("S1", "S2", "S3", "S4"):
            filename = f"Supplementary Data {key}.zip"
            path = args.supplement / "archives" / filename
            if not path.is_file():
                parser.error(f"Missing {path}. Extract the outer publication supplement first.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[filename]:
                raise ValueError(f"Archive checksum mismatch: {filename}")
            extract(path, staged / key)
        inner = staged / "S1" / "source-diagnostics"
        extract(inner / "Original supplementary archive.zip", inner / "original")
        staged.rename(dst)
    print(f"Prepared S1–S4 in {dst}")


if __name__ == "__main__":
    main()

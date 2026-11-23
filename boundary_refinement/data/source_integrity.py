"""Match historical training-source fingerprints when opening a registered fit."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _source_pairs():
    path = Path(__file__).with_name("released_source_digests.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    return {(old, row["current"]) for row in records.values() for old in row["released"]}


def matches_registered(recorded, current):
    """Compare every protocol field, allowing only the listed code digest pairs."""
    if recorded == current:
        return True
    if isinstance(current, dict) and isinstance(recorded, dict):
        return recorded.keys() == current.keys() and all(
            matches_registered(recorded[key], value) for key, value in current.items()
        )
    if isinstance(current, (list, tuple)) and isinstance(recorded, type(current)):
        return len(recorded) == len(current) and all(
            matches_registered(a, b) for a, b in zip(recorded, current)
        )
    if isinstance(current, str) and isinstance(recorded, str) and len(current) == len(recorded) == 64:
        return (recorded, current) in _source_pairs()
    return False

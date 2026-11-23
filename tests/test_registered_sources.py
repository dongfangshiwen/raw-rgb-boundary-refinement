"""Regression checks for accepting the published frozen fits after the package move."""
import hashlib
import json
import unittest

from boundary_refinement.data.source_integrity import matches_registered
from boundary_refinement.paths import SOURCE_ROOT


class RegisteredSourceTests(unittest.TestCase):
    def setUp(self):
        record_file = SOURCE_ROOT / "boundary_refinement/data/released_source_digests.json"
        self.records = json.loads(record_file.read_text(encoding="utf-8"))
        donor = self.records["boundary_refinement/models/operators/donor_colors.py"]
        self.released = donor["released"][0]
        self.current = donor["current"]

    def test_fingerprints_match_the_checked_out_sources(self):
        for path, record in self.records.items():
            with self.subTest(path=path):
                digest = hashlib.sha256((SOURCE_ROOT / path).read_bytes()).hexdigest()
                self.assertEqual(digest, record["current"])
                for released in record["released"]:
                    self.assertTrue(matches_registered(released, digest))

    def test_published_and_fresh_recipes(self):
        recorded = {"source": self.released, "seeds": [101, 211, 307], "weights": "a" * 64}
        current = {**recorded, "source": self.current}
        self.assertTrue(matches_registered(recorded, current))
        self.assertTrue(matches_registered(current, current))
        self.assertFalse(matches_registered(recorded, {**current, "seeds": [101, 307, 211]}))
        self.assertFalse(matches_registered(recorded, {**current, "weights": "b" * 64}))

    def test_unknown_and_cross_module_sources_are_rejected(self):
        self.assertFalse(matches_registered(self.released, "0" * 64))
        other = self.records["boundary_refinement/training/road_heads.py"]["current"]
        self.assertFalse(matches_registered(self.released, other))


if __name__ == "__main__":
    unittest.main()

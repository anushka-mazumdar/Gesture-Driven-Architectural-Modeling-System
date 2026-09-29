"""Local consistency check for the experimental ML feature vectors."""

import json
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ml_features import FEATURE_COUNT, extract_features


class MLFeaturesTest(unittest.TestCase):
    def test_processed_samples_have_fixed_finite_feature_vectors(self):
        processed_dir = PROJECT_ROOT / "dataset" / "processed"
        sample_paths = sorted(processed_dir.glob("*/*.json"))[:8]
        self.assertGreaterEqual(
            len(sample_paths), 3,
            f"Need at least three processed samples under {processed_dir}",
        )

        vectors = []
        for sample_path in sample_paths:
            with sample_path.open("r", encoding="utf-8") as sample_file:
                sample = json.load(sample_file)
            vector = extract_features(sample["points"])
            self.assertEqual(len(vector), FEATURE_COUNT, sample_path.name)
            self.assertTrue(all(math.isfinite(value) for value in vector), sample_path.name)
            vectors.append(vector)

        self.assertEqual(len({len(vector) for vector in vectors}), 1)


if __name__ == "__main__":
    unittest.main()

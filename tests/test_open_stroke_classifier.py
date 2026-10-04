"""Focused deterministic checks for the open-stroke classifier."""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.stroke_analysis import analyze
from shapes.stroke_classifier import ShapeClass, StrokeClassifier
from shapes.stroke_normalization import preprocess


OPEN_LABELS = {
    "straight_line", "polyline", "arc", "curve", "freeform_path",
}


def classify_open(points):
    normalized = preprocess(points, n=64)
    return StrokeClassifier(use_ml_model=False).classify(
        points, analyze(normalized), closed=False
    )


class OpenStrokeClassifierTest(unittest.TestCase):
    def test_representative_open_strokes(self):
        cases = {
            "straight_line": [(i, 2) for i in range(20)],
            "polyline": [(0, 0), (20, 0), (20, 20), (40, 20)],
            "arc": [
                (100 * math.cos(math.radians(deg)),
                 100 * math.sin(math.radians(deg)))
                for deg in range(0, 151, 3)
            ],
            "curve": [(i, 20 * math.sin(i / 8)) for i in range(80)],
            "freeform_path": [
                (i, (10 if i % 2 else -10) + (8 if i % 5 == 0 else 0))
                for i in range(30)
            ],
        }

        for expected, points in cases.items():
            with self.subTest(expected=expected):
                result = classify_open(points)
                self.assertEqual(result.kind, expected)
                self.assertTrue(math.isfinite(result.confidence))
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)

    def test_ambiguous_noisy_and_degenerate_inputs_are_safe_and_repeatable(self):
        cases = [
            [(i, ((i * 37) % 11) - 5) for i in range(30)],
            [(i, (i % 3) - 1) for i in range(20)],
            [(4, 4)] * 5,
            [],
        ]

        for points in cases:
            with self.subTest(point_count=len(points)):
                original = list(points)
                first = classify_open(points)
                second = classify_open(points)
                self.assertIn(first.kind, OPEN_LABELS | {"uncertain"})
                self.assertEqual(first.kind, second.kind)
                self.assertEqual(first.confidence, second.confidence)
                self.assertTrue(math.isfinite(first.confidence))
                self.assertEqual(points, original)

        invalid = StrokeClassifier(use_ml_model=False).classify(
            [], object(), closed=False
        )
        self.assertEqual(invalid.kind, "uncertain")
        self.assertEqual(invalid.confidence, 0.0)
        self.assertEqual(classify_open(cases[1]).kind, "uncertain")
        self.assertEqual(classify_open(cases[2]).kind, "uncertain")

    def test_closed_random_forest_probabilities_control_uncertainty(self):
        points = [(0, 0), (1, 0), (0.5, 1), (0, 0)]
        features = analyze(preprocess(points, n=64))
        classifier = StrokeClassifier(use_ml_model=False)

        with patch.object(
            classifier, "_classify_closed_ml",
            return_value=ShapeClass("triangle", 0.90, {
                "triangle": 0.90, "square": 0.04, "circle": 0.02,
                "ellipse": 0.01, "rectangle": 0.01,
                "pentagon": 0.01, "hexagon": 0.01,
            }),
        ):
            confident = classifier.classify(points, features, closed=True)
        self.assertEqual(confident.kind, "triangle")
        self.assertEqual(confident.confidence, 0.90)

        with patch.object(
            classifier, "_classify_closed_ml",
            return_value=ShapeClass("triangle", 0.40, {
                "triangle": 0.40, "square": 0.35, "circle": 0.10,
                "ellipse": 0.05, "rectangle": 0.04,
                "pentagon": 0.03, "hexagon": 0.03,
            }),
        ):
            ambiguous = classifier.classify(points, features, closed=True)
        self.assertEqual(ambiguous.kind, "uncertain")
        self.assertEqual(ambiguous.confidence, 0.40)

    def test_closed_deterministic_fallback_marks_weak_evidence_uncertain(self):
        classifier = StrokeClassifier(use_ml_model=False)
        triangle = [(0, 0), (10, 0), (5, 8), (0, 0)]
        confident_features = analyze(preprocess(triangle, n=64))
        confident = classifier.classify(triangle, confident_features, closed=True)
        self.assertEqual(confident.kind, "triangle")
        self.assertGreaterEqual(confident.confidence, 0.55)

        weak_features = analyze([])
        result = classifier.classify([], weak_features, closed=True)
        self.assertEqual(result.kind, "uncertain")
        self.assertEqual(result.confidence, 0.35)


if __name__ == "__main__":
    unittest.main()

"""Focused smoke checks for the ML-backed closed-shape pipeline.

Run with: python scripts/closed_ml_integration_smoke.py
"""

import json
import math
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_pipeline import StrokePipeline
from tools.ml_features import FEATURE_COUNT, FEATURE_NAMES


SUPPORTED = {
    "circle", "ellipse", "triangle", "square", "rectangle", "pentagon", "hexagon"
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def oval(cx, cy, rx, ry, count=64):
    return [
        (cx + rx * math.cos(2 * math.pi * i / count),
         cy + ry * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ] + [(cx + rx, cy)]


def regular_polygon(cx, cy, radius_x, radius_y, sides, per_side=16):
    vertices = [
        (cx + radius_x * math.cos(-math.pi / 2 + 2 * math.pi * i / sides),
         cy + radius_y * math.sin(-math.pi / 2 + 2 * math.pi * i / sides))
        for i in range(sides)
    ]
    points = []
    for i, start in enumerate(vertices):
        end = vertices[(i + 1) % sides]
        for step in range(per_side):
            t = step / per_side
            points.append((start[0] + (end[0] - start[0]) * t,
                           start[1] + (end[1] - start[1]) * t))
    points.append(points[0])
    return points


def fixtures():
    cx, cy = 320, 240
    return {
        "circle": oval(cx, cy, 90, 90),
        "ellipse": oval(cx, cy, 130, 70),
        "triangle": regular_polygon(cx, cy, 100, 100, 3),
        "square": regular_polygon(cx, cy, 90, 90, 4),
        "rectangle": regular_polygon(cx, cy, 130, 65, 4),
        "pentagon": regular_polygon(cx, cy, 100, 100, 5),
        "hexagon": regular_polygon(cx, cy, 100, 100, 6),
    }


def main():
    pipeline = StrokePipeline()
    classifier = pipeline.classifier

    metadata_path = PROJECT_ROOT / "models" / "shape_random_forest_final_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    check(metadata["feature_dimension"] == FEATURE_COUNT == 143,
          "saved metadata and feature builder agree on 143 features")
    check(metadata["feature_names"] == list(FEATURE_NAMES),
          "saved metadata feature ordering matches the feature builder")

    for expected_label, points in fixtures().items():
        closed, _, normalized_features, prediction, _ = pipeline.classify_stroke(points)
        check(closed, f"{expected_label}: pipeline recognizes closed stroke")
        check(prediction.kind in SUPPORTED,
              f"{expected_label}: returns a supported model class ({prediction.kind})")
        check(bool(prediction.probabilities),
              f"{expected_label}: returns model class probabilities")
        check(set(prediction.probabilities) == set(metadata["class_labels"]),
              f"{expected_label}: probability labels match model metadata")
        check(all(math.isfinite(value) and 0.0 <= value <= 1.0
                  for value in prediction.probabilities.values()),
              f"{expected_label}: probabilities are finite and within [0, 1]")
        check(abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9,
              f"{expected_label}: probabilities sum to one")
        check(abs(prediction.confidence - max(prediction.probabilities.values())) < 1e-12,
              f"{expected_label}: confidence matches the predicted class probability")
        check(normalized_features.point_count == 64,
              f"{expected_label}: existing preprocessing still supplies 64 points")

    square_points = fixtures()["square"]
    heuristic = StrokePipeline(classifier=StrokeClassifier(use_ml_model=False))
    _, _, _, expected_fallback, _ = heuristic.classify_stroke(square_points)
    broken_model = StrokeClassifier(
        model_path=PROJECT_ROOT / "models" / "missing_model_for_smoke.joblib",
        metadata_path=PROJECT_ROOT / "models" / "missing_metadata_for_smoke.json",
    )
    fallback_pipeline = StrokePipeline(classifier=broken_model)
    _, _, _, actual_fallback, _ = fallback_pipeline.classify_stroke(square_points)
    check(actual_fallback.kind == expected_fallback.kind,
          "missing model falls back to the existing deterministic closed classifier")
    check(actual_fallback.probabilities == {},
          "deterministic fallback does not report fabricated model probabilities")

    _, _, valid_features, _, _ = heuristic.classify_stroke(square_points)
    invalid_features = deepcopy(valid_features)
    invalid_features.points = [(float("nan"), 0.0)] * 64
    invalid_classifier = StrokeClassifier()
    invalid_result = invalid_classifier.classify(square_points, invalid_features, True)
    check(invalid_result.kind == expected_fallback.kind,
          "non-finite model features fall back to the deterministic closed classifier")
    check(invalid_result.probabilities == {},
          "invalid features return no fabricated model probabilities")

    check(classifier._ml_model is not None and classifier._ml_load_checked,
          "pipeline successfully loads and validates the Random Forest")
    check(classifier._ml_model.n_features_in_ == 143,
          "loaded model expects the validated 143-feature ordering")
    print("All focused closed-shape integration checks passed.")


if __name__ == "__main__":
    main()

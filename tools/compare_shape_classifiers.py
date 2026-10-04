"""Compare the production closed-shape classifier with the saved ML model."""

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.stroke_analysis import analyze
from shapes.stroke_classifier import StrokeClassifier
from tools.ml_features import FEATURE_COUNT, extract_features


MODEL_PATH = PROJECT_ROOT / "models" / "shape_random_forest.joblib"
TEST_SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits" / "test"
ML_TEST_DIR = PROJECT_ROOT / "dataset" / "ml" / "test"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "shape_classifier_comparison.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metrics_for(labels, predictions, class_labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_labels, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(
            labels, predictions, labels=class_labels, average="macro", zero_division=0
        )),
        "per_class_metrics": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(score),
                "support": int(count),
            }
            for label, p, r, score, count in zip(
                class_labels, precision, recall, f1, support
            )
        },
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main():
    model = joblib.load(MODEL_PATH)
    class_labels = [str(label) for label in model.classes_]

    test_paths = sorted(TEST_SPLIT_DIR.rglob("*.json"))
    ml_features = load_json(ML_TEST_DIR / "features.json")
    ml_labels = load_json(ML_TEST_DIR / "labels.json")
    sample_ids = load_json(ML_TEST_DIR / "sample_ids.json")

    if not test_paths:
        raise ValueError(f"No test samples found in {TEST_SPLIT_DIR}")
    if not (len(test_paths) == len(ml_features) == len(ml_labels) == len(sample_ids)):
        raise ValueError("Split test samples and ML test arrays have different counts")

    split_labels = []
    split_feature_rows = []
    closed_strokes = []
    for path in test_paths:
        sample = load_json(path)
        points = sample.get("points")
        label = sample.get("label") or path.parent.name
        if not isinstance(points, list) or not isinstance(label, str):
            raise ValueError(f"Invalid test sample: {path}")

        split_labels.append(label)
        split_feature_rows.append(extract_features(points))
        sealed_points = list(points)
        if sealed_points and sealed_points[0] != sealed_points[-1]:
            sealed_points.append(sealed_points[0])
        closed_strokes.append(sealed_points)

    # Ensure the two predictors receive rows from exactly the same samples
    # and in the same order, rather than relying on matching counts alone.
    if split_labels != ml_labels:
        raise ValueError("Test split labels do not align with dataset/ml/test labels")
    if not np.allclose(
        np.asarray(split_feature_rows, dtype=np.float64),
        np.asarray(ml_features, dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Test split feature rows do not align with dataset/ml/test")
    if any(len(row) != FEATURE_COUNT for row in ml_features):
        raise ValueError(f"ML test features must have dimension {FEATURE_COUNT}")
    if not all(math.isfinite(float(value)) for row in ml_features for value in row):
        raise ValueError("ML test features contain non-finite values")

    x_test = np.asarray(ml_features, dtype=np.float64)
    rf_predictions = model.predict(x_test)

    classifier = StrokeClassifier()
    deterministic_predictions = []
    for points in closed_strokes:
        features = analyze(points)
        result = classifier.classify(points, features, True)
        deterministic_predictions.append(result.kind)

    results = {
        "test_split": "dataset/splits/test",
        "ml_features": "dataset/ml/test",
        "sample_count": len(split_labels),
        "feature_dimension": FEATURE_COUNT,
        "sample_ids": sample_ids,
        "class_labels": class_labels,
        "sample_alignment_verified": True,
        "random_forest": metrics_for(split_labels, rf_predictions, class_labels),
        "deterministic_closed_shape": metrics_for(
            split_labels, deterministic_predictions, class_labels
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    for name in ("random_forest", "deterministic_closed_shape"):
        report = results[name]
        print(f"\n{name}")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro_f1']:.4f}")
        for label, score in report["per_class_metrics"].items():
            print(
                f"  {label}: precision={score['precision']:.4f}, "
                f"recall={score['recall']:.4f}, f1={score['f1']:.4f}, "
                f"support={score['support']}"
            )
        print("Confusion matrix (actual rows, predicted columns):")
        print("Labels:", ", ".join(class_labels))
        for row in report["confusion_matrix"]:
            print(" ".join(str(value) for value in row))

    print(f"\nSame test samples verified: {len(split_labels)}")
    print(f"Saved comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

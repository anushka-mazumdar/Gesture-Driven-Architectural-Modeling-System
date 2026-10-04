"""Evaluate the saved Random Forest once on the held-out test split."""

import json
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
MODEL_PATH = PROJECT_ROOT / "models" / "shape_random_forest.joblib"
TEST_DIR = PROJECT_ROOT / "dataset" / "ml" / "test"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "shape_random_forest_test_results.json"


def main():
    # Load only the saved model and held-out test features/labels.
    model = joblib.load(MODEL_PATH)
    with (TEST_DIR / "features.json").open("r", encoding="utf-8") as file:
        test_features = json.load(file)
    with (TEST_DIR / "labels.json").open("r", encoding="utf-8") as file:
        test_labels = json.load(file)

    if not test_features or len(test_features) != len(test_labels):
        raise ValueError("Test features and labels must be non-empty and aligned")

    x_test = np.asarray(test_features, dtype=np.float64)
    if x_test.ndim != 2 or x_test.shape[1] != 143:
        raise ValueError(f"Expected test features with shape (n, 143), got {x_test.shape}")
    if not np.isfinite(x_test).all():
        raise ValueError("Test features contain non-finite values")

    class_labels = [str(label) for label in model.classes_]
    if not set(test_labels).issubset(set(class_labels)):
        raise ValueError("Test split contains labels unknown to the saved model")

    # One prediction pass; all metrics below use this same prediction array.
    predictions = model.predict(x_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=class_labels,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, predictions, labels=class_labels)

    results = {
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "split": "dataset/ml/test",
        "sample_count": len(test_labels),
        "feature_dimension": int(x_test.shape[1]),
        "class_labels": class_labels,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(
            test_labels,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
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
        "confusion_matrix": {
            "labels": class_labels,
            "values": matrix.astype(int).tolist(),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test macro F1: {results['macro_f1']:.4f}")
    print("Per-class metrics:")
    for label, metrics in results["per_class_metrics"].items():
        print(
            f"  {label}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )
    print("Confusion matrix (actual rows, predicted columns):")
    print("Labels:", ", ".join(class_labels))
    for row in results["confusion_matrix"]["values"]:
        print(" ".join(str(value) for value in row))
    print(f"Saved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

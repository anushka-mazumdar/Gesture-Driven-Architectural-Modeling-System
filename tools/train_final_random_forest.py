"""Fit the selected Random Forest on train, then evaluate it once on test.

The held-out test split is only loaded after training and model saving. Test
metrics are reported and saved; they are never used to change this candidate.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "dataset" / "ml"
MODEL_PATH = PROJECT_ROOT / "models" / "shape_random_forest_final.joblib"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "shape_random_forest_final_test_results.json"

CLASS_LABELS = (
    "circle",
    "ellipse",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
)
FEATURE_DIMENSION = 143
RANDOM_SEED = 42
CONFIGURATION = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "class_weight": "balanced_subsample",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_xy(split_name):
    """Load and validate one feature/label pair."""
    split_dir = ML_DIR / split_name
    features = read_json(split_dir / "features.json")
    labels = read_json(split_dir / "labels.json")
    if not features or len(features) != len(labels):
        raise ValueError(f"{split_name} feature and label arrays must be non-empty and aligned")

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIMENSION or not np.isfinite(x).all():
        raise ValueError(f"{split_name} features must be finite rows of length {FEATURE_DIMENSION}")
    if any(label not in CLASS_LABELS for label in labels):
        raise ValueError(f"{split_name} contains a label outside the 7 configured classes")
    return x, np.asarray(labels, dtype=str)


def main():
    train_x, train_y = load_xy("train")
    if set(train_y.tolist()) != set(CLASS_LABELS):
        raise ValueError("Training data must contain all 7 configured classes")

    model = RandomForestClassifier(
        **CONFIGURATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    # Save the final candidate before reading the held-out test data.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    test_x, test_y = load_xy("test")
    sample_ids = read_json(ML_DIR / "test" / "sample_ids.json")
    if len(sample_ids) != len(test_y):
        raise ValueError("Test sample IDs do not align with test features and labels")

    # Exactly one inference pass on the test feature matrix.
    predictions = model.predict(test_x)
    precision, recall, f1, support = precision_recall_fscore_support(
        test_y,
        predictions,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(test_y, predictions, labels=CLASS_LABELS)

    results = {
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "training_split": "dataset/ml/train",
        "evaluation_split": "dataset/ml/test",
        "test_used_for_tuning": False,
        "training_sample_count": int(len(train_y)),
        "test_sample_count": int(len(test_y)),
        "feature_dimension": FEATURE_DIMENSION,
        "random_seed": RANDOM_SEED,
        "configuration": CONFIGURATION,
        "class_labels": list(CLASS_LABELS),
        "test_sample_ids": sample_ids,
        "test_accuracy": float(accuracy_score(test_y, predictions)),
        "test_macro_f1": float(f1_score(
            test_y,
            predictions,
            labels=CLASS_LABELS,
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
                CLASS_LABELS, precision, recall, f1, support
            )
        },
        "confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "values": matrix.astype(int).tolist(),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test macro F1: {results['test_macro_f1']:.4f}")
    print("Per-class metrics:")
    for label, metrics in results["per_class_metrics"].items():
        print(
            f"  {label}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"support={metrics['support']}"
        )
    print("Confusion matrix (actual rows, predicted columns):")
    print("Labels:", ", ".join(CLASS_LABELS))
    for row in results["confusion_matrix"]["values"]:
        print(" ".join(str(value) for value in row))
    print(f"Final candidate saved to: {MODEL_PATH}")
    print(f"Test report saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

"""Train and validate the experimental Random Forest shape classifier.

This script reads only dataset/ml/train and dataset/ml/validation. It does not
load or evaluate the held-out test split and does not integrate with the app.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "dataset" / "ml"
MODEL_PATH = PROJECT_ROOT / "models" / "shape_random_forest.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "shape_random_forest_metadata.json"

CLASS_LABELS = (
    "circle",
    "ellipse",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
)
RANDOM_SEED = 42
TREE_COUNT = 300


def load_split(split_name, expected_feature_count):
    """Load one prepared split and verify its feature/label alignment."""
    split_dir = ML_DIR / split_name
    with (split_dir / "features.json").open("r", encoding="utf-8") as file:
        features = json.load(file)
    with (split_dir / "labels.json").open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(features, list) or not isinstance(labels, list):
        raise ValueError(f"{split_name} features and labels must be JSON arrays")
    if not features or len(features) != len(labels):
        raise ValueError(
            f"{split_name} must have matching, non-empty feature and label arrays"
        )
    if any(len(row) != expected_feature_count for row in features):
        raise ValueError(
            f"{split_name} rows must each contain {expected_feature_count} features"
        )
    if any(not isinstance(label, str) or label not in CLASS_LABELS for label in labels):
        raise ValueError(f"{split_name} contains a label outside the 7 configured classes")

    values = np.asarray(features, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} contains non-finite feature values")
    return values, np.asarray(labels, dtype=str)


def print_metrics(split_name, actual, predicted):
    """Print aggregate accuracy/macro F1 and metrics for every class."""
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual,
        predicted,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=CLASS_LABELS,
        zero_division=0,
    )

    print(f"\n{split_name} metrics")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Class                 Precision  Recall  F1      Support")
    for label, p, r, score, count in zip(CLASS_LABELS, precision, recall, f1, support):
        print(f"{label:<21} {p:>9.4f}  {r:>6.4f}  {score:>6.4f}  {count:>7}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": {
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
    }


def main():
    feature_names_path = ML_DIR / "feature_names.json"
    with feature_names_path.open("r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if len(feature_names) != 143:
        raise ValueError(f"Expected the 143-feature representation, got {len(feature_names)}")

    train_x, train_y = load_split("train", len(feature_names))
    validation_x, validation_y = load_split("validation", len(feature_names))
    if set(train_y.tolist()) != set(CLASS_LABELS):
        raise ValueError("Training data must contain all 7 current shape classes")

    model = RandomForestClassifier(
        n_estimators=TREE_COUNT,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    train_metrics = print_metrics("Training", train_y, model.predict(train_x))
    validation_metrics = print_metrics(
        "Validation", validation_y, model.predict(validation_x)
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "random_seed": RANDOM_SEED,
        "n_estimators": TREE_COUNT,
        "feature_dimension": len(feature_names),
        "feature_names": feature_names,
        "class_labels": list(CLASS_LABELS),
        "training_samples": int(len(train_y)),
        "validation_samples": int(len(validation_y)),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nClass labels:", ", ".join(CLASS_LABELS))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()

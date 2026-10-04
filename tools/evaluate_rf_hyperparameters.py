"""Compare a small, controlled Random Forest configuration set.

Every candidate is fit on dataset/ml/train and scored on dataset/ml/validation.
The test split is not loaded. The selected parameters are recorded for later
use; this experiment never saves or replaces the production/previous model.

The candidates vary one setting at a time around the ML-3 baseline:
300 trees, unlimited depth, split=2, leaf=1, max_features='sqrt'. The set
checks tree count (150/500), depth (10/20), split size (4), leaf size (2),
and feature sampling (0.5/0.8) while holding all other settings fixed.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "dataset" / "ml"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "random_forest_hyperparameter_evaluation.json"

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
BASE_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}
CANDIDATES = (
    ("baseline", {}),
    ("trees_150", {"n_estimators": 150}),
    ("trees_500", {"n_estimators": 500}),
    ("depth_10", {"max_depth": 10}),
    ("depth_20", {"max_depth": 20}),
    ("split_4", {"min_samples_split": 4}),
    ("leaf_2", {"min_samples_leaf": 2}),
    ("features_0_5", {"max_features": 0.5}),
    ("features_0_8", {"max_features": 0.8}),
)


def load_split(name):
    split_dir = ML_DIR / name
    with (split_dir / "features.json").open("r", encoding="utf-8") as file:
        features = json.load(file)
    with (split_dir / "labels.json").open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not features or len(features) != len(labels):
        raise ValueError(f"{name} features and labels must be non-empty and aligned")
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 143 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite 143-dimensional feature rows")
    if any(label not in CLASS_LABELS for label in labels):
        raise ValueError(f"{name} contains a label outside the configured classes")
    return values, np.asarray(labels, dtype=str)


def main():
    train_x, train_y = load_split("train")
    validation_x, validation_y = load_split("validation")

    results = []
    for name, overrides in CANDIDATES:
        config = {**BASE_CONFIG, **overrides}
        model = RandomForestClassifier(
            **config,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        predictions = model.predict(validation_x)
        per_class_f1 = f1_score(
            validation_y,
            predictions,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )
        results.append({
            "name": name,
            "configuration": config,
            "validation_accuracy": float(accuracy_score(validation_y, predictions)),
            "validation_macro_f1": float(f1_score(
                validation_y,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )),
            "validation_per_class_f1": {
                label: float(score)
                for label, score in zip(CLASS_LABELS, per_class_f1)
            },
        })
        print(
            f"{name}: validation_accuracy={results[-1]['validation_accuracy']:.4f}, "
            f"validation_macro_f1={results[-1]['validation_macro_f1']:.4f}, "
            f"configuration={config}"
        )

    # Python's stable max keeps the first candidate if scores tie exactly.
    selected = max(results, key=lambda item: item["validation_macro_f1"])
    report = {
        "experiment": "controlled Random Forest hyperparameter evaluation",
        "training_split": "dataset/ml/train",
        "selection_split": "dataset/ml/validation",
        "test_split_used": False,
        "model_saved_or_modified": False,
        "feature_dimension": 143,
        "class_labels": list(CLASS_LABELS),
        "random_seed": RANDOM_SEED,
        "candidate_count": len(results),
        "selection_metric": "validation_macro_f1",
        "tie_break": "first listed candidate",
        "candidates": results,
        "selected_configuration": selected["configuration"],
        "selected_candidate": selected["name"],
        "selected_validation_macro_f1": selected["validation_macro_f1"],
        "selected_validation_accuracy": selected["validation_accuracy"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print(
        f"Selected by validation macro F1: {selected['name']} "
        f"({selected['validation_macro_f1']:.4f})"
    )
    print(f"Selected configuration: {selected['configuration']}")
    print(f"Saved evaluation: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Train an experimental SVM and compare validation scores with tuned RF.

Only dataset/ml/train and dataset/ml/validation are read. The SVM is saved
separately; this script neither reads the test split nor integrates models
into the application.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "dataset" / "ml"
TUNING_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "random_forest_hyperparameter_evaluation.json"
MODEL_PATH = PROJECT_ROOT / "models" / "shape_svm.joblib"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "svm_vs_tuned_random_forest.json"

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
SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
}


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split(name):
    split_dir = ML_DIR / name
    features = read_json(split_dir / "features.json")
    labels = read_json(split_dir / "labels.json")
    if not features or len(features) != len(labels):
        raise ValueError(f"{name} feature and label arrays must be non-empty and aligned")

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIMENSION or not np.isfinite(x).all():
        raise ValueError(f"{name} features must be finite rows of length {FEATURE_DIMENSION}")
    if any(label not in CLASS_LABELS for label in labels):
        raise ValueError(f"{name} contains a label outside the configured 7 classes")
    return x, np.asarray(labels, dtype=str)


def score(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0,
        )),
        "per_class_metrics": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(count),
            }
            for label, p, r, f, count in zip(
                CLASS_LABELS, precision, recall, f1, support
            )
        },
    }


def main():
    feature_names = read_json(ML_DIR / "feature_names.json")
    if len(feature_names) != FEATURE_DIMENSION:
        raise ValueError(f"Expected {FEATURE_DIMENSION} feature names")

    train_x, train_y = load_split("train")
    validation_x, validation_y = load_split("validation")
    if set(train_y.tolist()) != set(CLASS_LABELS):
        raise ValueError("Training data must contain all 7 configured classes")

    model = make_pipeline(
        StandardScaler(),
        SVC(**SVM_CONFIG),
    )
    model.fit(train_x, train_y)
    training_metrics = score(train_y, model.predict(train_x))
    validation_metrics = score(validation_y, model.predict(validation_x))

    # Use the previously measured tuned-RF validation result. Do not load or
    # fit the saved Random Forest, and never load the held-out test split.
    tuning = read_json(TUNING_RESULTS_PATH)
    if tuning.get("test_split_used") or tuning.get("selection_metric") != "validation_macro_f1":
        raise ValueError("Tuning report is not a validation-only Random Forest result")
    rf_result = next(
        (candidate for candidate in tuning["candidates"]
         if candidate["name"] == tuning["selected_candidate"]),
        None,
    )
    if rf_result is None:
        raise ValueError("Selected tuned Random Forest result is missing")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    results = {
        "training_split": "dataset/ml/train",
        "validation_split": "dataset/ml/validation",
        "test_split_used": False,
        "feature_dimension": FEATURE_DIMENSION,
        "class_labels": list(CLASS_LABELS),
        "svm": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "configuration": SVM_CONFIG,
            "training_sample_count": int(len(train_y)),
            "validation_sample_count": int(len(validation_y)),
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
        },
        "tuned_random_forest": {
            "configuration": rf_result["configuration"],
            "validation_accuracy": rf_result["validation_accuracy"],
            "validation_macro_f1": rf_result["validation_macro_f1"],
            "source": str(TUNING_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        },
        "comparison": {
            "validation_accuracy_difference_svm_minus_rf": (
                validation_metrics["accuracy"] - rf_result["validation_accuracy"]
            ),
            "validation_macro_f1_difference_svm_minus_rf": (
                validation_metrics["macro_f1"] - rf_result["validation_macro_f1"]
            ),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print("Class labels:", ", ".join(CLASS_LABELS))
    print(f"Feature dimension: {FEATURE_DIMENSION}")
    print(
        f"SVM training: accuracy={training_metrics['accuracy']:.4f}, "
        f"macro_f1={training_metrics['macro_f1']:.4f}"
    )
    print(
        f"SVM validation: accuracy={validation_metrics['accuracy']:.4f}, "
        f"macro_f1={validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Tuned RF validation: accuracy={rf_result['validation_accuracy']:.4f}, "
        f"macro_f1={rf_result['validation_macro_f1']:.4f}"
    )
    print(
        "Validation difference (SVM - RF): "
        f"accuracy={results['comparison']['validation_accuracy_difference_svm_minus_rf']:+.4f}, "
        f"macro_f1={results['comparison']['validation_macro_f1_difference_svm_minus_rf']:+.4f}"
    )
    print(f"SVM saved to: {MODEL_PATH}")
    print(f"Comparison saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

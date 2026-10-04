"""List the saved Random Forest's errors on its existing test split."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "shape_random_forest.joblib"
TEST_DIR = PROJECT_ROOT / "dataset" / "ml" / "test"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
BASELINE_PATH = EVALUATION_DIR / "shape_random_forest_test_results.json"
COMPARISON_PATH = EVALUATION_DIR / "shape_classifier_comparison.json"
OUTPUT_PATH = EVALUATION_DIR / "random_forest_test_mistakes.json"


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # These saved artifacts identify the original 61-row test evaluation.
    baseline = read_json(BASELINE_PATH)
    comparison = read_json(COMPARISON_PATH)
    features = read_json(TEST_DIR / "features.json")
    labels = read_json(TEST_DIR / "labels.json")
    sample_ids = read_json(TEST_DIR / "sample_ids.json")

    expected_count = baseline["sample_count"]
    if not (len(features) == len(labels) == len(sample_ids) == expected_count):
        raise ValueError("Test features, labels, IDs, and saved evaluation counts differ")
    if comparison.get("sample_ids") != sample_ids:
        raise ValueError("Test sample IDs differ from the saved classifier comparison")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Test sample IDs are not unique")

    model = joblib.load(MODEL_PATH)
    class_labels = [str(label) for label in model.classes_]
    x_test = np.asarray(features, dtype=np.float64)
    if x_test.ndim != 2 or not np.isfinite(x_test).all():
        raise ValueError("Test feature matrix is malformed or non-finite")

    # One probability pass supplies both predicted labels and confidences.
    probability_matrix = model.predict_proba(x_test)
    predicted_indices = np.argmax(probability_matrix, axis=1)
    predictions = [class_labels[index] for index in predicted_indices]

    accuracy = float(accuracy_score(labels, predictions))
    matrix = confusion_matrix(labels, predictions, labels=class_labels).astype(int).tolist()
    if abs(accuracy - float(baseline["accuracy"])) > 1e-12:
        raise ValueError("Recomputed predictions do not match the saved test evaluation")
    if matrix != baseline["confusion_matrix"]["values"]:
        raise ValueError("Recomputed confusion matrix differs from the saved evaluation")

    mistakes = []
    pair_counts = Counter()
    pair_sample_ids = defaultdict(list)
    for row_index, (sample_id, actual, predicted) in enumerate(
        zip(sample_ids, labels, predictions)
    ):
        if actual == predicted:
            continue
        probabilities = {
            label: float(probability_matrix[row_index, class_index])
            for class_index, label in enumerate(class_labels)
        }
        confidence = probabilities[predicted]
        mistakes.append({
            "sample_id": sample_id,
            "test_row": row_index,
            "actual_label": actual,
            "predicted_label": predicted,
            "predicted_confidence": confidence,
            "actual_label_probability": probabilities[actual],
            "class_probabilities": probabilities,
        })
        pair = (actual, predicted)
        pair_counts[pair] += 1
        pair_sample_ids[pair].append(sample_id)

    confusion_pairs = [
        {
            "actual_label": actual,
            "predicted_label": predicted,
            "count": count,
            "sample_ids": pair_sample_ids[(actual, predicted)],
        }
        for (actual, predicted), count in sorted(
            pair_counts.items(), key=lambda entry: (-entry[1], entry[0])
        )
    ]

    analysis = {
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "test_split": "dataset/ml/test",
        "baseline_evaluation": str(BASELINE_PATH.relative_to(PROJECT_ROOT)),
        "sample_count": expected_count,
        "accuracy_matches_saved_evaluation": True,
        "confusion_matrix_matches_saved_evaluation": True,
        "class_labels": class_labels,
        "misclassified_count": len(mistakes),
        "misclassified_samples": mistakes,
        "main_confusion_pairs": confusion_pairs,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print(f"Test samples: {expected_count}")
    print(f"Misclassified: {len(mistakes)}")
    print(f"Accuracy matches saved evaluation: {accuracy:.4f}")
    print("Misclassified samples:")
    for item in mistakes:
        print(
            f"  {item['sample_id']}: {item['actual_label']} -> "
            f"{item['predicted_label']} (confidence={item['predicted_confidence']:.4f}, "
            f"actual_probability={item['actual_label_probability']:.4f})"
        )
    print("Confusion pairs:")
    for pair in confusion_pairs:
        print(
            f"  {pair['actual_label']} -> {pair['predicted_label']}: "
            f"{pair['count']} ({', '.join(pair['sample_ids'])})"
        )
    print(f"Saved analysis: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

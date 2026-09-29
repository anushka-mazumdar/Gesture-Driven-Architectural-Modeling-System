"""Build feature/label/identity arrays from the fixed dataset splits.

The split files may have split-local filenames and omit source IDs. In that
case, IDs are recovered by matching each sample's label and points against
dataset/processed, whose filenames retain the original sample numbering.
This script never reshuffles, augments, or synthesizes split samples.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ml_features import FEATURE_COUNT, FEATURE_NAMES, extract_features


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "ml"


def point_signature(points):
    """Create an exact, stable signature for numeric point coordinates."""
    return tuple((float(point[0]), float(point[1])) for point in points)


def source_sample_id_index():
    """Index processed strokes so split-local names can map to source IDs."""
    index = defaultdict(list)
    for sample_path in sorted(PROCESSED_DIR.rglob("*.json")):
        with sample_path.open("r", encoding="utf-8") as sample_file:
            sample = json.load(sample_file)
        if not isinstance(sample, dict) or "points" not in sample:
            continue

        label = sample.get("label") or sample_path.parent.name
        source_id = sample.get("sample_id")
        if source_id is None:
            source_id = sample.get("id")
        if source_id is None:
            source_id = sample_path.stem
        # Include the class so IDs such as circle/sample_0001 and
        # square/sample_0001 remain distinct.
        stable_id = f"{label}/{source_id}"
        index[(label, point_signature(sample["points"]))].append(stable_id)
    return index


def load_split(split_dir, source_index, used_source_ids):
    """Load one existing split and align features, labels, and source IDs."""
    features = []
    labels = []
    sample_ids = []

    for sample_path in sorted(split_dir.rglob("*.json")):
        with sample_path.open("r", encoding="utf-8") as sample_file:
            sample = json.load(sample_file)
        if not isinstance(sample, dict):
            raise ValueError(f"Expected a JSON object in {sample_path}")

        points = sample.get("points")
        label = sample.get("label") or sample_path.parent.name
        if not isinstance(label, str) or not label:
            raise ValueError(f"Missing shape label in {sample_path}")
        if not isinstance(points, list):
            raise ValueError(f"Missing point list in {sample_path}")

        candidates = source_index.get((label, point_signature(points)), [])
        if not candidates:
            raise ValueError(
                f"Cannot trace {sample_path} back to a processed source sample "
                f"(label={label!r}); refusing to invent an ID"
            )

        # Use each source identity once across all splits. If a source is
        # repeated in the split input, retaining its ID lets the disjointness
        # check below catch the overlap.
        source_id = next(
            (candidate for candidate in candidates if candidate not in used_source_ids),
            candidates[0],
        )
        used_source_ids.add(source_id)

        vector = extract_features(points)
        features.append(vector)
        labels.append(label)
        sample_ids.append(source_id)

    return features, labels, sample_ids


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False, allow_nan=False)
        output_file.write("\n")


def verify_datasets(datasets):
    """Print alignment and cross-split identity checks before any output write."""
    aligned = True
    for split_name in SPLIT_NAMES:
        features, labels, sample_ids = datasets[split_name]
        counts_match = len(features) == len(labels) == len(sample_ids)
        finite = all(
            len(vector) == FEATURE_COUNT
            and all(math.isfinite(value) for value in vector)
            for vector in features
        )
        passed = counts_match and finite
        aligned = aligned and passed
        print(
            f"CHECK {split_name}: features={len(features)}, labels={len(labels)}, "
            f"sample_ids={len(sample_ids)}, dimension={FEATURE_COUNT}, "
            f"aligned_and_finite={'PASS' if passed else 'FAIL'}"
        )

    id_sets = {
        split_name: set(datasets[split_name][2])
        for split_name in SPLIT_NAMES
    }
    disjoint = True
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        overlap = id_sets[left] & id_sets[right]
        pair_passed = not overlap
        disjoint = disjoint and pair_passed
        print(
            f"CHECK {left}/{right} sample IDs disjoint: "
            f"overlap={len(overlap)}, {'PASS' if pair_passed else 'FAIL'}"
        )

    if not aligned or not disjoint:
        raise SystemExit("Dataset verification failed; dataset/ml was not written.")


def main():
    missing = [name for name in SPLIT_NAMES if not (SPLIT_DIR / name).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing split folder(s): {', '.join(missing)} under {SPLIT_DIR}"
        )
    if not PROCESSED_DIR.is_dir():
        raise SystemExit(f"Processed source folder does not exist: {PROCESSED_DIR}")

    source_index = source_sample_id_index()
    datasets = {}
    used_source_ids = set()
    for split_name in SPLIT_NAMES:
        datasets[split_name] = load_split(
            SPLIT_DIR / split_name, source_index, used_source_ids
        )

    # Verify all splits together before writing any feature or label arrays.
    verify_datasets(datasets)

    write_json(OUTPUT_DIR / "feature_names.json", list(FEATURE_NAMES))
    for split_name in SPLIT_NAMES:
        features, labels, sample_ids = datasets[split_name]
        split_output = OUTPUT_DIR / split_name
        write_json(split_output / "features.json", features)
        write_json(split_output / "labels.json", labels)
        write_json(split_output / "sample_ids.json", sample_ids)

        distribution = Counter(labels)
        class_counts = ", ".join(
            f"{label}: {distribution[label]}" for label in sorted(distribution)
        ) or "none"
        print(
            f"{split_name}: samples={len(features)}, "
            f"feature_dimension={FEATURE_COUNT}, classes={{ {class_counts} }}"
        )

    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

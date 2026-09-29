import json
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

TARGET_POINTS = 64

CLASSES = [
    "circle",
    "ellipse",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
    "regular_polygon",
    "irregular_polygon",
]


# ============================================================
# Point processing
# ============================================================

def remove_duplicate_points(points):
    """Remove consecutive duplicate coordinates."""

    if len(points) == 0:
        return []

    cleaned = [points[0]]

    for point in points[1:]:
        if point != cleaned[-1]:
            cleaned.append(point)

    return cleaned


def resample_stroke(points, target_count=TARGET_POINTS):
    """
    Resample a stroke so every sample contains the same
    number of points, distributed approximately by distance.
    """

    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return None

    # Distance between consecutive points
    deltas = np.diff(points, axis=0)

    distances = np.sqrt(
        np.sum(deltas ** 2, axis=1)
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(distances))
    )

    total_length = cumulative[-1]

    if total_length <= 1e-6:
        return None

    target_distances = np.linspace(
        0,
        total_length,
        target_count
    )

    x = np.interp(
        target_distances,
        cumulative,
        points[:, 0]
    )

    y = np.interp(
        target_distances,
        cumulative,
        points[:, 1]
    )

    return np.column_stack((x, y))


def normalize_stroke(points):
    """
    Center the stroke and normalize its overall size.

    Aspect ratio is preserved because X and Y are divided
    by the same scale factor.
    """

    points = np.asarray(points, dtype=np.float32)

    # Center around centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # Scale using maximum absolute coordinate
    scale = np.max(np.abs(points))

    if scale <= 1e-6:
        return None

    points = points / scale

    return points


# ============================================================
# Process one sample
# ============================================================

def process_sample(file_path, label):
    """Load and preprocess one raw JSON stroke."""

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_points = data.get("points", [])

        if len(raw_points) < 10:
            return None

        points = remove_duplicate_points(raw_points)

        if len(points) < 2:
            return None

        points = resample_stroke(points)

        if points is None:
            return None

        points = normalize_stroke(points)

        if points is None:
            return None

        return {
            "label": label,
            "points": points.tolist(),
        }

    except Exception as exc:
        print(f"Failed: {file_path.name} -> {exc}")
        return None


# ============================================================
# Main
# ============================================================

def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total = 0
    failed = 0

    for label in CLASSES:

        raw_class_dir = RAW_DIR / label
        processed_class_dir = PROCESSED_DIR / label

        processed_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if not raw_class_dir.exists():
            print(f"[SKIP] {label}: folder not found")
            continue

        files = sorted(
            raw_class_dir.glob("sample_*.json")
        )

        class_count = 0

        for file_path in files:

            processed = process_sample(
                file_path,
                label
            )

            if processed is None:
                failed += 1
                continue

            output_path = (
                processed_class_dir /
                file_path.name
            )

            with output_path.open(
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    processed,
                    f
                )

            class_count += 1
            total += 1

        print(
            f"[OK] {label}: "
            f"{class_count} samples"
        )

    print()
    print("=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Processed: {total}")
    print(f"Failed:    {failed}")
    print(f"Output:    {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
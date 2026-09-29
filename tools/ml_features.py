"""Feature extraction for the experimental shape-classification model.

This module is separate from the production StrokeClassifier. Inputs to
``extract_features`` are normalized strokes with exactly 64 (x, y) points.
Use ``prepare_stroke`` to canonicalize an unprocessed stroke first.
"""

import math

from shapes.stroke_analysis import analyze
from shapes.stroke_normalization import preprocess


POINT_COUNT = 64
COORDINATE_FEATURE_NAMES = tuple(
    name
    for index in range(POINT_COUNT)
    for name in (f"point_{index}_x", f"point_{index}_y")
)
GEOMETRY_FEATURE_NAMES = (
    "aspect_ratio",
    "perimeter",
    "area",
    "roundness",
    "bbox_fill",
    "solidity",
    "convexity",
    "endpoint_gap",
    "straightness",
    "approx_vertices",
    "hull_vertices",
    "turn_angle_mean_deg",
    "turn_angle_std_deg",
    "turn_angle_max_deg",
    "turn_angle_total_deg",
)
FEATURE_NAMES = COORDINATE_FEATURE_NAMES + GEOMETRY_FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)


def prepare_stroke(points):
    """Resample and normalize a raw stroke using the project's shared logic."""
    return preprocess(points, n=POINT_COUNT)


def _turn_angle_statistics(points):
    """Return absolute turn-angle mean, standard deviation, max, and total."""
    angles = []
    for before, current, after in zip(points, points[1:], points[2:]):
        incoming = (current[0] - before[0], current[1] - before[1])
        outgoing = (after[0] - current[0], after[1] - current[1])
        in_length = math.hypot(*incoming)
        out_length = math.hypot(*outgoing)
        if in_length <= 1e-9 or out_length <= 1e-9:
            continue
        dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        cosine = max(-1.0, min(1.0, dot / (in_length * out_length)))
        angles.append(math.degrees(math.acos(cosine)))

    if not angles:
        return (0.0, 0.0, 0.0, 0.0)

    mean = sum(angles) / len(angles)
    variance = sum((angle - mean) ** 2 for angle in angles) / len(angles)
    return (mean, math.sqrt(variance), max(angles), sum(angles))


def extract_features(normalized_points):
    """Build one fixed-length numeric vector from a normalized 64-point stroke.

    Coordinates are flattened in x/y order, followed by scale-aware contour
    descriptors and turn statistics from the analyzer's simplified corners.
    Invalid or non-finite input is rejected so model training cannot silently
    consume malformed samples.
    """
    points = []
    for index, point in enumerate(normalized_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"point {index} must contain x and y coordinates")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            raise ValueError(f"point {index} coordinates must be numeric") from None
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {index} coordinates must be finite")
        points.append((x, y))

    if len(points) != POINT_COUNT:
        raise ValueError(f"expected {POINT_COUNT} points, got {len(points)}")

    features = analyze(points)
    turn_stats = _turn_angle_statistics(features.approx_points)
    geometry = (
        features.aspect_ratio,
        features.perimeter,
        features.area,
        features.roundness,
        features.bbox_fill,
        features.solidity,
        features.convexity,
        features.endpoint_gap,
        features.straightness,
        float(features.approx_vertices),
        float(len(features.convex_hull)),
        *turn_stats,
    )
    vector = [coordinate for point in points for coordinate in point]
    vector.extend(float(value) for value in geometry)

    if len(vector) != FEATURE_COUNT or not all(math.isfinite(value) for value in vector):
        raise ValueError("feature extraction produced an invalid feature vector")
    return vector


def extract_features_from_raw(points):
    """Canonicalize a raw stroke, then return its experimental feature vector."""
    return extract_features(prepare_stroke(points))

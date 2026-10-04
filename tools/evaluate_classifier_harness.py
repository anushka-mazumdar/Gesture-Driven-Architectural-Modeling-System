"""Repeatable local evaluation of the real closed/open StrokeClassifier API.

Run from the repository root with:
    python tools/evaluate_classifier_harness.py

The harness reports observed outputs; it does not tune or change classifier
behavior and does not calculate accuracy from synthetic fixtures.
"""

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.stroke_analysis import analyze
from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_normalization import preprocess


OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "classifier_harness_results.json"


def regular_polygon(sides, radius=100.0, center=(0.0, 0.0), rotation=-math.pi / 2):
    points = [
        (center[0] + radius * math.cos(rotation + 2 * math.pi * i / sides),
         center[1] + radius * math.sin(rotation + 2 * math.pi * i / sides))
        for i in range(sides)
    ]
    return points + [points[0]]


def ellipse(rx=120.0, ry=70.0, count=120):
    return [(rx * math.cos(2 * math.pi * i / (count - 1)),
             ry * math.sin(2 * math.pi * i / (count - 1)))
            for i in range(count)]


def noisy_closed(points, amount=0.035):
    """Apply deterministic, low-amplitude radial jitter to a closed contour."""
    cx = sum(p[0] for p in points[:-1]) / max(1, len(points) - 1)
    cy = sum(p[1] for p in points[:-1]) / max(1, len(points) - 1)
    out = []
    for index, (x, y) in enumerate(points[:-1]):
        phase = math.sin(index * 2.399963229728653) + 0.5 * math.sin(index * 0.71)
        scale = 1.0 + amount * phase
        out.append((cx + (x - cx) * scale, cy + (y - cy) * scale))
    out.append(out[0])
    return out


def fixtures():
    """Return stable clean, noisy, ambiguous, degenerate and invalid fixtures."""
    cases = []

    def add(case_id, family, scenario, points, closed, expected=None, invalid=False):
        cases.append({
            "case_id": case_id,
            "family": family,
            "scenario": scenario,
            "expected_label": expected,
            "closed": closed,
            "points": points,
            "invalid_input": invalid,
        })

    closed_clean = {
        "circle": [(100 * math.cos(2 * math.pi * i / 120),
                    100 * math.sin(2 * math.pi * i / 120)) for i in range(121)],
        "ellipse": ellipse(),
        "triangle": regular_polygon(3),
        "square": regular_polygon(4),
        "rectangle": [(-130, -70), (130, -70), (130, 70), (-130, 70), (-130, -70)],
        "pentagon": regular_polygon(5),
        "hexagon": regular_polygon(6),
        "other_regular_polygon": regular_polygon(8),
        "irregular_polygon": [
            (-75, -35), (15, -60), (78, -8), (32, 12), (62, 68),
            (-8, 42), (-53, 65), (-35, 4), (-75, -35),
        ],
    }
    for class_name, points in closed_clean.items():
        expected = f"{len(points) - 1}-gon" if class_name == "other_regular_polygon" else (
            "polygon" if class_name == "irregular_polygon" else class_name
        )
        add(f"closed_{class_name}_clean", "closed", "clean", points, True, expected)

    add("closed_circle_noisy", "closed", "noisy",
        noisy_closed(closed_clean["circle"]), True, "circle")
    add("closed_hexagon_noisy", "closed", "noisy",
        noisy_closed(closed_clean["hexagon"], amount=0.025), True, "hexagon")

    # A rounded square contour intentionally sits between polygonal and
    # circular evidence. No expected label is assigned to this ambiguous case.
    ambiguous_closed = []
    corners = [(-70, -70), (70, -70), (70, 70), (-70, 70), (-70, -70)]
    for start, end in zip(corners, corners[1:]):
        for step in range(12):
            t = step / 12
            ambiguous_closed.append((start[0] + (end[0] - start[0]) * t,
                                     start[1] + (end[1] - start[1]) * t))
    ambiguous_closed.append(ambiguous_closed[0])
    add("closed_rounded_square_ambiguous", "closed", "ambiguous",
        ambiguous_closed, True)
    add("closed_repeated_point_degenerate", "closed", "degenerate",
        [(4.0, 4.0)] * 8, True)
    add("closed_malformed_invalid", "closed", "invalid",
        [(0.0, 0.0), ("not-a-number", 1.0), (0.0, 0.0)], True, invalid=True)

    # Open fixtures use enough span that their endpoints are not accidentally
    # near one another under the application's closure threshold.
    open_clean = {
        "straight_line": [(float(i * 10), 2.0) for i in range(31)],
        "polyline": [(0.0, 0.0), (120.0, 0.0), (120.0, 90.0), (260.0, 90.0)],
        "arc": [(100 * math.cos(math.radians(angle)),
                 100 * math.sin(math.radians(angle)))
                for angle in range(0, 181, 3)],
        "curve": [(float(i * 5), 55 * math.sin(i / 11)) for i in range(81)],
        "freeform_path": [
            (float(i * 10), (55.0 if i % 2 else -55.0) +
             (28.0 if i % 5 == 0 else 0.0)) for i in range(25)
        ],
    }
    for label, points in open_clean.items():
        add(f"open_{label}_clean", "open", "clean", points, False, label)

    add("open_straight_line_noisy", "open", "noisy",
        [(float(i * 10), (1.5 if i % 2 else -1.5)) for i in range(31)],
        False, "straight_line")
    add("open_curve_noisy", "open", "noisy",
        [(float(i * 5), 53 * math.sin(i / 11) + 1.2 * math.sin(i * 1.7))
         for i in range(81)], False, "curve")
    add("open_shallow_bend_ambiguous", "open", "ambiguous",
        [(float(i * 10), 0.004 * (i * 10) ** 2) for i in range(31)], False)
    add("open_zigzag_ambiguous", "open", "ambiguous",
        [(float(i * 8), float((i % 2) * 10)) for i in range(31)], False)
    add("open_repeated_point_degenerate", "open", "degenerate",
        [(4.0, 4.0)] * 8, False)
    add("open_malformed_invalid", "open", "invalid",
        [(0.0, 0.0), (1.0, float("nan")), (100.0, 2.0)], False,
        invalid=True)
    add("open_empty_invalid", "open", "invalid", [], False, invalid=True)
    return cases


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run_case(classifier, case):
    points = case["points"]
    result = {
        "case_id": case["case_id"],
        "family": case["family"],
        "scenario": case["scenario"],
        "expected_label": case["expected_label"],
        "closed": case["closed"],
        "point_count": len(points),
        "actual_label": None,
        "confidence": None,
        "probabilities": None,
        "uncertain": None,
        "status": "ok",
        "expectation_met": None,
        "failure_type": None,
        "error": None,
        "input_processing_error": None,
    }

    features = None
    if not case["invalid_input"]:
        try:
            # Mirror StrokePipeline's shared resampling/normalization before
            # invoking the classifier's existing public method.
            normalized = preprocess(points, n=64)
            features = analyze(normalized)
        except Exception as error:
            result["input_processing_error"] = {
                "type": type(error).__name__, "message": str(error),
            }
    try:
        prediction = classifier.classify(points, features, case["closed"])
        result["actual_label"] = prediction.kind
        result["confidence"] = float(prediction.confidence)
        result["probabilities"] = prediction.probabilities or None
        result["uncertain"] = prediction.kind in {"uncertain", "unknown"}
    except Exception as error:
        result["status"] = "error"
        result["failure_type"] = "classifier_exception"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    if result["expected_label"] is not None and result["status"] == "ok":
        result["expectation_met"] = (
            result["actual_label"] == result["expected_label"]
        )
        if not result["expectation_met"]:
            result["failure_type"] = "prediction_mismatch"
    return result


def run_harness(output_path=OUTPUT_PATH):
    classifier = StrokeClassifier()
    captured_warnings = _WarningCapture()
    classifier_logger = logging.getLogger("shapes.stroke_classifier")
    classifier_logger.addHandler(captured_warnings)
    try:
        results = [run_case(classifier, case) for case in fixtures()]
    finally:
        classifier_logger.removeHandler(captured_warnings)

    loaded_model = classifier._ml_model is not None
    model_load_warning = next(
        (message for message in captured_warnings.messages
         if "could not load/validate model" in message),
        None,
    )
    failures = [result for result in results if result["status"] == "error"]
    mismatches = [result for result in results
                  if result["failure_type"] == "prediction_mismatch"]
    summary = {
        "case_count": len(results),
        "classified_count": sum(
            result["actual_label"] not in (None, "uncertain", "unknown")
            for result in results
        ),
        "uncertain_count": sum(bool(result["uncertain"]) for result in results),
        "failure_count": len(failures),
        "prediction_mismatch_count": len(mismatches),
        "observed_labels": dict(sorted(Counter(
            result["actual_label"] for result in results
            if result["actual_label"] is not None
        ).items())),
        "by_family": {},
        "random_forest_loaded": loaded_model,
        "random_forest_load_warning": model_load_warning,
    }
    for family in ("closed", "open"):
        family_results = [result for result in results if result["family"] == family]
        summary["by_family"][family] = {
            "case_count": len(family_results),
            "classified_count": sum(
                result["actual_label"] not in (None, "uncertain", "unknown")
                for result in family_results
            ),
            "uncertain_count": sum(bool(result["uncertain"])
                                    for result in family_results),
            "failure_count": sum(result["status"] == "error"
                                 for result in family_results),
            "prediction_mismatch_count": sum(
                result["failure_type"] == "prediction_mismatch"
                for result in family_results
            ),
            "observed_labels": dict(sorted(Counter(
                result["actual_label"] for result in family_results
                if result["actual_label"] is not None
            ).items())),
        }

    payload = {
        "harness": "StrokeClassifier local synthetic fixture evaluation",
        "harness_version": 1,
        "classifier_api": "StrokeClassifier.classify(points, normalized_features, closed)",
        "model_path": str(classifier.model_path.relative_to(PROJECT_ROOT)),
        "summary": summary,
        "results": results,
        "failures": [result for result in results
                     if result["failure_type"] is not None],
    }
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")

    print(f"Classifier harness: {summary['case_count']} cases")
    print(f"  Classified: {summary['classified_count']} | uncertain: "
          f"{summary['uncertain_count']} | expected-label mismatches: "
          f"{summary['prediction_mismatch_count']} | execution failures: "
          f"{summary['failure_count']}")
    print(f"  Random Forest loaded: {loaded_model}")
    if model_load_warning:
        print(f"  Random Forest load detail: {model_load_warning}")
    for family, data in summary["by_family"].items():
        print(f"  {family.title()}: {data['case_count']} cases, "
              f"{data['classified_count']} classified, "
              f"{data['uncertain_count']} uncertain, "
              f"{data['prediction_mismatch_count']} expected-label mismatches, "
              f"{data['failure_count']} execution failures")
        print(f"    Observed labels: {data['observed_labels']}")
    print(f"  Results saved to: {output_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="JSON output path (default: evaluation/classifier_harness_results.json)")
    args = parser.parse_args()
    run_harness(args.output)


if __name__ == "__main__":
    main()

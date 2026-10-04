"""End-to-end recommendation, sketch derivation, and mesh construction checks."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.candidate_constructors import (
    CANDIDATE_SPECS,
    CandidateMesh,
    CandidateParameterError,
    SketchError,
    build_candidate,
    derive_parameters_from_sketch,
    get_parameter_schema,
    validate_parameters,
)
from shapes.recommendation_api import (
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNCERTAIN,
    STATUS_UNSUPPORTED,
    get_recommendations,
)
from shapes.shape_taxonomy import (
    get_3d_candidate_ids_for_shape,
    load_taxonomy,
)
from shapes.stroke_analysis import analyze
from shapes.stroke_classifier import ShapeClass, StrokeClassifier
from shapes.stroke_normalization import preprocess
from tests.test_candidate_constructors import SKETCHES


CLASS_SKETCH = {
    "circle": "circle",
    "ellipse": "ellipse",
    "triangle": "triangle",
    "square": "square",
    "rectangle": "rectangle",
    "pentagon": "pentagon",
    "hexagon": "hexagon",
    "other_regular_polygon": "9-gon",
    "irregular_polygon": "polygon",
    "straight_line": "straight_line",
    "polyline": "polyline",
    "arc": "arc",
    "curve": "curve",
    "freeform_path": "freeform_path",
}


def _assert_mesh_valid(test, mesh):
    test.assertIsInstance(mesh, CandidateMesh)
    test.assertEqual(mesh.vertices.ndim, 2)
    test.assertEqual(mesh.vertices.shape[1], 3)
    test.assertEqual(mesh.normals.shape, mesh.vertices.shape)
    test.assertGreater(len(mesh.indices), 0)
    test.assertEqual(len(mesh.indices) % 3, 0)
    test.assertTrue(np.all(np.isfinite(mesh.vertices)))
    test.assertTrue(np.all(np.isfinite(mesh.normals)))
    test.assertTrue(all(0 <= index < len(mesh.vertices) for index in mesh.indices))


class RecommendationConstructionIntegrationTest(unittest.TestCase):
    def test_all_taxonomy_classifications_recommend_schema_valid_constructible_candidates(self):
        taxonomy = load_taxonomy()
        shapes_by_id = {shape["id"]: shape for shape in taxonomy["classes"]}
        all_ids = set(CANDIDATE_SPECS)
        covered_ids = set()

        for shape_id, sketch_label in CLASS_SKETCH.items():
            shape = shapes_by_id[f"shape.{('closed' if shape_id in {'circle', 'ellipse', 'triangle', 'square', 'rectangle', 'pentagon', 'hexagon', 'other_regular_polygon', 'irregular_polygon'} else 'open')}.{shape_id}"]
            classifier_label = (
                "9-gon" if shape_id == "other_regular_polygon"
                else "polygon" if shape_id == "irregular_polygon"
                else shape_id
            )
            # ShapeClass is the stable result contract emitted by the existing
            # classifier; this loop exercises every taxonomy label, including
            # categories that a particular sample is ambiguous about.
            classifier_result = ShapeClass(classifier_label, 0.9)
            recommendation = get_recommendations(classifier_result)
            expected_ids = get_3d_candidate_ids_for_shape(classifier_label)
            self.assertEqual(recommendation.status, STATUS_OK, shape_id)
            self.assertEqual(recommendation.shape_id, shape["id"])
            self.assertEqual(recommendation.candidate_ids, expected_ids)
            self.assertEqual(
                [candidate.rank for candidate in recommendation.candidates],
                list(range(1, len(expected_ids) + 1)),
            )
            self.assertEqual(
                get_recommendations(classifier_result).candidate_ids, expected_ids,
                "candidate order must be stable across repeated lookup",
            )

            sketch = SKETCHES[sketch_label]
            source_sides = recommendation.source_sides
            for candidate in recommendation.candidates:
                with self.subTest(shape=shape_id, candidate=candidate.id):
                    self.assertIn(candidate.id, all_ids)
                    self.assertEqual(candidate.construction_status, "candidate")
                    self.assertEqual(
                        candidate.requires_source_sides,
                        bool("sides" in CANDIDATE_SPECS[candidate.id].param_names),
                    )
                    schema = get_parameter_schema(candidate.id)
                    self.assertEqual(schema["id"], candidate.id)
                    self.assertTrue(schema["parameters"])
                    self.assertTrue(schema["sketch_derivation"])

                    params = derive_parameters_from_sketch(
                        candidate.id, sketch, source_sides=source_sides
                    )
                    self.assertEqual(set(params), set(CANDIDATE_SPECS[candidate.id].param_names))
                    self.assertEqual(validate_parameters(candidate.id, params), params)
                    _assert_mesh_valid(self, build_candidate(candidate.id, params))
                    covered_ids.add(candidate.id)

        # The taxonomy intentionally exposes only the simple user-facing
        # shortlist; every exposed candidate is still schema-valid and
        # constructible, while additional registered constructors remain
        # available for future use without appearing in recommendations.
        self.assertTrue(covered_ids)
        self.assertTrue(covered_ids.issubset(all_ids))

    def test_real_classifier_results_flow_into_bounded_recommendations(self):
        classifier = StrokeClassifier(use_ml_model=False)
        closed_labels = {
            "circle", "ellipse", "triangle", "square", "rectangle",
            "pentagon", "hexagon", "polygon",
        }
        for label, sketch in SKETCHES.items():
            points = list(sketch)
            closed = label in closed_labels or label.endswith("-gon")
            features = analyze(preprocess(points, n=64))
            classified = classifier.classify(points, features, closed)
            recommendation = get_recommendations(classified)
            with self.subTest(sketch_label=label, actual=classified.kind):
                if classified.kind in {"uncertain", "unknown"}:
                    self.assertEqual(recommendation.status, STATUS_UNCERTAIN)
                    self.assertEqual(recommendation.candidates, ())
                else:
                    self.assertEqual(recommendation.status, STATUS_OK)
                    self.assertEqual(
                        recommendation.candidate_ids,
                        get_3d_candidate_ids_for_shape(classified.kind),
                    )

    def test_uncertain_unknown_and_unsupported_inputs_have_no_candidates(self):
        cases = (
            (ShapeClass("uncertain", 0.2), STATUS_UNCERTAIN),
            (ShapeClass("unknown", 0.0), STATUS_UNCERTAIN),
            (ShapeClass("not-a-shape", 1.0), STATUS_UNSUPPORTED),
            (ShapeClass(None, 0.0), STATUS_INVALID),
            (None, STATUS_INVALID),
        )
        for classifier_result, expected_status in cases:
            with self.subTest(result=classifier_result):
                result = get_recommendations(classifier_result)
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.candidates, ())
                self.assertIsNone(result.default_candidate_id)

    def test_malformed_sketch_and_parameter_data_fail_safely(self):
        for candidate_id in get_3d_candidate_ids_for_shape("circle"):
            with self.subTest(candidate=candidate_id):
                with self.assertRaises(SketchError):
                    derive_parameters_from_sketch(candidate_id, [(0, 0), (1, math.nan)])

        with self.assertRaises(CandidateParameterError):
            validate_parameters("solid.cube", {"size": math.nan})
        with self.assertRaises(CandidateParameterError):
            build_candidate("solid.cube", {"size": -1})


if __name__ == "__main__":
    unittest.main()

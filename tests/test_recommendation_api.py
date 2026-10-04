"""Tests for the shared classifier-output -> 3D candidate recommendation API."""

import json
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.recommendation_api import (
    STATUS_INVALID,
    STATUS_OK,
    STATUS_UNCERTAIN,
    STATUS_UNSUPPORTED,
    get_recommendations,
)
from shapes.shape_recommender import ShapeRecommender
from shapes.shape_taxonomy import get_3d_candidate_ids_for_shape, load_taxonomy
from shapes.stroke_capture import StrokeRecord
from shapes.stroke_classifier import ShapeClass
from shapes.stroke_pipeline import StrokePipeline


SUPPORTED_LABELS = [
    "circle", "ellipse", "triangle", "square", "rectangle", "pentagon",
    "hexagon", "7-gon", "12-gon", "polygon",
    "straight_line", "polyline", "arc", "curve", "freeform_path",
]
EXPECTED_CANDIDATE_IDS = {
    "circle": ["solid.sphere", "solid.ellipsoid", "solid.cylinder", "solid.cone", "solid.torus"],
    "ellipse": ["solid.sphere", "solid.ellipsoid", "solid.cylinder", "solid.cone", "solid.torus"],
    "triangle": ["solid.triangular_prism", "solid.triangular_pyramid", "solid.cone"],
    "square": ["solid.cube", "solid.cuboid", "solid.square_prism", "solid.square_pyramid"],
    "rectangle": ["solid.cube", "solid.cuboid", "solid.rectangular_prism", "solid.rectangular_pyramid"],
    "pentagon": ["solid.pentagonal_prism", "solid.pentagonal_pyramid", "solid.cylinder", "solid.regular_cone"],
    "hexagon": ["solid.hexagonal_prism", "solid.hexagonal_pyramid", "solid.cylinder", "solid.regular_cone"],
    "7-gon": ["solid.regular_prism", "solid.regular_pyramid", "solid.cylinder", "solid.regular_cone"],
    "12-gon": ["solid.regular_prism", "solid.regular_pyramid", "solid.cylinder", "solid.regular_cone"],
    "polygon": ["solid.profile_prism", "solid.profile_loft", "solid.profile_pyramid"],
    "straight_line": ["sweep.rod", "sweep.beam", "solid.cylinder"],
    "polyline": ["sweep.beam", "sweep.polyline_beam", "sweep.swept_ribbon"],
    "arc": ["sweep.arc_tube", "sweep.curved_beam", "sweep.arc_ribbon"],
    "curve": ["sweep.curved_tube", "sweep.curved_beam", "sweep.swept_ribbon"],
    "freeform_path": ["sweep.swept_ribbon", "sweep.path_tube", "sweep.freeform_beam"],
}
EXPECTED_CANDIDATE_LABELS = {
    "circle": ["Sphere", "Ellipsoid", "Cylinder", "Cone", "Torus"],
    "ellipse": ["Sphere", "Ellipsoid", "Cylinder", "Cone", "Torus"],
    "triangle": ["Triangular Prism", "Triangular Pyramid", "Cone"],
    "square": ["Cube", "Cuboid", "Prism", "Pyramid"],
    "rectangle": ["Cube", "Cuboid", "Prism", "Pyramid"],
    "pentagon": ["Prism", "Pyramid", "Cylinder", "Conical Form"],
    "hexagon": ["Prism", "Pyramid", "Cylinder", "Conical Form"],
    "7-gon": ["Prism", "Pyramid", "Cylinder", "Conical Form"],
    "12-gon": ["Prism", "Pyramid", "Cylinder", "Conical Form"],
    "polygon": ["Extruded Polygon", "Irregular Prism", "Irregular Pyramid"],
    "straight_line": ["Rod", "Beam", "Cylinder"],
    "polyline": ["Beam", "Polygonal Tube", "Ribbon"],
    "arc": ["Arc Tube", "Curved Beam", "Ribbon"],
    "curve": ["Curved Tube", "Curved Beam", "Ribbon"],
    "freeform_path": ["Ribbon", "Tube", "Beam"],
}


class RecommendationApiTest(unittest.TestCase):
    def test_every_supported_label_matches_taxonomy_order(self):
        for label in SUPPORTED_LABELS:
            with self.subTest(label=label):
                result = get_recommendations(ShapeClass(label, 0.9))
                expected = EXPECTED_CANDIDATE_IDS[label]
                self.assertEqual(result.status, STATUS_OK)
                self.assertEqual(result.candidate_ids, expected)
                self.assertEqual(
                    [candidate.label for candidate in result.candidates],
                    EXPECTED_CANDIDATE_LABELS[label],
                )
                self.assertEqual(
                    get_3d_candidate_ids_for_shape(label), expected
                )
                self.assertEqual(result.default_candidate_id, expected[0])
                self.assertEqual([c.rank for c in result.candidates],
                                 list(range(1, len(expected) + 1)))
                self.assertEqual(result.confidence, 0.9)

    def test_input_forms_are_equivalent(self):
        from_class = get_recommendations(ShapeClass("square", 0.8))
        from_dict = get_recommendations({"kind": "square", "confidence": 0.8})
        from_str = get_recommendations(" Square ", confidence=0.8)
        self.assertEqual(from_class, from_dict)
        self.assertEqual(from_class, from_str)
        self.assertEqual(from_class.shape_id, "shape.closed.square")
        self.assertEqual(from_class.source_sides, 4)

    def test_dynamic_ngon_carries_sides_and_display_label(self):
        result = get_recommendations("9-gon")
        self.assertEqual(result.shape_id, "shape.closed.other_regular_polygon")
        self.assertEqual(result.display_label, "9-gon")
        self.assertEqual(result.source_sides, 9)
        self.assertEqual(result.candidate_ids, EXPECTED_CANDIDATE_IDS["7-gon"])
        self.assertTrue(all(
            candidate.requires_source_sides
            for candidate in result.candidates
            if candidate.id != "solid.cylinder"
        ))

    def test_uncertain_unsupported_and_invalid_return_no_candidates(self):
        cases = {
            "uncertain": STATUS_UNCERTAIN,
            "unknown": STATUS_UNCERTAIN,
            "hexagram": STATUS_UNSUPPORTED,
            "2-gon": STATUS_UNSUPPORTED,
            "": STATUS_INVALID,
            None: STATUS_INVALID,
            42: STATUS_INVALID,
        }
        for label, status in cases.items():
            with self.subTest(label=label):
                result = get_recommendations(label)
                self.assertEqual(result.status, status)
                self.assertEqual(result.candidates, ())
                self.assertFalse(result.has_candidates)
                self.assertIsNone(result.default_candidate_id)
        self.assertEqual(get_recommendations(object()).status, STATUS_INVALID)

    def test_confidence_is_sanitized(self):
        for raw, expected in ((1.7, 1.0), (-0.2, 0.0), (math.nan, None),
                              (math.inf, None), ("0.5", None), (True, None)):
            with self.subTest(raw=raw):
                self.assertEqual(
                    get_recommendations("circle", confidence=raw).confidence,
                    expected)

    def test_max_candidates(self):
        result = get_recommendations("circle", max_candidates=2)
        self.assertEqual(result.candidate_ids, ["solid.sphere", "solid.ellipsoid"])
        for bad in (0, -1, 1.5, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                get_recommendations("circle", max_candidates=bad)

    def test_to_dict_is_json_serializable(self):
        payload = json.loads(json.dumps(get_recommendations("triangle", 0.75).to_dict()))
        self.assertEqual(payload["status"], STATUS_OK)
        self.assertEqual(payload["default_candidate_id"], "solid.triangular_prism")
        self.assertEqual(payload["candidates"][0]["rank"], 1)
        json.dumps(get_recommendations("uncertain").to_dict())

    def test_results_are_detached_from_taxonomy(self):
        before = load_taxonomy()
        result = get_recommendations("circle")
        with self.assertRaises(Exception):
            result.candidates[0].id = "mutated"
        self.assertEqual(load_taxonomy(), before)


class RecommendationIntegrationTest(unittest.TestCase):
    def test_recommender_mesh_recipe_is_unchanged(self):
        recommender = ShapeRecommender()
        self.assertEqual(
            recommender.recommend_candidates(ShapeClass("circle", 1.0)).candidate_ids,
            get_3d_candidate_ids_for_shape("circle"))

    def test_pipeline_attaches_candidates_without_changing_mesh_kind(self):
        rect = [(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]
        rect = [(x0 + (x1 - x0) * t / 20, y0 + (y1 - y0) * t / 20)
                for (x0, y0), (x1, y1) in zip(rect, rect[1:]) for t in range(20)]
        rect.append(rect[0])
        record = StrokeRecord(raw_points=list(rect), points=list(rect))
        result = StrokePipeline().convert(record)

        self.assertIsNotNone(result.candidates)
        self.assertIs(record.candidates, result.candidates)
        self.assertEqual(result.candidates.classifier_label, result.shape_class.kind)
        if result.shape_class.kind == "uncertain":
            self.assertEqual(result.candidates.status, STATUS_UNCERTAIN)
        else:
            self.assertEqual(result.candidates.status, STATUS_OK)
            self.assertEqual(result.recommendation.kind, "polygon")


if __name__ == "__main__":
    unittest.main()

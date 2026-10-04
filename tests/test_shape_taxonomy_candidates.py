"""Tests for bounded, data-driven 2D-to-3D candidate mappings."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shapes.shape_taxonomy import (
    get_3d_candidate_ids_for_shape,
    get_3d_candidates_for_shape,
    load_taxonomy,
)
from shapes.candidate_constructors import list_constructible_candidates


EXPECTED_CANDIDATE_IDS = {
    "circle": ["solid.sphere", "solid.ellipsoid", "solid.cylinder", "solid.cone", "solid.torus"],
    "ellipse": ["solid.sphere", "solid.ellipsoid", "solid.cylinder", "solid.cone", "solid.torus"],
    "triangle": ["solid.triangular_prism", "solid.triangular_pyramid", "solid.cone"],
    "square": ["solid.cube", "solid.cuboid", "solid.square_prism", "solid.square_pyramid"],
    "rectangle": ["solid.cube", "solid.cuboid", "solid.rectangular_prism", "solid.rectangular_pyramid"],
    "pentagon": ["solid.pentagonal_prism", "solid.pentagonal_pyramid", "solid.cylinder", "solid.regular_cone"],
    "hexagon": ["solid.hexagonal_prism", "solid.hexagonal_pyramid", "solid.cylinder", "solid.regular_cone"],
    "7-gon": ["solid.regular_prism", "solid.regular_pyramid", "solid.cylinder", "solid.regular_cone"],
    "polygon": ["solid.profile_prism", "solid.profile_loft", "solid.profile_pyramid"],
    "straight_line": ["sweep.rod", "sweep.beam", "solid.cylinder"],
    "polyline": ["sweep.beam", "sweep.polyline_beam", "sweep.swept_ribbon"],
    "arc": ["sweep.arc_tube", "sweep.curved_beam", "sweep.arc_ribbon"],
    "curve": ["sweep.curved_tube", "sweep.curved_beam", "sweep.swept_ribbon"],
    "freeform_path": ["sweep.swept_ribbon", "sweep.path_tube", "sweep.freeform_beam"],
}


class ShapeTaxonomyCandidateTest(unittest.TestCase):
    def test_every_taxonomy_class_has_valid_bounded_candidate_references(self):
        taxonomy = load_taxonomy()
        candidate_ids = {
            candidate["id"] for candidate in taxonomy["candidate_3d_families"]
        }
        self.assertTrue(candidate_ids)

        for shape in taxonomy["classes"]:
            labels = list(shape.get("classifier_labels", []))
            if shape.get("classifier_label_pattern"):
                labels.append("9-gon")
            if not labels:
                self.fail(f"No classifier label or pattern for {shape['id']}")

            expected_ids = shape["candidate_3d_family_ids"]
            self.assertTrue(set(expected_ids).issubset(candidate_ids), shape["id"])
            if shape["category"] == "classification_state":
                self.assertEqual(expected_ids, [], shape["id"])
            else:
                self.assertTrue(expected_ids, shape["id"])

            for label in labels:
                with self.subTest(shape_id=shape["id"], label=label):
                    self.assertEqual(get_3d_candidate_ids_for_shape(label), expected_ids)
                    self.assertEqual(
                        [candidate["id"] for candidate in get_3d_candidates_for_shape(label)],
                        expected_ids,
                    )

    def test_exact_simple_candidate_ids_and_order_for_every_shape(self):
        constructible = set(list_constructible_candidates())
        for label, expected_ids in EXPECTED_CANDIDATE_IDS.items():
            with self.subTest(label=label):
                self.assertEqual(get_3d_candidate_ids_for_shape(label), expected_ids)
                self.assertTrue(set(expected_ids).issubset(constructible), label)

    def test_uncertain_and_unsupported_labels_return_empty_candidates(self):
        for label in ("uncertain", "unknown", "not-a-shape", ""):
            with self.subTest(label=label):
                self.assertEqual(get_3d_candidates_for_shape(label), [])
                self.assertEqual(get_3d_candidate_ids_for_shape(label), [])
        self.assertEqual(get_3d_candidates_for_shape(None), [])


if __name__ == "__main__":
    unittest.main()

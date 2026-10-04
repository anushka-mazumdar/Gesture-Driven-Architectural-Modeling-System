"""Regression checks for circle construction and its recommendation panel."""

import math
import unittest

from interaction.recommendation_mode import RecommendationInteractionMode
from render.primitives import PolygonMesh
from render.threejs_renderer import ThreeJSRenderer
from shapes.stroke_capture import StrokeRecord
from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_pipeline import StrokePipeline
from shapes.shape_3d_factory import Shape3DFactory


def _stroke(points):
    return StrokeRecord(raw_points=list(points), points=list(points))


def _circle_points():
    return [
        (320 + 100 * math.cos(2 * math.pi * index / 64),
         240 + 100 * math.sin(2 * math.pi * index / 64))
        for index in range(65)
    ]


def _square_points():
    return [(220, 140), (420, 140), (420, 340), (220, 340), (220, 140)]


class CircleDefaultCandidateTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = StrokePipeline(
            classifier=StrokeClassifier(use_ml_model=False),
            factory=Shape3DFactory(panel_width=640, panel_height=480),
        )

    def test_circle_uses_sphere_and_panel_keeps_all_configured_candidates(self):
        result = self.pipeline.convert(_stroke(_circle_points()))
        self.assertEqual(result.shape_class.kind, "circle")
        self.assertEqual(result.candidates.default_candidate_id, "solid.sphere")
        self.assertEqual(result.mesh.candidate_id, "solid.sphere")

        renderer = ThreeJSRenderer()
        renderer.set_candidate_recommendation(result.candidates)
        self.assertTrue(renderer.has_candidate_panel)
        self.assertEqual(
            [item["id"] for item in renderer._candidate_panel["candidates"]],
            ["solid.sphere", "solid.ellipsoid", "solid.cylinder",
             "solid.cone", "solid.torus"],
        )
        mode = RecommendationInteractionMode(renderer)
        self.assertTrue(mode.update(None, now=1.0))
        self.assertEqual(renderer._candidate_index, 0)

        # Alternate candidates remain navigable and fist-confirmable.
        self.assertTrue(renderer.navigate_candidate("SWIPE_LEFT"))
        selected_id = renderer._candidate_panel["candidates"][1]["id"]
        self.assertTrue(renderer.confirm_candidate(selected_id))
        self.assertEqual(renderer.selected_candidate["id"], selected_id)
        self.assertFalse(renderer.has_candidate_panel)

    def test_other_closed_shape_keeps_legacy_extruded_mesh(self):
        result = self.pipeline.convert(_stroke(_square_points()))
        self.assertEqual(result.shape_class.kind, "square")
        self.assertIsInstance(result.mesh, PolygonMesh)


if __name__ == "__main__":
    unittest.main()

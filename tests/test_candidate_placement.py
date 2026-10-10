"""Confirmed recommendation candidates replace previews using captured sketches."""

import math
import unittest

import numpy as np

from interaction.candidate_placement import CandidatePlacement
from interaction.object_selection import ObjectSelection
from render.threejs_renderer import ThreeJSRenderer
from shapes.candidate_constructors import CandidateMesh, derive_parameters_from_sketch
from shapes.shape_3d_factory import Shape3DFactory
from shapes.stroke_capture import StrokeRecord
from shapes.stroke_classifier import ShapeClass
from shapes.stroke_pipeline import StrokePipeline


class FixedClassifier:
    def __init__(self, label):
        self.label = label

    def classify(self, points, features, closed):
        return ShapeClass(self.label, 0.99)


def _circle():
    return [(320 + 90 * math.cos(2 * math.pi * i / 48),
             240 + 90 * math.sin(2 * math.pi * i / 48)) for i in range(49)]


SKETCHES = {
    "circle": _circle(),
    "square": [(220, 140), (420, 140), (420, 340), (220, 340), (220, 140)],
    "rectangle": [(180, 150), (460, 150), (460, 310), (180, 310), (180, 150)],
    "triangle": [(320, 120), (440, 330), (200, 330), (320, 120)],
    "straight_line": [(120, 220), (180, 215), (250, 212), (330, 205), (420, 200)],
}


class CandidatePlacementTests(unittest.TestCase):
    def _convert_and_confirm_last_candidate(self, label):
        points = SKETCHES[label]
        record = StrokeRecord(raw_points=list(points), points=list(points))
        result = StrokePipeline(
            classifier=FixedClassifier(label),
            factory=Shape3DFactory(panel_width=640, panel_height=480),
        ).convert(record)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape_class.kind, label)
        self.assertEqual(result.candidates.status, "ok")

        renderer = ThreeJSRenderer()
        renderer.add_object(result.mesh)
        renderer.set_candidate_recommendation(result.candidates)
        selection = ObjectSelection(renderer)
        placement = CandidatePlacement(renderer, selection=selection)
        self.assertTrue(placement.begin(result.record, result.mesh))

        preview = result.mesh
        preview.position[:] = [13.0, -7.0, 24.0]
        preview.rotation[:] = [0.1, 0.2, 0.3]
        preview.scale[:] = [1.2, 0.8, 1.1]
        preview.color = (239 / 255.0, 83 / 255.0, 80 / 255.0)

        candidate_ids = result.candidates.candidate_ids
        for _ in range(len(candidate_ids) - 1):
            self.assertTrue(renderer.navigate_candidate("SWIPE_LEFT"))
        selected_id = candidate_ids[-1]
        expected_parameters = derive_parameters_from_sketch(
            selected_id, points, source_sides=result.candidates.source_sides
        )

        self.assertTrue(placement.confirm(selected_id))
        self.assertEqual(renderer.selected_candidate["id"], selected_id)
        self.assertEqual(len(renderer.objects), 1)
        placed = renderer.objects[0]
        self.assertIsNot(placed, preview)
        self.assertIsInstance(placed, CandidateMesh)
        self.assertEqual(placed.candidate_id, selected_id)
        self.assertEqual(placed.parameters, expected_parameters)
        np.testing.assert_allclose(placed.position, [13.0, -7.0, 24.0])
        np.testing.assert_allclose(placed.rotation, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(placed.scale, [1.2, 0.8, 1.1])
        np.testing.assert_allclose(placed.color, preview.color)
        self.assertEqual(placed.shape_class.kind, label)
        self.assertIs(placement.active_object, placed)
        self.assertIs(selection.get_selected(), placed)
        self.assertTrue(placed.selected)
        self.assertFalse(renderer.has_candidate_panel)
        self.assertIsNone(placement.record)
        self.assertFalse(placement.confirm(selected_id))
        self.assertEqual(len(renderer.objects), 1)

    def test_circle_selection_constructs_and_replaces_preview(self):
        self._convert_and_confirm_last_candidate("circle")

    def test_square_selection_constructs_and_replaces_preview(self):
        self._convert_and_confirm_last_candidate("square")

    def test_rectangle_selection_constructs_and_replaces_preview(self):
        self._convert_and_confirm_last_candidate("rectangle")

    def test_triangle_selection_constructs_and_replaces_preview(self):
        self._convert_and_confirm_last_candidate("triangle")

    def test_open_stroke_selection_constructs_and_replaces_preview(self):
        self._convert_and_confirm_last_candidate("straight_line")

    def test_new_recommendation_replaces_stale_pending_preview(self):
        renderer = ThreeJSRenderer()
        placement = CandidatePlacement(renderer)
        first = self._convert("square")
        renderer.set_candidate_recommendation(first.candidates)
        renderer.add_object(first.mesh)
        self.assertTrue(placement.begin(first.record, first.mesh))

        second = self._convert("rectangle")
        renderer.set_candidate_recommendation(second.candidates)
        renderer.add_object(second.mesh)
        self.assertTrue(placement.begin(second.record, second.mesh))
        self.assertNotIn(first.mesh, renderer.objects)
        self.assertEqual(renderer.objects, [second.mesh])

    def test_cancel_removes_only_pending_preview_and_closes_recommendation(self):
        renderer = ThreeJSRenderer()
        placement = CandidatePlacement(renderer)
        result = self._convert("triangle")
        renderer.set_candidate_recommendation(result.candidates)
        renderer.add_object(result.mesh)
        self.assertTrue(placement.begin(result.record, result.mesh))

        placement.cancel()
        self.assertEqual(renderer.objects, [])
        self.assertFalse(renderer.has_candidate_panel)
        self.assertIsNone(placement.preview_mesh)
        self.assertIsNone(placement.record)

    def test_reset_removes_the_active_constructed_object(self):
        renderer = ThreeJSRenderer()
        selection = ObjectSelection(renderer)
        placement = CandidatePlacement(renderer, selection=selection)
        result = self._convert("circle")
        renderer.set_candidate_recommendation(result.candidates)
        renderer.add_object(result.mesh)
        placement.begin(result.record, result.mesh)
        candidate_id = result.candidates.candidate_ids[0]
        self.assertTrue(placement.confirm(candidate_id))
        active = placement.active_object
        self.assertEqual(renderer.objects, [active])

        placement.reset()
        self.assertEqual(renderer.objects, [])
        self.assertIsNone(placement.active_object)
        self.assertIsNone(selection.get_selected())
        self.assertFalse(renderer.has_candidate_panel)

    def test_stale_confirmation_cannot_consume_new_panel_or_add_object(self):
        renderer = ThreeJSRenderer()
        placement = CandidatePlacement(renderer)
        old_result = self._convert("square")
        renderer.set_candidate_recommendation(old_result.candidates)
        renderer.add_object(old_result.mesh)
        placement.begin(old_result.record, old_result.mesh)

        current_result = self._convert("square")
        renderer.set_candidate_recommendation(current_result.candidates)
        renderer.add_object(current_result.mesh)
        stale_id = old_result.candidates.candidate_ids[0]
        self.assertFalse(placement.confirm(stale_id))
        self.assertEqual(renderer.objects, [current_result.mesh])
        self.assertTrue(renderer.has_candidate_panel)
        self.assertIsNone(placement.active_object)

        placement.begin(current_result.record, current_result.mesh)
        current_id = current_result.candidates.candidate_ids[0]
        self.assertTrue(placement.confirm(current_id))
        self.assertEqual(len(renderer.objects), 1)
        self.assertEqual(renderer.objects[0].candidate_id, current_id)

    def _convert(self, label):
        points = SKETCHES[label]
        record = StrokeRecord(raw_points=list(points), points=list(points))
        return StrokePipeline(
            classifier=FixedClassifier(label),
            factory=Shape3DFactory(panel_width=640, panel_height=480),
        ).convert(record)


if __name__ == "__main__":
    unittest.main()

import unittest

from gestures.candidate_swipe_gate import CandidateSwipeGate
from interaction.recommendation_mode import (
    MODE_NORMAL,
    MODE_RECOMMENDATION,
    RecommendationInteractionMode,
)
from render.threejs_renderer import ThreeJSRenderer


class CandidateNavigationTests(unittest.TestCase):
    def setUp(self):
        self.renderer = ThreeJSRenderer()
        self.renderer.set_candidate_recommendation({
            "status": "ok",
            "display_label": "Circle",
            "candidates": [
                {"id": "sphere", "label": "Sphere"},
                {"id": "cylinder", "label": "Cylinder"},
                {"id": "cone", "label": "Cone"},
            ],
        })

    def test_candidates_keep_api_order_and_navigation_wraps(self):
        self.assertEqual(
            [item["id"] for item in self.renderer._candidate_panel["candidates"]],
            ["sphere", "cylinder", "cone"],
        )
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertTrue(self.renderer.navigate_candidate("SWIPE_RIGHT"))
        self.assertEqual(self.renderer._candidate_index, 2)
        self.assertTrue(self.renderer.navigate_candidate("SWIPE_LEFT"))
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertFalse(self.renderer.navigate_candidate("SWIPE_UP"))

    def test_empty_or_uncertain_result_clears_navigation_state(self):
        self.renderer.navigate_candidate("SWIPE_RIGHT")
        self.renderer.set_candidate_recommendation({
            "status": "uncertain", "display_label": "Uncertain", "candidates": []
        })
        self.assertFalse(self.renderer.has_candidate_panel)
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertFalse(self.renderer.navigate_candidate("SWIPE_LEFT"))

    def test_candidate_confirmation_is_current_only_and_once(self):
        self.renderer.navigate_candidate("SWIPE_LEFT")
        self.assertFalse(self.renderer.confirm_candidate("sphere"))
        self.assertTrue(self.renderer.confirm_candidate("cylinder"))
        self.assertFalse(self.renderer.has_candidate_panel)
        self.assertEqual(self.renderer.selected_candidate, {
            "id": "cylinder", "label": "Cylinder", "shape_label": "Circle",
        })
        self.assertFalse(self.renderer.confirm_candidate("cylinder"))

    def test_swipe_gate_settles_rejects_jitter_and_applies_cooldown(self):
        gate = CandidateSwipeGate(settle_seconds=0.1, cooldown_seconds=0.5)
        self.assertIsNone(gate.update("SWIPE_RIGHT", now=1.0))
        self.assertIsNone(gate.update("SWIPE_LEFT", now=1.04))
        self.assertIsNone(gate.update(None, now=1.2))

        self.assertIsNone(gate.update("SWIPE_RIGHT", now=2.0))
        self.assertEqual(gate.update(None, now=2.11), "SWIPE_RIGHT")
        self.assertIsNone(gate.update("SWIPE_RIGHT", now=2.3))
        self.assertIsNone(gate.update(None, now=2.7))
        self.assertIsNone(gate.update("SWIPE_LEFT", active=False, now=3.0))
        self.assertIsNone(gate.update(None, now=3.2))

    def test_recommendation_mode_owns_and_consumes_hand_input(self):
        mode = RecommendationInteractionMode(
            self.renderer,
            CandidateSwipeGate(settle_seconds=0.1, cooldown_seconds=0.5),
        )
        # The frame that first opens the recommendation is consumed but is
        # not treated as a swipe inside the newly opened panel.
        self.assertTrue(mode.update("SWIPE_RIGHT", now=1.0))
        self.assertEqual(mode.mode, MODE_RECOMMENDATION)
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertTrue(mode.update("SWIPE_RIGHT", now=1.2))
        # Intermediate frames are consumed too, rather than reaching normal
        # drawing, selection, manipulation, deletion, or depth handlers.
        self.assertTrue(mode.update(None, now=1.31))
        self.assertEqual(self.renderer._candidate_index, 2)
        self.assertTrue(mode.update("SWIPE_UP", now=1.4))
        self.assertEqual(self.renderer._candidate_index, 2)

    def test_panel_dismissal_resets_mode_selection_and_pending_swipe(self):
        gate = CandidateSwipeGate(settle_seconds=0.2, cooldown_seconds=0.5)
        mode = RecommendationInteractionMode(self.renderer, gate)
        self.renderer.navigate_candidate("SWIPE_RIGHT")
        mode.update(None, now=1.9)
        mode.update("SWIPE_LEFT", now=2.0)
        self.assertEqual(gate.pending_direction, "SWIPE_LEFT")
        mode.dismiss()
        self.assertEqual(mode.mode, MODE_NORMAL)
        self.assertFalse(self.renderer.has_candidate_panel)
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertIsNone(gate.pending_direction)
        self.assertFalse(mode.update("SWIPE_RIGHT", now=2.3))

    def test_uncertain_result_releases_recommendation_mode(self):
        mode = RecommendationInteractionMode(self.renderer)
        self.assertTrue(mode.update(None, now=5.0))
        self.renderer.set_candidate_recommendation({
            "status": "uncertain", "display_label": "Uncertain", "candidates": []
        })
        # The closing transition consumes this frame so the same hand pose
        # cannot also trigger a normal-mode scene action.
        self.assertTrue(mode.update("SWIPE_LEFT", now=5.2))
        self.assertEqual(mode.mode, MODE_NORMAL)
        self.assertFalse(mode.update(None, now=5.3))

    def test_confirmed_candidate_closes_panel_and_returns_mode_safely(self):
        mode = RecommendationInteractionMode(self.renderer)
        self.assertTrue(mode.update(None, now=1.0))
        self.assertTrue(self.renderer.confirm_candidate("sphere"))
        self.assertFalse(self.renderer.has_candidate_panel)
        # The held-fist frame that closed the panel is consumed.
        self.assertTrue(mode.update("FIST", now=2.0))
        self.assertEqual(mode.mode, MODE_NORMAL)
        self.assertFalse(mode.update(None, now=2.1))

    def test_cancelling_panel_returns_control_to_normal_mode(self):
        mode = RecommendationInteractionMode(self.renderer)
        self.assertTrue(mode.update(None, now=3.0))
        self.renderer.set_candidate_recommendation(None)
        self.assertTrue(mode.update("SWIPE_LEFT", now=3.1))
        self.assertEqual(mode.mode, MODE_NORMAL)
        self.assertFalse(mode.update(None, now=3.2))


if __name__ == "__main__":
    unittest.main()

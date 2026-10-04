"""Search stays scoped to candidates from the shared Recommendation API."""

import unittest

from render.threejs_renderer import ThreeJSRenderer
from shapes.recommendation_api import get_recommendations


class CandidateSearchTests(unittest.TestCase):
    def setUp(self):
        self.renderer = ThreeJSRenderer()
        self.recommendation = get_recommendations("circle")
        self.renderer.set_candidate_recommendation(self.recommendation)
        self.api_ids = self.recommendation.candidate_ids

    def test_search_filters_api_data_without_changing_stable_ids_or_order(self):
        self.assertTrue(self.renderer.set_candidate_search_query("cyl"))
        self.assertEqual(
            [candidate["id"] for candidate in self.renderer._visible_candidates()],
            ["solid.cylinder"],
        )
        self.assertEqual(
            [candidate["id"] for candidate in self.renderer._candidate_panel["candidates"]],
            self.api_ids,
        )

    def test_no_results_cannot_navigate_or_confirm_and_clearing_restores_roster(self):
        self.assertTrue(self.renderer.set_candidate_search_query("pyramid"))
        self.assertEqual(self.renderer._visible_candidates(), [])
        self.assertFalse(self.renderer.navigate_candidate("SWIPE_LEFT"))
        self.assertFalse(self.renderer.confirm_candidate("solid.sphere"))

        self.assertTrue(self.renderer.set_candidate_search_query(""))
        self.assertEqual(
            [candidate["id"] for candidate in self.renderer._visible_candidates()],
            self.api_ids,
        )

    def test_matching_candidate_still_uses_existing_navigation_and_confirmation(self):
        self.renderer.set_candidate_search_query("torus")
        self.assertEqual(
            [candidate["id"] for candidate in self.renderer._visible_candidates()],
            ["solid.torus"],
        )
        self.assertTrue(self.renderer.navigate_candidate("SWIPE_RIGHT"))
        self.assertEqual(self.renderer._candidate_index, 0)
        self.assertTrue(self.renderer.confirm_candidate("solid.torus"))
        self.assertEqual(self.renderer.selected_candidate["id"], "solid.torus")


if __name__ == "__main__":
    unittest.main()

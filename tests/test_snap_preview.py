import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from interaction.snapping import Snapping
from render.snap_anchors import SnapAnchor
from render.threejs_renderer import ThreeJSRenderer


def mesh(anchor_specs, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    return SimpleNamespace(
        snap_anchors=tuple(
            SnapAnchor(anchor_id, kind, tuple(point))
            for anchor_id, kind, point in anchor_specs
        ),
        position=np.asarray(position, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        scale=np.asarray(scale, dtype=float),
    )


class SnapPreviewTests(unittest.TestCase):
    def setUp(self):
        self.moving = mesh([("m-v", "vertex", (0, 0, 0))])

    def test_detects_nearest_compatible_anchor_pair(self):
        target = mesh([
            ("far", "vertex", (25, 0, 0)),
            ("near", "vertex", (12, 0, 0)),
            ("wrong-kind", "edge", (1, 0, 0)),
        ])
        snapping = Snapping(enabled=True)
        moving_before = self.moving.position.copy()
        target_before = target.position.copy()
        preview = snapping.preview_nearest_anchors(
            self.moving, [self.moving, target]
        )
        self.assertEqual(preview.target_anchor_id, "near")
        self.assertEqual(preview.moving_anchor_id, "m-v")
        self.assertEqual(preview.kind, "vertex")
        self.assertAlmostEqual(preview.distance, 12)
        np.testing.assert_array_equal(self.moving.position, moving_before)
        np.testing.assert_array_equal(target.position, target_before)
        self.assertIsNone(snapping.snap_candidate)
        self.assertEqual(snapping._groups, [])

    def test_threshold_is_inclusive_and_rejects_outside(self):
        snapping = Snapping(enabled=True)
        at_limit = mesh([("limit", "vertex", (snapping.ANCHOR_PREVIEW_DISTANCE, 0, 0))])
        self.assertIsNotNone(snapping.preview_nearest_anchors(self.moving, [at_limit]))
        outside = mesh([("outside", "vertex", (snapping.ANCHOR_PREVIEW_DISTANCE + 0.001, 0, 0))])
        self.assertIsNone(snapping.preview_nearest_anchors(self.moving, [outside]))

    def test_uses_transformed_world_anchor_positions(self):
        transformed = mesh([("v", "vertex", (2, 0, 0))],
                           position=(10, 20, 0), rotation=(0, 0, 90), scale=(2, 1, 1))
        target = mesh([("target", "vertex", (10, 24, 0))])
        preview = Snapping(enabled=True).preview_nearest_anchors(
            transformed, [transformed, target]
        )
        self.assertEqual(preview.moving_position, (10.0, 24.0, 0.0))
        self.assertEqual(preview.distance, 0)

    def test_disabled_and_no_valid_target_clear_preview(self):
        target = mesh([("v", "vertex", (2, 0, 0))])
        snapping = Snapping(enabled=False)
        self.assertIsNone(snapping.preview_nearest_anchors(self.moving, [target]))
        self.assertIsNone(snapping.snap_preview)
        self.assertIsNone(Snapping(enabled=True).preview_nearest_anchors(
            self.moving, [self.moving]
        ))
        incompatible = mesh([("edge", "edge", (1, 0, 0))])
        self.assertIsNone(Snapping(enabled=True).preview_nearest_anchors(
            self.moving, [incompatible]
        ))

    def test_equal_distance_selection_is_deterministic(self):
        first = mesh([("z-anchor", "vertex", (10, 0, 0))])
        second = mesh([("a-anchor", "vertex", (-10, 0, 0))])
        snapping = Snapping(enabled=True)
        previews = [snapping.preview_nearest_anchors(self.moving, [second, first])
                    for _ in range(3)]
        self.assertTrue(all(item.target_object is second for item in previews))
        self.assertEqual([item.target_anchor_id for item in previews], ["a-anchor"] * 3)

    def test_world_anchor_cache_hits_and_invalidates_for_transform_and_geometry(self):
        target = mesh([("v", "vertex", (2, 0, 0))])
        snapping = Snapping(enabled=True)
        first = snapping._world_anchors(target)[0][("vertex", "v")][1].copy()
        snapping._world_anchors(target)
        self.assertEqual(snapping.preview_stats["cache_rebuilds"], 1)
        self.assertEqual(snapping.preview_stats["cache_hits"], 1)
        target.position[0] = 10
        second = snapping._world_anchors(target)[0][("vertex", "v")][1]
        np.testing.assert_allclose(second - first, (10, 0, 0))
        target.snap_anchors = (SnapAnchor("v", "vertex", (5, 0, 0)),)
        third = snapping._world_anchors(target)[0][("vertex", "v")][1]
        np.testing.assert_allclose(third, (15, 0, 0))
        self.assertEqual(snapping.preview_stats["cache_rebuilds"], 3)

    def test_preview_commit_rejects_stale_transform_result(self):
        target = mesh([("target", "vertex", (10, 0, 0))])
        snapping = Snapping(enabled=True)
        preview = snapping.preview_nearest_anchors(self.moving, [self.moving, target])
        self.assertIsNotNone(preview)
        target.position[0] = 11
        before = self.moving.position.copy()
        self.assertFalse(snapping.commit_preview(self.moving, [self.moving, target]))
        np.testing.assert_array_equal(self.moving.position, before)
        self.assertIsNone(snapping.snap_preview)

    def test_spatial_index_limits_candidates_and_preserves_nearest_result(self):
        moving = mesh([("m", "vertex", (0, 0, 0))])
        target = mesh([("far", "vertex", (500, 0, 0)),
                       ("near", "vertex", (2, 0, 0))])
        # Add many anchors outside the neighboring cells to exercise pruning.
        target.snap_anchors += tuple(SnapAnchor(f"d{i}", "vertex", (1000+i*50, 0, 0))
                                     for i in range(100))
        snapping = Snapping(enabled=True)
        preview = snapping.preview_nearest_anchors(moving, [moving, target])
        self.assertEqual(preview.target_anchor_id, "near")
        self.assertLess(snapping.preview_stats["candidates_checked"], 5)

    def test_webview_preview_reuses_marker_and_connector_geometry(self):
        source = (Path(__file__).resolve().parents[1]
                  / "webview_app" / "web" / "main.js").read_text(encoding="utf-8")
        marker_setup = source.split("for (const [radius, halo]", 1)[1].split(
            "const snapPreviewPositions", 1
        )[0]
        self.assertIn("marker.visible = false", marker_setup)
        self.assertIn("snapPreviewGroup.add(snapPreviewConnector)", source)
        handler = source.split("window.receiveSnapPreviewFromPython = function (preview) {", 1)[1].split(
            "};", 1
        )[0]
        self.assertNotIn("new THREE.SphereGeometry", handler)
        self.assertNotIn("new THREE.BufferGeometry", handler)
        self.assertIn("snapPreviewPositions.set(values)", handler)
        self.assertIn("snapPreviewGeometry.attributes.position.needsUpdate = true", handler)

    def test_renderer_preview_creation_and_removal_are_separate_from_meshes(self):
        class Window:
            def __init__(self):
                self.calls = []

            def evaluate_js(self, script):
                self.calls.append(script)

        renderer = ThreeJSRenderer()
        renderer._window = Window()
        renderer._ready = True
        target = mesh([("target-v", "vertex", (10, 0, 0))])
        preview = Snapping(enabled=True).preview_nearest_anchors(
            self.moving, [self.moving, target]
        )
        renderer.set_snap_preview(preview)
        self.assertEqual(renderer._snap_preview["target_anchor_id"], "target-v")
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())
        self.assertTrue(any('"snap_preview": {' in call
                            and '"target_anchor_id": "target-v"' in call
                            for call in renderer._window.calls))
        self.assertTrue(any('"meshes": {"added": [], "removed": [], "transforms": []}' in call
                            for call in renderer._window.calls))
        renderer.set_snap_preview(None)
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())
        self.assertIsNone(renderer._snap_preview)
        self.assertTrue(any('"snap_preview": null' in call
                            for call in renderer._window.calls))


if __name__ == "__main__":
    unittest.main()

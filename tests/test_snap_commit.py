import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from interaction.snapping import Snapping
from render.snap_anchors import SnapAnchor


def make_mesh(anchors, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    return SimpleNamespace(
        snap_anchors=tuple(anchors),
        position=np.asarray(position, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        scale=np.asarray(scale, dtype=float),
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
    )


def vertex(anchor_id, point, key):
    return SnapAnchor(anchor_id, "vertex", tuple(point), source_vertices=(key,))


class SnapCommitTests(unittest.TestCase):
    def preview(self, moving, target, kind, all_objects=None):
        snapping = Snapping(enabled=True)
        result = snapping.preview_nearest_anchors(
            moving, all_objects if all_objects is not None else [moving, target]
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, kind)
        return snapping

    def test_vertex_snap_translates_and_preserves_geometry_rotation_and_scale(self):
        moving = make_mesh([vertex("move-v", (1, 0, 0), (1, 0, 0))],
                           position=(3, 2, 0), rotation=(0, 0, 90), scale=(2, 1, 1))
        target = make_mesh([vertex("target-v", (10, 10, 0), (10, 10, 0))])
        before_vertices = moving.vertices.copy()
        before_rotation = moving.rotation.copy()
        before_scale = moving.scale.copy()
        snapping = self.preview(moving, target, "vertex")

        self.assertTrue(snapping.commit_preview(moving, [moving, target]))
        source = moving.snap_anchors[0].world_position(moving)
        destination = target.snap_anchors[0].world_position(target)
        np.testing.assert_allclose(source, destination, atol=1e-8)
        np.testing.assert_array_equal(moving.vertices, before_vertices)
        np.testing.assert_array_equal(moving.rotation, before_rotation)
        np.testing.assert_array_equal(moving.scale, before_scale)
        self.assertIsNone(snapping.snap_preview)
        self.assertEqual(len(snapping.assemblies), 1)
        self.assertEqual(len(snapping.assemblies[0]), 2)

    def test_edge_snap_rotates_deterministically_and_aligns_edge_anchor(self):
        key_a, key_b = (0, 0, 0), (10, 0, 0)
        move_anchors = [
            vertex("ma", (0, 0, 0), key_a),
            vertex("mb", (2, 0, 0), key_b),
            SnapAnchor("move-edge", "edge", (1, 0, 0),
                       source_vertices=(key_a, key_b)),
        ]
        target_anchors = [
            vertex("ta", (0, 0, 0), key_a),
            vertex("tb", (0, 2, 0), key_b),
            SnapAnchor("target-edge", "edge", (0, 1, 0),
                       source_vertices=(key_a, key_b)),
        ]
        moving = make_mesh(move_anchors)
        target = make_mesh(target_anchors, position=(1, -1, 0))
        snapping = self.preview(moving, target, "edge")
        self.assertEqual(snapping.snap_preview.moving_anchor_id, "move-edge")

        self.assertTrue(snapping.commit_preview(moving, [moving, target]))
        move_direction = (moving.snap_anchors[1].world_position(moving)
                          - moving.snap_anchors[0].world_position(moving))
        target_direction = (target.snap_anchors[1].world_position(target)
                            - target.snap_anchors[0].world_position(target))
        move_direction /= np.linalg.norm(move_direction)
        target_direction /= np.linalg.norm(target_direction)
        np.testing.assert_allclose(move_direction, target_direction, atol=1e-8)
        np.testing.assert_allclose(
            moving.snap_anchors[2].world_position(moving),
            target.snap_anchors[2].world_position(target), atol=1e-8,
        )
        self.assertAlmostEqual(float(moving.scale[0]), 1.0)

    def test_face_snap_turns_normals_to_face_each_other(self):
        moving = make_mesh([
            SnapAnchor("move-face", "face", (1 / 3, 1 / 3, 0), (0, 0, 1))
        ])
        target = make_mesh([
            SnapAnchor("target-face", "face", (0, 0, 0), (0, 0, 1))
        ], position=(5, 0, 0))
        snapping = self.preview(moving, target, "face")

        self.assertTrue(snapping.commit_preview(moving, [moving, target]))
        moving_normal = np.asarray(moving.snap_anchors[0].world_normal(moving))
        target_normal = np.asarray(target.snap_anchors[0].world_normal(target))
        np.testing.assert_allclose(moving_normal, -target_normal, atol=1e-8)
        np.testing.assert_allclose(
            moving.snap_anchors[0].world_position(moving),
            target.snap_anchors[0].world_position(target), atol=1e-8,
        )

    def test_transformed_target_and_mover_use_current_world_positions(self):
        moving = make_mesh([vertex("move", (2, 0, 0), (2, 0, 0))],
                           position=(1, -3, 2), rotation=(0, 0, 90), scale=(2, 1, 1))
        target = make_mesh([vertex("target", (0, 2, 0), (0, 2, 0))],
                           position=(8, 4, 1), rotation=(0, 0, -90), scale=(1, 2, 1))
        snapping = self.preview(moving, target, "vertex")
        self.assertTrue(snapping.commit_preview(moving, [moving, target]))
        np.testing.assert_allclose(
            moving.snap_anchors[0].world_position(moving),
            target.snap_anchors[0].world_position(target), atol=1e-8,
        )
        np.testing.assert_array_equal(moving.scale, (2, 1, 1))

    def test_disabled_or_missing_preview_does_nothing(self):
        moving = make_mesh([vertex("move", (0, 0, 0), (0, 0, 0))])
        target = make_mesh([vertex("target", (4, 0, 0), (4, 0, 0))])
        before = moving.position.copy()
        disabled = Snapping(enabled=False)
        self.assertFalse(disabled.commit_preview(moving, [moving, target]))
        self.assertIsNone(disabled.preview_nearest_anchors(moving, [moving, target]))
        self.assertFalse(disabled.commit_preview(moving, [moving, target]))
        np.testing.assert_array_equal(moving.position, before)

        enabled_no_preview = Snapping(enabled=True)
        self.assertFalse(enabled_no_preview.commit_preview(moving, [moving, target]))
        no_target = Snapping(enabled=True)
        self.assertIsNone(no_target.preview_nearest_anchors(moving, [moving]))
        self.assertFalse(no_target.commit_preview(moving, [moving]))
        np.testing.assert_array_equal(moving.position, before)

    def test_commit_is_one_shot_and_repeated_commit_is_stable(self):
        moving = make_mesh([vertex("move", (0, 0, 0), (0, 0, 0))])
        target = make_mesh([vertex("target", (5, 1, 0), (5, 1, 0))])
        snapping = self.preview(moving, target, "vertex")
        self.assertTrue(snapping.commit_preview(moving, [moving, target]))
        committed_position = moving.position.copy()
        self.assertFalse(snapping.commit_preview(moving, [moving, target]))
        np.testing.assert_array_equal(moving.position, committed_position)

        # An assembly cannot snap to one of its own members; repeating the
        # confirmation cannot create duplicate assembly state.
        self.assertIsNone(snapping.preview_nearest_anchors(moving, [moving, target]))
        self.assertFalse(snapping.commit_preview(moving, [moving, target]))
        np.testing.assert_allclose(moving.position, committed_position, atol=1e-8)
        self.assertEqual(len(snapping.assemblies), 1)
        self.assertEqual(len(snapping.assemblies[0]), 2)

    def test_manipulation_release_commits_before_clearing_preview(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8"
        )
        release_body = source.split("def exit_manipulation():", 1)[1].split(
            "\ndef try_convert_stroke():", 1
        )[0]
        commit_at = release_body.index(
            "snapping.commit_preview(manipulation.active_object, renderer.objects)"
        )
        clear_at = release_body.index("snapping.clear_preview()")
        self.assertLess(commit_at, clear_at)


if __name__ == "__main__":
    unittest.main()

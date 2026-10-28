import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from interaction.snapping import Snapping
from render.snap_anchors import SnapAnchor, transform_point
from shapes.candidate_constructors import build_candidate


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

    def _rod_and_sphere(self, rotation=(0, 0, 0)):
        rod = make_mesh([
            vertex("endpoint:positive", (50, 0, 0), (50, 0, 0)),
            vertex("side-vertex", (0, 15, 15), (0, 15, 15)),
        ],
                        rotation=rotation)
        rod.vertices = np.asarray([
            (-50, 0, 0), (50, 0, 0), (-50, 5, 0), (-50, -5, 0),
            (50, 5, 0), (50, -5, 0), (-50, 0, 5), (50, 0, 5),
            (-50, 0, -5), (50, 0, -5),
        ], dtype=float)
        sphere = make_mesh([vertex("sphere-surface", (0, 0, 0), (0, 0, 0))],
                           position=(15, 15, 0))
        sphere.vertices = np.asarray([
            (0, 0, 0), (10, 0, 0), (-10, 0, 0), (0, 10, 0),
            (0, -10, 0), (0, 0, 10), (0, 0, -10),
        ], dtype=float)
        return rod, sphere

    def _assert_rod_meets_target_along_its_axis(self, rod, sphere, original_vertices):
        target_point = sphere.snap_anchors[0].world_position(sphere)
        endpoint = rod.snap_anchors[0].world_position(rod)
        np.testing.assert_allclose(endpoint, target_point, atol=1e-8)
        local_ends = np.asarray([(-50, 0, 0), (50, 0, 0)], dtype=float)
        from render.snap_anchors import transform_point
        world_ends = np.asarray([transform_point(rod, point) for point in local_ends])
        rod_axis = world_ends[1] - world_ends[0]
        center = world_ends.mean(axis=0)
        connection_direction = target_point - center
        rod_axis /= np.linalg.norm(rod_axis)
        connection_direction /= np.linalg.norm(connection_direction)
        self.assertGreater(float(np.dot(rod_axis, connection_direction)), 0.999999)
        np.testing.assert_array_equal(rod.vertices, original_vertices)
        np.testing.assert_allclose(np.linalg.norm(world_ends[1] - world_ends[0]), 100.0)

    def test_rod_to_sphere_vertex_snap_aligns_axis_and_places_endpoint(self):
        rod, sphere = self._rod_and_sphere()
        original_vertices = rod.vertices.copy()
        original_scale = rod.scale.copy()
        snapping = self.preview(rod, sphere, "vertex")
        self.assertEqual(snapping.snap_preview.moving_anchor_id, "endpoint:positive")
        self.assertTrue(snapping.commit_preview(rod, [rod, sphere]))
        self._assert_rod_meets_target_along_its_axis(rod, sphere, original_vertices)
        np.testing.assert_array_equal(rod.scale, original_scale)

    def test_rotated_rod_to_sphere_snap_keeps_length_and_aligns_world_axis(self):
        rod, sphere = self._rod_and_sphere(rotation=(0, 0, 90))
        original_vertices = rod.vertices.copy()
        snapping = self.preview(rod, sphere, "vertex")
        self.assertEqual(snapping.snap_preview.moving_anchor_id, "endpoint:positive")
        self.assertTrue(snapping.commit_preview(rod, [rod, sphere]))
        self._assert_rod_meets_target_along_its_axis(rod, sphere, original_vertices)

    def test_actual_rod_attaches_to_cone_base_not_a_side_face(self):
        rod = build_candidate("sweep.rod", {"length": 200, "radius": 10})
        cone = build_candidate("solid.cone")
        rod.rotation[:] = (0, 0, 90)
        rod.position[:] = (0, -140, 0)
        source_end = next(a for a in rod.snap_anchors
                          if a.anchor_id.startswith("endpoint:")
                          and a.local_position[0] > 0)
        cone_base = next(a for a in cone.snap_anchors
                         if a.anchor_id.startswith("endpoint:")
                         and a.local_position[1] < 0)
        source_before = source_end.world_position(rod)
        target_before = cone_base.world_position(cone)
        center_before = rod.position.copy()
        axis_before = transform_point(rod, (100, 0, 0)) - transform_point(rod, (-100, 0, 0))
        axis_before /= np.linalg.norm(axis_before)
        target_direction_before = target_before - center_before
        target_direction_before /= np.linalg.norm(target_direction_before)
        snapping = Snapping(enabled=True)
        preview = snapping.preview_nearest_anchors(rod, [rod, cone])
        self.assertIsNotNone(preview)
        self.assertEqual(preview.moving_anchor_id, source_end.anchor_id)
        self.assertEqual(preview.target_anchor_id, cone_base.anchor_id)
        self.assertEqual(preview.kind, "vertex")
        self.assertEqual(preview.target_kind, "vertex")
        self.assertLessEqual(preview.distance, Snapping.ANCHOR_PREVIEW_DISTANCE)
        np.testing.assert_allclose(preview.moving_position, source_before, atol=1e-5)
        np.testing.assert_allclose(preview.target_position, target_before, atol=1e-5)

        original_length = float(np.ptp(rod.vertices[:, 0]))
        rotation_before = rod.rotation.copy()
        position_before = rod.position.copy()
        self.assertTrue(snapping.commit_preview(rod, [rod, cone]))
        committed_end = source_end.world_position(rod)
        center_after = rod.position.copy()
        axis_after = transform_point(rod, (100, 0, 0)) - transform_point(rod, (-100, 0, 0))
        axis_after /= np.linalg.norm(axis_after)
        target_direction_after = target_before - center_after
        target_direction_after /= np.linalg.norm(target_direction_after)
        diagnostic = {
            "source_anchor": source_end.anchor_id,
            "target_anchor": cone_base.anchor_id,
            "source_before": source_before.tolist(),
            "target_before": target_before.tolist(),
            "center_before": center_before.tolist(),
            "center_after": center_after.tolist(),
            "rotation_before": rotation_before.tolist(),
            "rotation_after": rod.rotation.tolist(),
            "axis_before_world": axis_before.tolist(),
            "target_direction_before": target_direction_before.tolist(),
            "axis_after_world": axis_after.tolist(),
            "target_direction_after": target_direction_after.tolist(),
        }
        np.testing.assert_allclose(committed_end, target_before, atol=1e-5,
                                   err_msg=repr(diagnostic))
        self.assertGreater(float(np.dot(axis_after, target_direction_after)), 0.999,
                           msg=repr(diagnostic))
        self.assertAlmostEqual(float(np.ptp(rod.vertices[:, 0])), original_length, places=5,
                               msg=repr(diagnostic))
        np.testing.assert_array_equal(rod.scale, (1, 1, 1))

    def test_actual_rod_attaches_to_sphere_surface_without_tilting_or_shrinking(self):
        rod = build_candidate("sweep.rod", {"length": 200, "radius": 10})
        sphere = build_candidate("solid.sphere", {"radius": 50})
        rod.position[:] = (120, 0, 0)
        source_end = next(a for a in rod.snap_anchors
                          if a.anchor_id.startswith("endpoint:")
                          and a.local_position[0] < 0)
        snapping = Snapping(enabled=True)
        preview = snapping.preview_nearest_anchors(rod, [rod, sphere])
        self.assertIsNotNone(preview)
        self.assertEqual(preview.moving_anchor_id, source_end.anchor_id)
        self.assertIn(preview.target_kind, {"vertex", "edge", "face"})
        target_anchor = next(a for a in sphere.snap_anchors
                             if a.kind == preview.target_kind
                             and a.anchor_id == preview.target_anchor_id)
        target_world = target_anchor.world_position(sphere)
        preview_record = {
            "moving_anchor_id": preview.moving_anchor_id,
            "target_anchor_id": preview.target_anchor_id,
            "rod_position_before": rod.position.tolist(),
            "rod_rotation_before": rod.rotation.tolist(),
            "source_world_before": list(preview.moving_position),
            "target_world": target_world.tolist(),
        }
        initial_length = float(np.ptp(rod.vertices[:, 0]))
        self.assertTrue(snapping.commit_preview(rod, [rod, sphere]),
                        msg=repr(preview_record))
        moved_end = source_end.world_position(rod)
        np.testing.assert_allclose(moved_end, target_world, atol=1e-5,
                                   err_msg=repr(preview_record))
        world_ends = np.asarray([
            transform_point(rod, (-100, 0, 0)),
            transform_point(rod, (100, 0, 0)),
        ])
        center = world_ends.mean(axis=0)
        endpoint_direction = moved_end - center
        endpoint_direction /= np.linalg.norm(endpoint_direction)
        target_direction = target_world - center
        target_direction /= np.linalg.norm(target_direction)
        diagnostic = {**preview_record,
                      "rod_position_after": rod.position.tolist(),
                      "rod_rotation_after": rod.rotation.tolist(),
                      "axis_toward_endpoint": endpoint_direction.tolist(),
                      "target_direction_after": target_direction.tolist()}
        self.assertGreater(float(np.dot(endpoint_direction, target_direction)), 0.999,
                           msg=repr(diagnostic))
        self.assertAlmostEqual(float(np.ptp(rod.vertices[:, 0])), initial_length, places=5,
                               msg=repr(diagnostic))

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

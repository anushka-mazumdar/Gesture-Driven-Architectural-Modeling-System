"""Anchor metadata is local-space, deterministic, and passive while snapping is off."""

import unittest
from types import SimpleNamespace

import numpy as np

from interaction.snapping import SnapGroup, Snapping
from render.primitives import PolygonMesh, RibbonMesh
from render.snap_anchors import build_snap_anchors
from shapes.candidate_constructors import build_candidate


VERTICES = np.asarray([
    (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
], dtype=float)
TETRA_FACES = [
    0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2,
]


class SnapAnchorTests(unittest.TestCase):
    def test_generates_stable_center_vertex_edge_and_face_anchors(self):
        anchors = build_snap_anchors(VERTICES, TETRA_FACES)
        repeated = build_snap_anchors(VERTICES.copy(), list(TETRA_FACES))
        kinds = [anchor.kind for anchor in anchors]
        self.assertEqual(kinds.count("center"), 1)
        self.assertEqual(kinds.count("vertex"), 4)
        self.assertEqual(kinds.count("edge"), 6)
        self.assertEqual(kinds.count("face"), 4)
        self.assertEqual([anchor.anchor_id for anchor in anchors],
                         [anchor.anchor_id for anchor in repeated])

    def test_anchor_positions_and_face_normals_are_geometric(self):
        anchors = build_snap_anchors(VERTICES, TETRA_FACES)
        center = next(anchor for anchor in anchors if anchor.kind == "center")
        origin = next(anchor for anchor in anchors
                      if anchor.kind == "vertex" and anchor.local_position == (0, 0, 0))
        x_edge = next(anchor for anchor in anchors
                      if anchor.kind == "edge" and np.allclose(anchor.local_position, (1, 0, 0)))
        xy_face = next(anchor for anchor in anchors
                       if anchor.kind == "face"
                       and np.allclose(anchor.local_position, (2 / 3, 2 / 3, 0)))
        np.testing.assert_allclose(center.local_position, (1, 1, 1))
        np.testing.assert_allclose(origin.local_position, (0, 0, 0))
        np.testing.assert_allclose(x_edge.local_position, (1, 0, 0))
        np.testing.assert_allclose(xy_face.local_normal, (0, 0, 1))

    def test_mesh_objects_expose_anchor_data(self):
        polygon = PolygonMesh([(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)])
        ribbon = RibbonMesh([(-10, 0, 0), (0, 10, 0), (10, 0, 0)])
        candidate = build_candidate("solid.cube", {"size": 20})
        for mesh in (polygon, ribbon, candidate):
            with self.subTest(mesh=type(mesh).__name__):
                self.assertTrue(mesh.snap_anchors)
                self.assertTrue(any(anchor.kind == "vertex" for anchor in mesh.snap_anchors))
                self.assertTrue(any(anchor.kind == "edge" for anchor in mesh.snap_anchors))
                self.assertTrue(any(anchor.kind == "face" for anchor in mesh.snap_anchors))

    def test_world_anchor_positions_follow_scale_rotation_and_translation(self):
        anchor = next(anchor for anchor in build_snap_anchors(VERTICES, TETRA_FACES)
                      if anchor.kind == "vertex" and anchor.local_position == (2, 0, 0))
        obj = SimpleNamespace(
            position=np.asarray((10, 20, 30), dtype=float),
            rotation=np.asarray((0, 0, 90), dtype=float),
            scale=np.asarray((0.5, 1, 1), dtype=float),
        )
        np.testing.assert_allclose(anchor.world_position(obj), (10, 21, 30), atol=1e-7)
        obj.position[:] = (-4, 5, 6)
        np.testing.assert_allclose(anchor.world_position(obj), (-4, 6, 6), atol=1e-7)

        face = next(anchor for anchor in build_snap_anchors(VERTICES, TETRA_FACES)
                    if anchor.kind == "face"
                    and np.allclose(anchor.local_normal, (0, 0, 1)))
        np.testing.assert_allclose(face.world_normal(obj), (0, 0, 1), atol=1e-7)

    def test_snapping_stays_disabled_until_explicitly_enabled(self):
        moving = PolygonMesh([(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)])
        target = PolygonMesh([(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)])
        target.position[0] = 25

        disabled = Snapping(snap_distance=100)
        self.assertFalse(disabled.enabled)
        self.assertIsNone(disabled.update(moving, [moving, target]))
        self.assertFalse(disabled.has_candidate())
        self.assertFalse(target.highlighted)

        group = SnapGroup(moving)
        group.add(target, target.position - moving.position)
        disabled._groups.append(group)
        before = target.position.copy()
        disabled.propagate_move(moving, 10, 5)
        np.testing.assert_array_equal(target.position, before)

        enabled = Snapping(snap_distance=100, enabled=True)
        self.assertIs(enabled.update(moving, [moving, target]), target)
        self.assertTrue(enabled.has_candidate())


if __name__ == "__main__":
    unittest.main()

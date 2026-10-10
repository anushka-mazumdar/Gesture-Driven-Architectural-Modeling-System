import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from interaction.snapping import SnapAssembly, Snapping
from render.snap_anchors import SnapAnchor


def vertex(anchor_id, point, key):
    return SnapAnchor(anchor_id, "vertex", tuple(point), source_vertices=(key,))


def make_object(anchor_id, anchor_point=(0, 0, 0), position=(0, 0, 0),
                rotation=(0, 0, 0), scale=(1, 1, 1)):
    key = tuple(anchor_point)
    return SimpleNamespace(
        snap_anchors=(vertex(anchor_id, anchor_point, key),),
        position=np.asarray(position, dtype=float),
        rotation=np.asarray(rotation, dtype=float),
        scale=np.asarray(scale, dtype=float),
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        highlighted=False,
        selected=False,
    )


def snap_into(snapping, moving, target, objects):
    preview = snapping.preview_nearest_anchors(moving, objects)
    if preview is None:
        raise AssertionError("expected a valid vertex snap preview")
    if not snapping.commit_preview(moving, objects):
        raise AssertionError("expected the snap preview to commit")


class SnapAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.snapping = Snapping(enabled=True)
        self.first = make_object("a", position=(0, 0, 0))
        self.second = make_object("b", anchor_point=(-10, 0, 0),
                                  position=(20, 0, 0))
        self.objects = [self.first, self.second]

    def test_committed_snap_creates_assembly_and_keeps_scene_objects_individual(self):
        snap_into(self.snapping, self.second, self.first, self.objects)

        self.assertEqual(len(self.snapping.assemblies), 1)
        assembly = self.snapping.assemblies[0]
        self.assertIsInstance(assembly, SnapAssembly)
        self.assertEqual(assembly.members, [self.first, self.second])
        self.assertIs(self.snapping.get_group(self.first), assembly)
        self.assertIs(self.snapping.get_group(self.second), assembly)
        self.assertEqual(self.objects, [self.first, self.second])

    def test_snap_to_existing_assembly_merges_multiple_connected_objects(self):
        third = make_object("c", anchor_point=(-10, 0, 0), position=(22, 0, 0))
        self.objects.append(third)
        snap_into(self.snapping, self.second, self.first, self.objects)
        snap_into(self.snapping, third, self.second, self.objects)

        self.assertEqual(len(self.snapping.assemblies), 1)
        self.assertEqual({id(member) for member in self.snapping.assemblies[0].members},
                         {id(self.first), id(self.second), id(third)})

    def test_snap_merges_two_preexisting_assemblies(self):
        third = make_object("c", anchor_point=(-10, 0, 0), position=(42, 0, 0))
        fourth = make_object("d", anchor_point=(-10, 0, 0), position=(62, 0, 0))
        self.objects.extend((third, fourth))
        snap_into(self.snapping, self.second, self.first, self.objects)
        snap_into(self.snapping, fourth, third, self.objects)
        self.assertEqual(len(self.snapping.assemblies), 2)

        snap_into(self.snapping, third, self.second, self.objects)

        self.assertEqual(len(self.snapping.assemblies), 1)
        self.assertEqual({id(member) for member in self.snapping.assemblies[0].members},
                         {id(self.first), id(self.second), id(third), id(fourth)})

    def test_movement_preserves_member_offsets(self):
        snap_into(self.snapping, self.second, self.first, self.objects)
        before = self.second.position - self.first.position

        self.first.position += np.array([14.0, -5.0, 3.0])
        self.snapping.propagate_move(self.first, 14.0, -5.0)
        self.snapping.propagate_depth(self.first, 3.0)

        np.testing.assert_allclose(self.second.position - self.first.position,
                                   before, atol=1e-8)

    def test_existing_assembly_stays_together_when_new_snapping_is_disabled(self):
        snap_into(self.snapping, self.second, self.first, self.objects)
        before = self.second.position - self.first.position
        self.snapping.enabled = False

        self.first.position += np.array([8.0, 3.0, 0.0])
        self.snapping.propagate_move(self.first, 8.0, 3.0)

        np.testing.assert_allclose(self.second.position - self.first.position,
                                   before, atol=1e-8)
        self.assertIsNone(self.snapping.preview_nearest_anchors(
            self.first, self.objects
        ))

    def test_rotation_moves_members_about_active_object_and_keeps_relative_rotation(self):
        self.second.position = np.array([20.0, 0.0, 0.0])
        snap_into(self.snapping, self.second, self.first, self.objects)
        assembly = self.snapping.assemblies[0]
        before_offset = self.second.position - self.first.position
        before_rotation = self.snapping._rotation_matrix(self.second.rotation)
        delta = np.array([0.0, 0.0, 90.0])

        self.first.rotation += delta
        self.snapping.propagate_rotate(self.first, 0.0, 0.0)

        np.testing.assert_allclose(
            self.second.position - self.first.position,
            self.snapping._rotation_matrix(delta) @ before_offset,
            atol=1e-8,
        )
        after_relative = (self.snapping._rotation_matrix(self.first.rotation).T
                          @ self.snapping._rotation_matrix(self.second.rotation))
        expected_relative = (self.snapping._rotation_matrix(np.zeros(3)).T
                             @ before_rotation)
        np.testing.assert_allclose(after_relative, expected_relative, atol=1e-8)

    def test_scaling_preserves_relative_layout_and_member_scale_ratio(self):
        self.second.position = np.array([20.0, 0.0, 0.0])
        self.second.scale = np.array([2.0, 2.0, 2.0])
        snap_into(self.snapping, self.second, self.first, self.objects)
        before_offset = self.second.position - self.first.position
        before_relative_scale = self.second.scale / self.first.scale

        self.first.scale[:] = 1.5
        self.snapping.propagate_scale(self.first, 1.5)

        np.testing.assert_allclose(self.second.position - self.first.position,
                                   before_offset * 1.5, atol=1e-8)
        np.testing.assert_allclose(self.second.scale / self.first.scale,
                                   before_relative_scale, atol=1e-8)

    def test_per_frame_transforms_do_not_resnapshot_the_whole_assembly(self):
        snap_into(self.snapping, self.second, self.first, self.objects)
        assembly = self.snapping.assemblies[0]
        with patch.object(assembly, "remember_transforms",
                          wraps=assembly.remember_transforms) as snapshot:
            for _ in range(100):
                self.first.position[0] += 0.1
                self.snapping.propagate_move(self.first, 0.1, 0.0)
                self.first.rotation[2] += 0.2
                self.snapping.propagate_rotate(self.first, 0.0, 0.2)
                self.first.scale[:] *= 1.001
                self.snapping.propagate_scale(self.first, self.first.scale[0])

        self.assertEqual(snapshot.call_count, 0)
        self.assertEqual(len(assembly._last_states), len(assembly.members))


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from types import SimpleNamespace

import numpy as np

from interaction.snapping import SnapAssembly, Snapping
from render.threejs_renderer import ThreeJSRenderer


class RecordingWindow:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, source):
        self.calls.append(source)


class SceneMesh:
    kind = "mesh"

    def __init__(self, index):
        self.vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.normals = np.asarray([[0, 0, 1]] * 3, dtype=np.float32)
        self.indices = [0, 1, 2]
        self.color = [0.3, 0.6, 0.9]
        self.position = np.asarray([float(index), 0.0, 0.0])
        self.rotation = np.zeros(3)
        self.scale = np.ones(3)


def frame_payload(call):
    marker = "window.receiveFrameFromPython("
    return json.loads(call[call.index(marker) + len(marker):-1])


class SnapAssemblyPerformanceTests(unittest.TestCase):
    def test_moving_and_rotating_assembly_reuses_geometry_and_sends_one_frame_call(self):
        objects = [SceneMesh(index) for index in range(24)]
        renderer = ThreeJSRenderer()
        window = RecordingWindow()
        renderer._window = window
        renderer._ready = True
        for obj in objects:
            renderer.add_object(obj)

        assembly = SnapAssembly(objects[0])
        for obj in objects[1:]:
            assembly.add(obj, obj.position - objects[0].position)
        snapping = Snapping(enabled=True)
        snapping._groups.append(assembly)
        initial_offsets = {
            id(obj): obj.position.copy() - objects[0].position.copy()
            for obj in objects[1:]
        }
        renderer._sync_scene()

        for _ in range(60):
            objects[0].position[0] += 0.5
            snapping.propagate_move(objects[0], 0.5, 0.0)
            objects[0].rotation[2] += 0.5
            snapping.propagate_rotate(objects[0], 0.0, 0.5)
            renderer._sync_scene()

        expected_rotation = snapping._rotation_matrix((0.0, 0.0, 30.0))
        for obj in objects[1:]:
            expected = expected_rotation @ initial_offsets[id(obj)]
            np.testing.assert_allclose(obj.position - objects[0].position,
                                       expected, atol=1e-8)

        self.assertEqual(renderer.performance_snapshot()["mesh_geometry_payloads"],
                         len(objects))
        self.assertEqual(len(window.calls), 61)
        for call in window.calls[1:]:
            payload = frame_payload(call)
            self.assertEqual(payload["meshes"]["added"], [])
            self.assertEqual(len(payload["meshes"]["transforms"]), len(objects))

    def test_sphere_rod_assembly_repeated_rotation_matches_rendered_transforms(self):
        sphere, rod = SceneMesh(0), SceneMesh(1)
        sphere.position[:] = (12.0, -4.0, 3.0)
        rod.position[:] = (42.0, -4.0, 3.0)
        rod.rotation[:] = (0.0, 0.0, 0.0)
        renderer = ThreeJSRenderer()
        window = RecordingWindow()
        renderer._window = window
        renderer._ready = True
        renderer.add_object(sphere)
        renderer.add_object(rod)
        renderer._sync_scene()

        assembly = SnapAssembly(sphere)
        assembly.add(rod, rod.position - sphere.position)
        snapping = Snapping(enabled=True)
        snapping._groups.append(assembly)
        initial_delta = rod.position.copy() - sphere.position.copy()
        initial_relative_rotation = (
            snapping._rotation_matrix(sphere.rotation).T
            @ snapping._rotation_matrix(rod.rotation)
        )

        for angle in (17.0, 49.0, -23.0, 91.0, 138.0):
            sphere.position += np.asarray((1.25, -0.5, 0.75))
            snapping.propagate_move(sphere, 1.25, -0.5)
            snapping.propagate_depth(sphere, 0.75)
            sphere.rotation[:] = (angle * 0.2, angle * -0.15, angle)
            snapping.propagate_rotate(sphere, 0.0, 0.0)
            renderer._sync_scene()

            rotation = snapping._rotation_matrix(sphere.rotation)
            expected_position = sphere.position + rotation @ initial_delta
            expected_rotation = rotation @ initial_relative_rotation
            np.testing.assert_allclose(rod.position, expected_position, atol=2e-6)
            np.testing.assert_allclose(
                snapping._rotation_matrix(rod.rotation), expected_rotation,
                atol=2e-6,
            )
            payload = frame_payload(window.calls[-1])["meshes"]["transforms"]
            by_id = {item["id"]: item for item in payload}
            sphere_id = renderer._bridge_ids[id(sphere)]
            rod_id = renderer._bridge_ids[id(rod)]
            np.testing.assert_allclose(by_id[sphere_id]["position"], sphere.position, atol=1e-7)
            np.testing.assert_allclose(by_id[rod_id]["position"], expected_position, atol=2e-6)
            np.testing.assert_allclose(by_id[rod_id]["rotation"], rod.rotation, atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.norm(rod.position - sphere.position)),
                                   float(np.linalg.norm(initial_delta)), places=5)


if __name__ == "__main__":
    unittest.main()

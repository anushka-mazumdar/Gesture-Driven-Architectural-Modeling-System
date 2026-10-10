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


if __name__ == "__main__":
    unittest.main()

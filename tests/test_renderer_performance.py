import json
import time
import unittest
from pathlib import Path

import numpy as np

from render.threejs_renderer import ThreeJSRenderer
from shapes.stroke_capture import StrokeCapture


class SlowWindow:
    def __init__(self, delay=0.08):
        self.delay = delay
        self.calls = []

    def evaluate_js(self, source):
        time.sleep(self.delay)
        self.calls.append(source)


def bridge_payload(source):
    marker = "window.receiveFrameFromPython("
    start = source.index(marker) + len(marker)
    return json.loads(source[start:-1])


class RendererPerformanceTests(unittest.TestCase):
    def test_render_requests_do_not_wait_for_slow_webview_and_coalesce(self):
        renderer = ThreeJSRenderer()
        self.addCleanup(renderer.close_bridge)
        window = SlowWindow()
        renderer._window = window
        renderer._ready = True
        mesh = type("Mesh", (), {})()
        mesh.vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        mesh.normals = np.asarray([[0, 0, 1]] * 3, dtype=float)
        mesh.indices = [0, 1, 2]
        mesh.color = [1.0, 0.0, 0.0]
        mesh.position = np.asarray([0.0, 0.0, 0.0])
        mesh.rotation = np.asarray([0.0, 0.0, 0.0])
        mesh.scale = np.asarray([1.0, 1.0, 1.0])
        mesh.kind = "mesh"
        renderer.add_object(mesh)
        capture = StrokeCapture(min_dist=0, min_points=2, smoothing=0.0)
        capture.start_stroke()

        started = time.perf_counter()
        for index in range(40):
            capture.add_point((index, index * 2))
            renderer.set_drawing_state(points=capture.get_current(),
                                       drawing_enabled=True)
            renderer.render()
        request_seconds = time.perf_counter() - started

        self.assertLess(request_seconds, 0.05)
        self.assertEqual(capture.get_current(),
                         [(index, index * 2) for index in range(40)])
        self.assertTrue(renderer.wait_for_sync(timeout=2.0))
        stats = renderer.performance_snapshot()
        self.assertEqual(stats["requests"], 40)
        self.assertGreater(stats["coalesced"], 0)
        self.assertLess(stats["javascript_calls"], stats["requests"])
        self.assertEqual(stats["mesh_geometry_payloads"], 1)
        payloads = [bridge_payload(call) for call in window.calls]
        self.assertEqual(sum(len(payload["meshes"]["added"]) for payload in payloads), 1)

        # Existing object transforms are still delivered without retransmitting
        # the vertex/index arrays.
        mesh.position[0] = 12.0
        renderer.render()
        self.assertTrue(renderer.wait_for_sync(timeout=2.0))
        final_frame = bridge_payload(window.calls[-1])
        self.assertEqual(final_frame["meshes"]["added"], [])
        self.assertEqual(final_frame["meshes"]["transforms"][0]["position"], [12.0, 0.0, 0.0])

    def test_drawing_preview_updates_existing_raw_point_buffer(self):
        source = Path("webview_app/web/main.js").read_text(encoding="utf-8")
        self.assertIn("drawingStrokeHalfWidth = 1.35", source)
        self.assertIn("drawingLineGeometry.setDrawRange(0, Math.max(0, points.length - 1) * 6)", source)
        self.assertIn("const centerX = points[index][0] - 320", source)
        self.assertIn("const centerY = 240 - points[index][1]", source)
        self.assertNotIn("new THREE.TubeGeometry(\n      curve", source)


if __name__ == "__main__":
    unittest.main()

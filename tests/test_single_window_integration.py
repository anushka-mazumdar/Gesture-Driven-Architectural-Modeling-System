import unittest
from pathlib import Path
from unittest.mock import patch

from gestures.candidate_swipe_gate import CandidateSwipeGate
from interaction.manipulation import Manipulation
from interaction.recommendation_mode import MODE_NORMAL, RecommendationInteractionMode
from render.threejs_renderer import ThreeJSRenderer
from shapes.shape_3d_factory import Shape3DFactory
from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_capture import StrokeCapture
from shapes.stroke_pipeline import StrokePipeline
from webview_app.application_window import create_application_window


class Event:
    def __init__(self):
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


class FakeWindow:
    def __init__(self):
        self.events = type("Events", (), {"loaded": Event(), "closing": Event()})()
        self.js_calls = []

    def evaluate_js(self, source):
        self.js_calls.append(source)


class FakeWebview:
    def __init__(self):
        self.windows = []

    def create_window(self, *args, **kwargs):
        window = FakeWindow()
        self.windows.append((window, args, kwargs))
        return window


class SingleWindowIntegrationTests(unittest.TestCase):
    def test_application_starts_one_webview_for_drawing_and_scene(self):
        webview = FakeWebview()
        renderer = ThreeJSRenderer()
        stopped = []
        window = create_application_window(webview, object(), renderer,
                                           lambda: stopped.append(True))

        self.assertEqual(len(webview.windows), 1)
        self.assertIs(renderer._window, window)
        self.assertEqual(len(window.events.closing.callbacks), 1)
        self.assertEqual(webview.windows[0][2]["width"], 1280)
        self.assertEqual(webview.windows[0][2]["height"], 720)
        markup = Path("webview_app/web/index.html").read_text(encoding="utf-8")
        self.assertIn('id="application-viewport"', markup)
        viewport = markup.split('id="application-viewport"', 1)[1].split("</main>", 1)[0]
        self.assertIn('id="scene-container"', viewport)
        self.assertIn('id="candidate-panel"', viewport)
        self.assertNotIn('id="application-workspace"', markup)
        self.assertNotIn('id="drawing-frame"', markup)
        main_source = Path("main.py").read_text(encoding="utf-8")
        self.assertNotIn("SketchPanel", main_source)
        self.assertNotIn("cv2.imshow", main_source)
        self.assertNotIn("create_window(", main_source)
        page_script = Path("webview_app/web/main.js").read_text(encoding="utf-8")
        self.assertIn("receiveDrawingStateFromPython", page_script)
        self.assertIn("TubeGeometry", page_script)
        self.assertIn("DynamicDrawUsage", page_script)
        self.assertIn("receiveFrameFromPython", page_script)
        self.assertNotIn("getContext('2d')", page_script)

    def test_draw_recommend_swipe_confirm_then_resume_scene_interaction(self):
        webview = FakeWebview()
        renderer = ThreeJSRenderer()
        window = create_application_window(webview, object(), renderer, lambda: None)
        renderer._ready = True

        # A tracked-hand stroke is captured in Python coordinates and drawn
        # directly by the WebView's Three.js overlay.
        capture = StrokeCapture(min_dist=1, min_points=5, smoothing=0.0)
        capture.start_stroke()
        points = [(150, 130), (300, 130), (300, 280), (150, 280), (152, 132)]
        for point in points:
            capture.add_point(point)
        stroke = capture.finish()
        self.assertEqual(len(stroke.raw_points), len(points))
        renderer.set_drawing_state(points=capture.get_current(),
                                   drawing_enabled=True, cursor=(0.25, 0.3))
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())

        # Run the captured path through the project's actual classifier and
        # mesh factory, observing that the classifier receives the stroke.
        classifier = StrokeClassifier(use_ml_model=False)
        original_classify = classifier.classify
        with patch.object(classifier, "classify", wraps=original_classify) as classify:
            result = StrokePipeline(
                classifier=classifier,
                factory=Shape3DFactory(panel_width=640, panel_height=480),
            ).convert(stroke)
        classify.assert_called_once()
        self.assertIsNotNone(result.mesh)
        renderer.add_object(result.mesh)

        self.assertEqual(result.candidates.status, "ok")
        renderer.set_candidate_recommendation(result.candidates)
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())
        mode = RecommendationInteractionMode(
            renderer, CandidateSwipeGate(settle_seconds=0.1, cooldown_seconds=0.5)
        )
        self.assertTrue(mode.update(None, now=1.0))  # recommendation owns input
        self.assertTrue(mode.update("SWIPE_LEFT", now=1.1))
        self.assertTrue(mode.update(None, now=1.21))
        self.assertEqual(renderer._candidate_index, 1)
        selected_id = renderer._candidate_panel["candidates"][1]["id"]

        self.assertTrue(renderer.confirm_candidate(selected_id))
        self.assertFalse(renderer.has_candidate_panel)
        self.assertEqual(renderer.selected_candidate["id"], selected_id)
        self.assertTrue(mode.update("FIST", now=2.0))  # consume close frame
        self.assertEqual(mode.mode, MODE_NORMAL)
        self.assertFalse(mode.update(None, now=2.1))

        # Normal scene manipulation remains usable, and both the final 2D
        # drawing frame and the Three.js mesh travel through that same window.
        mesh = result.mesh
        manipulation = Manipulation()
        manipulation.set_object(mesh)
        manipulation.update_move((320, 240), 640, 480)
        manipulation.update_move((350, 260), 640, 480)
        self.assertNotEqual(mesh.position[:2].tolist(), [0.0, 0.0])
        renderer.set_drawing_state(points=[], drawing_enabled=True,
                                   cursor=(0.5, 0.5))
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())

        self.assertIs(renderer._window, window)
        joined_calls = "\n".join(window.js_calls)
        self.assertIn("receiveFrameFromPython", joined_calls)
        self.assertIn('"points": [[150.0, 130.0]', joined_calls)
        self.assertIn('"meshes": {"added": [{', joined_calls)
        self.assertIn('"candidate_panel": {"status": "ok"', joined_calls)


if __name__ == "__main__":
    unittest.main()

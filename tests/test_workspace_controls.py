import unittest
from pathlib import Path

from interaction.snapping import Snapping
from interaction.workspace_settings import WorkspaceSettings
from render.threejs_renderer import ThreeJSRenderer


ROOT = Path(__file__).resolve().parents[1]


class FakeWindow:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, source):
        self.calls.append(source)


class WorkspaceControlsTests(unittest.TestCase):
    def test_snapping_toggle_updates_python_engine_and_ui_state(self):
        snapping = Snapping(enabled=False)
        renderer = ThreeJSRenderer()
        settings = WorkspaceSettings(snapping, renderer)

        self.assertTrue(settings.set_snapping_enabled(True))
        self.assertTrue(snapping.enabled)
        self.assertTrue(renderer._workspace_settings["snapping_enabled"])

        self.assertFalse(settings.set_snapping_enabled(False))
        self.assertFalse(snapping.enabled)
        self.assertFalse(renderer._workspace_settings["snapping_enabled"])

    def test_color_palette_updates_python_stroke_color(self):
        renderer = ThreeJSRenderer()
        settings = WorkspaceSettings(Snapping(), renderer)
        red = settings.set_stroke_color("#ef5350")
        self.assertEqual(red, (239 / 255.0, 83 / 255.0, 80 / 255.0))
        self.assertEqual(renderer._workspace_settings["stroke_color"], list(red))
        self.assertIsNone(settings.set_stroke_color("red"))

    def test_renderer_sends_python_owned_settings_to_webview(self):
        renderer = ThreeJSRenderer()
        renderer._window = FakeWindow()
        renderer._ready = True
        settings = WorkspaceSettings(Snapping(), renderer)
        settings.set_snapping_enabled(True)
        renderer.render()
        self.assertTrue(renderer.wait_for_sync())
        self.assertTrue(any(
            "receiveFrameFromPython" in call
            and '"workspace_settings": {"snapping_enabled": true' in call
            and '"snapping_enabled": true' in call
            for call in renderer._window.calls
        ))

    def test_controls_and_snapping_bridge_live_in_the_single_workspace(self):
        markup = (ROOT / "webview_app" / "web" / "index.html").read_text(encoding="utf-8")
        viewport = markup.split('id="application-viewport"', 1)[1].split("</main>", 1)[0]
        for control_id in (
            "workspace-panel", "workspace-search-input", "draw-mode-button",
            "touch-mode-button", "mesh-view-toggle", "snapping-toggle",
        ):
            with self.subTest(control=control_id):
                self.assertIn(f'id="{control_id}"', viewport)
        self.assertIn('aria-label="New shape color"', markup)
        self.assertIn('data-color="#ef5350"', markup)
        self.assertIn('data-color="#f4d35e"', markup)
        self.assertIn('aria-label="Detected shape candidates"', viewport)

        script = (ROOT / "webview_app" / "web" / "main.js").read_text(encoding="utf-8")
        self.assertIn("api?.set_snapping_enabled", script)
        self.assertIn("receiveWorkspaceSettingsFromPython", script)
        self.assertIn("api?.set_stroke_color", script)
        self.assertIn("new THREE.Color(...state.stroke_color)", script)
        self.assertIn("is-recommendation-active", script)
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("workspace_settings.set_snapping_enabled(enabled)", main_source)


if __name__ == "__main__":
    unittest.main()

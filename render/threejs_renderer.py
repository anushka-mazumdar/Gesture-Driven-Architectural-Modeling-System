"""
The sole 3D renderer backend: a Three.js scene in a pywebview window.

ThreeJSRenderer exposes the object-management surface main.py drives —
`objects`, add_object(), remove_object(), clear_objects(),
render(display) — and pushes the same mesh objects (vertices/normals/
indices/color/position/rotation/scale convention — see
render/mesh_bridge.py) to the Three.js scene over the Python -> JS
bridge. Each frame it resends a full snapshot of the scene, so in-place
transform edits from interaction/manipulation.py reach the live WebView.
"""

import json

from render.mesh_bridge import serialize_mesh, serialize_hand


class ThreeJSRenderer:

    def __init__(self, width=640, height=480):

        self.width  = width
        self.height = height
        self.objects = []

        self._window = None
        self._ready  = False
        self._hud_text = None

    # ─────────────────────────────
    def attach_window(self, window):
        """Attach the pywebview Window hosting the Three.js scene.

        Syncing is deferred until the window reports `loaded`, so the
        first push doesn't race the page's own script setup.
        """
        self._window = window
        window.events.loaded += self._on_loaded

    def _on_loaded(self, window):
        self._ready = True

    # ─────────────────────────────
    def add_object(self, obj):
        self.objects.append(obj)

    def remove_object(self, obj):
        if obj in self.objects:
            self.objects.remove(obj)

    def clear_objects(self):
        self.objects.clear()

    def set_hud_text(self, text):
        """Small top-left overlay text in the Three.js window (e.g. the
        currently selected shape's classification). None/'' hides it."""
        self._hud_text = text or None

    # ─────────────────────────────
    def render(self, display):
        """Push the current scene state to Three.js, if the window is
        attached and ready. Runs every call (not just on add/remove) so
        in-place transform edits from interaction/manipulation.py —
        which mutate an existing object's position/rotation/scale
        without re-adding it — still reach the live WebView scene.
        The 3D scene lives in its own WebView window, not composited
        onto `display`, so the 2D camera/sketch frame passes through
        unchanged.
        """
        if self._ready and self._window is not None:
            self._sync_scene()
        return display

    def _sync_scene(self):
        hand_obj = None
        meshes = []
        for obj in self.objects:
            if getattr(obj, 'kind', None) == 'hand':
                hand_obj = obj
            elif len(getattr(obj, 'vertices', [])) > 0:
                meshes.append(serialize_mesh(obj))

        payload = json.dumps(meshes)
        self._window.evaluate_js(
            f"window.receiveMeshesFromPython && window.receiveMeshesFromPython({payload})"
        )

        hand_payload = json.dumps(serialize_hand(hand_obj) if hand_obj is not None else None)
        self._window.evaluate_js(
            f"window.receiveHandFromPython && window.receiveHandFromPython({hand_payload})"
        )

        hud_payload = json.dumps(self._hud_text)
        self._window.evaluate_js(
            f"window.receiveHudFromPython && window.receiveHudFromPython({hud_payload})"
        )

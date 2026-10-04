"""
The sole 3D renderer backend: a Three.js scene in a pywebview window.

ThreeJSRenderer exposes the object-management surface main.py drives —
    `objects`, add_object(), remove_object(), clear_objects(),
    render() — and pushes the same mesh objects (vertices/normals/
indices/color/position/rotation/scale convention — see
render/mesh_bridge.py) to the Three.js scene over the Python -> JS
bridge. Each frame it resends a full snapshot of the scene, so in-place
transform edits from interaction/manipulation.py reach the live WebView.
"""

import json
import math

from render.mesh_bridge import serialize_mesh, serialize_hand


class ThreeJSRenderer:

    def __init__(self, width=640, height=480):

        self.width  = width
        self.height = height
        self.objects = []

        self._window = None
        self._ready  = False
        self._hud_text = None
        self._snap_preview = None
        self._candidate_panel = None
        self._candidate_index = 0
        self._candidate_search_query = ""
        self._candidate_generation = 0
        self._candidate_hand_state = None
        self.selected_candidate = None
        self._drawing_state = {
            "points": [], "drawing_enabled": False, "cursor": None,
            "near_object": False, "delete_progress": 0.0,
        }

    @property
    def has_candidate_panel(self):
        return self._candidate_panel is not None

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

    def replace_object(self, old_obj, new_obj):
        """Replace a scene object in place so its order stays stable."""
        try:
            index = self.objects.index(old_obj)
        except ValueError:
            return False
        self.objects[index] = new_obj
        return True

    def clear_objects(self):
        self.objects.clear()

    def set_hud_text(self, text):
        """Small top-left overlay text in the Three.js window (e.g. the
        currently selected shape's classification). None/'' hides it."""
        self._hud_text = text or None

    def set_snap_preview(self, preview):
        """Set a display-only snap proposal, or clear it with None."""
        if preview is None:
            self._snap_preview = None
            return
        payload = preview.to_dict() if hasattr(preview, "to_dict") else preview
        if not isinstance(payload, dict):
            self._snap_preview = None
            return
        try:
            moving = [float(value) for value in payload["moving_position"]]
            target = [float(value) for value in payload["target_position"]]
            distance = float(payload["distance"])
            if (len(moving) != 3 or len(target) != 3
                    or not all(math.isfinite(value) for value in moving + target)
                    or not math.isfinite(distance)):
                raise ValueError
            self._snap_preview = {
                "moving_anchor_id": str(payload["moving_anchor_id"]),
                "target_anchor_id": str(payload["target_anchor_id"]),
                "kind": str(payload["kind"]),
                "moving_position": moving,
                "target_position": target,
                "distance": distance,
            }
        except (KeyError, TypeError, ValueError):
            self._snap_preview = None

    def set_candidate_recommendation(self, recommendation):
        """Set the display-only taxonomy recommendation shown in the WebView.

        Only a valid, non-empty shared RecommendationResult is sent. Uncertain,
        unsupported, malformed, and empty results clear the panel state.
        """
        self._candidate_generation += 1
        payload = None
        if hasattr(recommendation, "to_dict"):
            try:
                payload = recommendation.to_dict()
            except Exception:
                payload = None
        elif isinstance(recommendation, dict):
            payload = recommendation

        if (not isinstance(payload, dict)
                or payload.get("status") != "ok"
                or not isinstance(payload.get("display_label"), str)
                or not payload["display_label"].strip()
                or not isinstance(payload.get("candidates"), list)
                or not payload["candidates"]):
            self._candidate_panel = None
            self._candidate_index = 0
            self._candidate_search_query = ""
            self._candidate_hand_state = None
            self.selected_candidate = None
            return

        candidates = payload["candidates"]
        if any(not isinstance(candidate, dict)
               or not isinstance(candidate.get("id"), str)
               or not candidate["id"].strip()
               or not isinstance(candidate.get("label"), str)
               or not candidate["label"].strip()
               for candidate in candidates):
            self._candidate_panel = None
            self._candidate_index = 0
            self._candidate_search_query = ""
            self._candidate_hand_state = None
            self.selected_candidate = None
            return
        # Send only the fields the panel renders; keep taxonomy ordering and
        # IDs exactly as supplied by the shared Recommendation API.
        self._candidate_panel = {
            "status": "ok",
            "display_label": payload["display_label"],
            "candidates": [
                {"id": candidate["id"], "label": candidate["label"]}
                for candidate in candidates
            ],
        }
        self._candidate_index = 0
        self._candidate_search_query = ""
        self._candidate_hand_state = None
        self.selected_candidate = None

    @property
    def candidate_generation(self):
        """Monotonic identity for the currently active recommendation flow."""
        return self._candidate_generation

    def set_candidate_search_query(self, query):
        """Filter the active API candidate roster by its display name.

        The stored recommendation remains untouched; search only changes the
        visible subset and its selection index, preserving API order and IDs.
        """
        if not self.has_candidate_panel or not isinstance(query, str):
            return False
        next_query = query.strip()
        if next_query == self._candidate_search_query:
            return True

        visible_before = self._visible_candidates()
        selected_id = (visible_before[self._candidate_index]["id"]
                       if visible_before and self._candidate_index < len(visible_before)
                       else None)
        self._candidate_search_query = next_query
        visible_after = self._visible_candidates()
        if selected_id and any(item["id"] == selected_id for item in visible_after):
            self._candidate_index = next(
                index for index, item in enumerate(visible_after)
                if item["id"] == selected_id
            )
        else:
            self._candidate_index = 0
        return True

    def _visible_candidates(self):
        if not self._candidate_panel:
            return []
        query = self._candidate_search_query.casefold()
        candidates = self._candidate_panel["candidates"]
        if not query:
            return candidates
        return [candidate for candidate in candidates
                if query in candidate["label"].casefold()]

    def set_candidate_hand_state(self, x=None, y=None, closed_fist=False):
        """Send normalized camera-cursor and fist state for card dwell hit testing."""
        if (not self.has_candidate_panel
                or isinstance(x, bool) or not isinstance(x, (int, float))
                or isinstance(y, bool) or not isinstance(y, (int, float))
                or not math.isfinite(x) or not math.isfinite(y)
                or not isinstance(closed_fist, bool)):
            self._candidate_hand_state = None
            return
        self._candidate_hand_state = {
            "x": min(1.0, max(0.0, float(x))),
            "y": min(1.0, max(0.0, float(y))),
            "closed_fist": closed_fist,
        }

    def set_drawing_state(self, points=(), drawing_enabled=False,
                          cursor=None, near_object=False, delete_progress=0.0,
                          hint_visible=True):
        """Send Python-captured stroke/cursor state to the WebView overlay."""
        safe_points = []
        if isinstance(points, (list, tuple)):
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or any(isinstance(value, bool)
                               or not isinstance(value, (int, float))
                               or not math.isfinite(value) for value in point)):
                    continue
                safe_points.append([
                    min(float(self.width), max(0.0, float(point[0]))),
                    min(float(self.height), max(0.0, float(point[1]))),
                ])

        safe_cursor = None
        if (isinstance(cursor, (list, tuple)) and len(cursor) == 2
                and all(not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(value) for value in cursor)):
            safe_cursor = [
                min(1.0, max(0.0, float(cursor[0]))),
                min(1.0, max(0.0, float(cursor[1]))),
            ]

        try:
            progress = float(delete_progress)
        except (TypeError, ValueError):
            progress = 0.0
        if not math.isfinite(progress):
            progress = 0.0
        self._drawing_state = {
            "points": safe_points,
            "drawing_enabled": bool(drawing_enabled),
            "cursor": safe_cursor,
            "near_object": bool(near_object),
            "delete_progress": min(1.0, max(0.0, progress)),
            "hint_visible": bool(hint_visible),
        }

    def confirm_candidate(self, candidate_id):
        """Confirm the currently highlighted ID once and close its panel."""
        if not self.has_candidate_panel or not isinstance(candidate_id, str):
            return False
        visible_candidates = self._visible_candidates()
        if not visible_candidates or self._candidate_index >= len(visible_candidates):
            return False
        candidate = visible_candidates[self._candidate_index]
        if candidate_id != candidate["id"]:
            return False
        self.selected_candidate = {
            "id": candidate["id"],
            "label": candidate["label"],
            "shape_label": self._candidate_panel["display_label"],
        }
        self._candidate_panel = None
        self._candidate_index = 0
        self._candidate_search_query = ""
        self._candidate_hand_state = None
        return True

    def navigate_candidate(self, direction):
        """Move the candidate highlight in taxonomy order, wrapping at ends."""
        if not self.has_candidate_panel or direction not in ("SWIPE_LEFT", "SWIPE_RIGHT"):
            return False
        count = len(self._visible_candidates())
        if not count:
            return False
        # Match the card rail's physical movement to the hand: rightward hand
        # motion advances the rail right (toward the previous candidate).
        delta = 1 if direction == "SWIPE_LEFT" else -1
        self._candidate_index = (self._candidate_index + delta) % count
        return True

    # ─────────────────────────────
    def render(self, display=None):
        """Push the current scene state to Three.js, if the window is
        attached and ready. Runs every call (not just on add/remove) so
        in-place transform edits from interaction/manipulation.py —
        which mutate an existing object's position/rotation/scale
        without re-adding it — still reach the live WebView scene.
        Python-captured stroke points and cursor state are rendered by a
        transparent canvas over the Three.js scene in this WebView.
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

        preview_payload = json.dumps(self._snap_preview)
        self._window.evaluate_js(
            "window.receiveSnapPreviewFromPython && "
            f"window.receiveSnapPreviewFromPython({preview_payload})"
        )

        hand_payload = json.dumps(serialize_hand(hand_obj) if hand_obj is not None else None)
        self._window.evaluate_js(
            f"window.receiveHandFromPython && window.receiveHandFromPython({hand_payload})"
        )

        hud_payload = json.dumps(self._hud_text)
        self._window.evaluate_js(
            f"window.receiveHudFromPython && window.receiveHudFromPython({hud_payload})"
        )

        candidate_payload = self._candidate_panel
        if candidate_payload is not None:
            candidate_payload = dict(candidate_payload)
            visible_candidates = self._visible_candidates()
            candidate_payload["search_query"] = self._candidate_search_query
            candidate_payload["selected_id"] = (
                visible_candidates[self._candidate_index]["id"]
                if visible_candidates else None
            )
        candidate_payload = json.dumps(candidate_payload)
        self._window.evaluate_js(
            "window.receiveCandidatePanelFromPython && "
            f"window.receiveCandidatePanelFromPython({candidate_payload})"
        )

        hand_state_payload = json.dumps(
            self._candidate_hand_state if self.has_candidate_panel else None
        )
        self._window.evaluate_js(
            "window.receiveCandidateHandStateFromPython && "
            f"window.receiveCandidateHandStateFromPython({hand_state_payload})"
        )
        drawing_payload = json.dumps(self._drawing_state)
        self._window.evaluate_js(
            "window.receiveDrawingStateFromPython && "
            f"window.receiveDrawingStateFromPython({drawing_payload})"
        )

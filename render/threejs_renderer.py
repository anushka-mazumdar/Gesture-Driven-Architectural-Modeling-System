"""
The sole 3D renderer backend: a Three.js scene in a pywebview window.

ThreeJSRenderer exposes the object-management surface main.py drives —
    `objects`, add_object(), remove_object(), clear_objects(),
    render() — and pushes the same mesh objects (vertices/normals/
indices/color/position/rotation/scale convention — see
render/mesh_bridge.py) to the Three.js scene over the Python -> JS
bridge. Bridge work is coalesced on a background thread. New objects send
geometry once; per-frame changes send only transforms and current UI state.
"""

import json
import math
import threading
import time

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
        self._workspace_settings = {
            "snapping_enabled": False,
            "mesh_view": False,
            "stroke_color": [0x66 / 255.0, 0xBF / 255.0, 1.0],
        }
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
        self._state_lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._sync_requested = threading.Event()
        self._sync_stopped = threading.Event()
        self._sync_condition = threading.Condition()
        self._sync_thread = None
        self._requested_generation = 0
        self._completed_generation = 0
        self._last_synchronized_generation = 0
        self._bridge_ids = {}
        self._bridge_objects = {}
        self._next_bridge_id = 1
        self._mesh_transforms = {}
        self._bridge_stats = {
            "requests": 0, "completed": 0, "coalesced": 0,
            "mesh_geometry_payloads": 0, "javascript_calls": 0,
            "last_sync_seconds": 0.0,
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
        self._ensure_sync_worker()

    def _ensure_sync_worker(self):
        with self._worker_lock:
            if self._sync_thread is not None and self._sync_thread.is_alive():
                return
            self._sync_stopped.clear()
            self._sync_thread = threading.Thread(
                target=self._sync_worker, name="threejs-bridge", daemon=True
            )
            self._sync_thread.start()

    def _sync_worker(self):
        while not self._sync_stopped.is_set():
            self._sync_requested.wait()
            if self._sync_stopped.is_set():
                break
            self._sync_requested.clear()
            with self._sync_condition:
                generation = self._requested_generation
            started = time.perf_counter()
            try:
                if self._ready and self._window is not None:
                    self._sync_scene()
            except Exception as error:
                print(f"Three.js bridge update failed: {error}")
            elapsed = time.perf_counter() - started
            with self._sync_condition:
                self._completed_generation = max(self._completed_generation, generation)
                self._bridge_stats["completed"] += 1
                self._bridge_stats["last_sync_seconds"] = elapsed
                self._bridge_stats["coalesced"] += max(
                    0, generation - self._last_synchronized_generation - 1
                )
                self._last_synchronized_generation = max(
                    self._last_synchronized_generation, generation
                )
                self._sync_condition.notify_all()

    def wait_for_sync(self, timeout=1.0):
        """Wait for queued bridge work; intended for integration checks."""
        deadline = time.monotonic() + timeout
        with self._sync_condition:
            target = self._requested_generation
            while self._completed_generation < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._sync_condition.wait(remaining)
            return True

    def performance_snapshot(self):
        """Return bridge counters for local diagnostics and performance tests."""
        with self._sync_condition:
            return dict(self._bridge_stats)

    def close_bridge(self):
        self._sync_stopped.set()
        self._sync_requested.set()
        thread = self._sync_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    # ─────────────────────────────
    def add_object(self, obj):
        with self._state_lock:
            self.objects.append(obj)

    def remove_object(self, obj):
        with self._state_lock:
            if obj in self.objects:
                self.objects.remove(obj)

    def replace_object(self, old_obj, new_obj):
        """Replace a scene object in place so its order stays stable."""
        with self._state_lock:
            try:
                index = self.objects.index(old_obj)
            except ValueError:
                return False
            self.objects[index] = new_obj
        return True

    def clear_objects(self):
        with self._state_lock:
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

    def set_workspace_settings(self, **settings):
        """Store display settings and the Python-owned snapping switch."""
        for key in ("snapping_enabled", "mesh_view"):
            if key in settings:
                self._workspace_settings[key] = bool(settings[key])
        if "stroke_color" in settings:
            try:
                color = [float(value) for value in settings["stroke_color"]]
                if (len(color) == 3 and all(math.isfinite(value) and 0 <= value <= 1
                                            for value in color)):
                    self._workspace_settings["stroke_color"] = color
            except (TypeError, ValueError):
                pass

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
                          hint_visible=True, stroke_color=None):
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
        color = self._workspace_settings.get("stroke_color", [0.4, 0.75, 1.0])
        if stroke_color is not None:
            try:
                candidate_color = [float(value) for value in stroke_color]
                if (len(candidate_color) == 3
                        and all(math.isfinite(value) and 0 <= value <= 1
                                for value in candidate_color)):
                    color = candidate_color
            except (TypeError, ValueError):
                pass
        self._drawing_state = {
            "points": safe_points,
            "drawing_enabled": bool(drawing_enabled),
            "cursor": safe_cursor,
            "near_object": bool(near_object),
            "delete_progress": min(1.0, max(0.0, progress)),
            "hint_visible": bool(hint_visible),
            "stroke_color": list(color),
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
        """Queue the latest scene state without waiting for WebView/JS work.

        The sync worker coalesces pending updates while retaining current
        hand, cursor, and object transform state.
        """
        if self._ready and self._window is not None:
            self._ensure_sync_worker()
            with self._sync_condition:
                self._requested_generation += 1
                self._bridge_stats["requests"] += 1
            self._sync_requested.set()
        return display

    def _sync_scene(self):
        with self._state_lock:
            objects = list(self.objects)
        hand_obj = None
        added = []
        transforms = []
        active_ids = set()
        for obj in objects:
            if getattr(obj, "kind", None) == "hand":
                hand_obj = obj
                continue
            if not len(getattr(obj, "vertices", [])):
                continue
            identity = id(obj)
            bridge_id = self._bridge_ids.get(identity)
            if bridge_id is None:
                bridge_id = self._next_bridge_id
                self._next_bridge_id += 1
                self._bridge_ids[identity] = bridge_id
                self._bridge_objects[identity] = obj
            active_ids.add(bridge_id)
            transform = {
                "id": bridge_id,
                "position": [float(value) for value in obj.position],
                "rotation": [float(value) for value in obj.rotation],
                "scale": [float(value) for value in obj.scale],
            }
            transform_key = (tuple(transform["position"]),
                             tuple(transform["rotation"]),
                             tuple(transform["scale"]))
            if bridge_id not in self._mesh_transforms:
                mesh_data = serialize_mesh(obj)
                mesh_data["id"] = bridge_id
                added.append(mesh_data)
                self._mesh_transforms[bridge_id] = transform_key
            elif self._mesh_transforms[bridge_id] != transform_key:
                transforms.append(transform)
                self._mesh_transforms[bridge_id] = transform_key

        removed = [bridge_id for bridge_id in self._mesh_transforms
                   if bridge_id not in active_ids]
        for bridge_id in removed:
            self._mesh_transforms.pop(bridge_id, None)
            for identity, assigned_id in list(self._bridge_ids.items()):
                if assigned_id == bridge_id:
                    self._bridge_ids.pop(identity, None)
                    self._bridge_objects.pop(identity, None)
        delta = {"added": added, "removed": removed, "transforms": transforms}
        self._bridge_stats["mesh_geometry_payloads"] += len(added)

        candidate_payload = self._candidate_panel
        if candidate_payload is not None:
            candidate_payload = dict(candidate_payload)
            visible_candidates = self._visible_candidates()
            candidate_payload["search_query"] = self._candidate_search_query
            candidate_payload["selected_id"] = (
                visible_candidates[self._candidate_index]["id"]
                if visible_candidates else None
            )
        frame = {
            "meshes": delta,
            "snap_preview": self._snap_preview,
            "workspace_settings": self._workspace_settings,
            "hand": serialize_hand(hand_obj) if hand_obj is not None else None,
            "hud": self._hud_text,
            "candidate_panel": candidate_payload,
            "candidate_hand_state": (self._candidate_hand_state
                                      if self.has_candidate_panel else None),
            "drawing": self._drawing_state,
        }
        payload = json.dumps(frame)
        self._window.evaluate_js(
            "window.receiveFrameFromPython && window.receiveFrameFromPython(" + payload + ")"
        )
        self._bridge_stats["javascript_calls"] += 1

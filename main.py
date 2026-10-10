import math
import threading
import time

from vision.hand_tracking         import HandTracker
from vision.landmark_utils        import (
    is_pinch, is_open_palm, is_closed_fist,
    is_index_only, is_peace_sign,
    get_hand_tilt_vector, get_peace_spread,
    palm_center
)
from gestures.gesture_stabilizer  import GestureStabilizer
from gestures.gesture_motion      import GestureMotion, PinchScale
from gestures.gesture_timer       import GestureTimer
from gestures.gesture_cooldown    import GestureCooldown
from gestures.gesture_state       import GestureStateMachine
from gestures.candidate_swipe_gate import CandidateSwipeGate
from shapes.stroke_capture     import StrokeCapture
from shapes.stroke_pipeline    import StrokePipeline
from shapes.shape_3d_factory   import Shape3DFactory
from render.threejs_renderer      import ThreeJSRenderer
from interaction.object_selection import ObjectSelection
from interaction.manipulation     import Manipulation
from interaction.snapping         import Snapping
from interaction.recommendation_mode import RecommendationInteractionMode
from interaction.candidate_placement import CandidatePlacement
from interaction.workspace_settings import WorkspaceSettings
from render.hand_mesh              import HandMesh
from webview_app.application_window import create_application_window

INDEX_TIP = 8
THUMB_TIP = 4
WRIST     = 0
PANEL_W   = 640
PANEL_H   = 480

# Set by _launch_application_and_run's window-closing handler so run_app's
# background thread stops (and releases the webcam) instead of being
# killed abruptly when the daemon thread dies with the process.
_stop_event = threading.Event()
_clear_scene_requested = threading.Event()

tracker          = HandTracker()
capture          = StrokeCapture()
factory          = Shape3DFactory(panel_width=PANEL_W, panel_height=PANEL_H)
pipeline         = StrokePipeline(factory=factory)
renderer         = ThreeJSRenderer(PANEL_W, PANEL_H)

obj_selection    = ObjectSelection(renderer)
snapping         = Snapping()
workspace_settings = WorkspaceSettings(snapping, renderer)
manipulation     = Manipulation(snapping=snapping)
hand_mesh        = HandMesh()  # 3D hand overlay

renderer.add_object(hand_mesh)
stabilizer       = GestureStabilizer(buffer_size=6)
cooldown         = GestureCooldown(cooldown_time=0.6)
motion           = GestureMotion()
candidate_swipe_gate = CandidateSwipeGate(settle_seconds=0.12, cooldown_seconds=0.6)
recommendation_mode = RecommendationInteractionMode(renderer, candidate_swipe_gate)
candidate_placement = CandidatePlacement(renderer, selection=obj_selection)
scale_detector   = PinchScale()
timer            = GestureTimer()
state_machine    = GestureStateMachine()
drawing_enabled = False

gesture           = None
state             = None
prev_x            = None
prev_y            = None
hand_speed        = 0.0                              # px/frame velocity magnitude
SMOOTH            = 0.45
converting        = False
convert_start     = 0.0
CONVERT_DURATION  = 1.2
manipulating      = False
scale_only_mode   = False
pinch_lost_frames = 0
PINCH_EXIT_FRAMES = 6   # frames pinch must be gone before exiting

delete_candidate  = None
delete_start_time = None
DELETE_HOLD       = 1.0



def enter_manipulation(obj, scale_only=False):
    global manipulating, scale_only_mode
    manipulation.set_object(obj)
    manipulating    = True
    scale_only_mode = scale_only
    if scale_only:
        manipulation.enter_scale_mode()


def exit_manipulation():
    global manipulating, scale_only_mode
    selected = obj_selection.get_selected()
    snapping.commit_preview(manipulation.active_object, renderer.objects)
    snapping.cancel_snap(renderer.objects)
    snapping.clear_preview()
    renderer.set_snap_preview(None)
    manipulation.stop_move()
    manipulation.stop_rotate()
    manipulation.stop_scale()
    obj_selection.deselect_all()
    manipulating    = False
    scale_only_mode = False


def try_convert_stroke():
    global converting, convert_start
    record = capture.finish()
    if record:
        result = pipeline.convert(record)
        renderer.set_candidate_recommendation(
            result.candidates if result is not None else None
        )
        mesh = result.mesh if result is not None else None
        if mesh:
            mesh.shape_class = result.shape_class
            renderer.add_object(mesh)
            candidate_placement.begin(result.record, mesh)
        else:
            candidate_placement.cancel_pending()
        converting    = True
        convert_start = time.time()


def process_normal_hand(landmarks, x, y, tilt, swipe, hand_speed,
                        navigation_swipe_in_progress=False):
    """Run the existing draw/manipulation/delete controls in NORMAL mode."""
    global gesture, pinch_lost_frames, delete_candidate, delete_start_time
    global drawing_enabled

    pinching = is_pinch(landmarks)
    index_only = is_index_only(landmarks) and not pinching
    peace = is_peace_sign(landmarks)
    fist = is_closed_fist(landmarks)
    open_palm = is_open_palm(landmarks)

    raw_gesture = None
    if pinching:
        raw_gesture = "PINCH"
    elif fist:
        raw_gesture = "FIST"
    elif open_palm:
        raw_gesture = "OPEN PALM"
    elif peace:
        raw_gesture = "PEACE"
    elif index_only:
        raw_gesture = "INDEX"

    stable_gesture = stabilizer.update(raw_gesture)
    gesture = cooldown.update(stable_gesture)

    if open_palm:
        if timer.check("OPEN PALM", 1.5):
            drawing_enabled = True
        if manipulating:
            exit_manipulation()

    hovered_obj = get_object_at(x, y)
    near_object = hovered_obj is not None
    delete_progress = 0.0

    if (fist and hovered_obj is not None and not manipulating
            and not navigation_swipe_in_progress):
        if delete_candidate is not hovered_obj:
            delete_candidate = hovered_obj
            delete_start_time = time.time()

        elapsed = time.time() - delete_start_time
        delete_progress = min(elapsed / DELETE_HOLD, 1.0)
        if elapsed >= DELETE_HOLD:
            if getattr(delete_candidate, 'kind', None) != 'hand':
                renderer.remove_object(delete_candidate)
            delete_candidate = None
            delete_start_time = None
    else:
        delete_candidate = None
        delete_start_time = None

    if manipulating:
        if open_palm:
            exit_manipulation()
        elif peace:
            pinch_lost_frames = 0
            if not manipulation.in_scale_mode:
                manipulation._prev_spread = None
                manipulation.in_scale_mode = True
            manipulation.update_move((x, y), PANEL_W, PANEL_H)
            manipulation.update_rotate_free(tilt)
            manipulation.update_scale_peace(get_peace_spread(landmarks))
            selected = obj_selection.get_selected()
            if hand_speed > Snapping.DETACH_SPEED and selected:
                snapping.detach(selected)
        else:
            pinch_lost_frames += 1
            if pinch_lost_frames >= PINCH_EXIT_FRAMES:
                pinch_lost_frames = 0
                exit_manipulation()

        if swipe in ('UP', 'DOWN'):
            manipulation.update_depth(swipe)

    elif index_only and drawing_enabled and not converting:
        if capture.is_paused():
            capture.resume_stroke()
        elif not capture.is_drawing():
            renderer.set_candidate_recommendation(None)
            candidate_placement.cancel_pending()
            capture.start_stroke(color=workspace_settings.stroke_color)
        capture.add_point((x, y))

    elif (drawing_enabled and not converting and not fist and not index_only
            and not navigation_swipe_in_progress):
        if peace:
            hit = get_object_at(x, y)
            if hit:
                obj_selection.selected_object = hit
                hit.selected = True
                enter_manipulation(hit, scale_only=False)

    # Anchor preview is display-only and is evaluated after the normal
    # manipulation updates, so it reflects current world transforms.
    if manipulating and peace:
        preview = snapping.preview_nearest_anchors(
            manipulation.active_object, renderer.objects
        )
        renderer.set_snap_preview(preview)
    elif not manipulating or open_palm:
        snapping.clear_preview()
        renderer.set_snap_preview(None)

    if not index_only and not manipulating and drawing_enabled and not converting:
        if capture.is_drawing():
            capture.pause_stroke()
        elif capture.is_paused() and capture.pause_expired() and capture.has_points():
            try_convert_stroke()

    return near_object, delete_progress


def get_object_at(x, y, threshold=120):

    for obj in renderer.objects:

        if getattr(obj, 'kind', None) == 'hand':
            continue

        bounds = obj.get_bounds()

        if bounds is None:
            sc_x =  obj.position[0] + PANEL_W / 2
            sc_y = -obj.position[1] + PANEL_H / 2
        else:
            scene_cx = obj.position[0] + float(bounds['center'][0])
            scene_cy = obj.position[1] + float(bounds['center'][1])
            sc_x =  scene_cx + PANEL_W / 2
            sc_y = -scene_cy + PANEL_H / 2

        if math.hypot(x - sc_x, y - sc_y) < threshold:
            return obj

    return None


def run_app():
    """The application loop — gesture detection, stroke processing,
    shape generation and manipulation logic. Runs on a background thread
    while pywebview owns the main thread's GUI loop; _stop_event lets the
    window-closing handler shut it down cleanly.
    """
    global gesture, state, prev_x, prev_y, hand_speed
    global converting, convert_start, manipulating, scale_only_mode
    global pinch_lost_frames, delete_candidate, delete_start_time

    while not _stop_event.is_set():

        if _clear_scene_requested.is_set():
            _clear_scene_requested.clear()
            if not recommendation_mode.active:
                candidate_placement.reset()
                capture.cancel()
                if manipulating:
                    manipulation.stop_move()
                    manipulation.stop_rotate()
                    manipulation.stop_scale()
                obj_selection.deselect_all()
                snapping.cancel_snap(renderer.objects)
                snapping.clear_preview()
                renderer.set_snap_preview(None)
                renderer.clear_objects()
                renderer.add_object(hand_mesh)
                renderer.set_candidate_hand_state(None)
                renderer.set_drawing_state(points=[], drawing_enabled=drawing_enabled,
                                           cursor=None, near_object=False,
                                           delete_progress=0.0, hint_visible=True,
                                           stroke_color=workspace_settings.stroke_color)
                renderer.set_hud_text(None)
                converting = False
                manipulating = False
                scale_only_mode = False
                pinch_lost_frames = 0
                delete_candidate = None
                delete_start_time = None

        frame, landmarks = tracker.get_frame()

        if frame is None:
            break

        tilt            = (0.0, 0.0)
        swipe           = None
        cursor          = None
        near_object     = False
        delete_progress = 0.0

        if landmarks is not None:

            index_pos = landmarks[INDEX_TIP]
            thumb_pos = landmarks[THUMB_TIP]
            wrist_pos = landmarks[WRIST]

            x = int(index_pos[0])
            y = int(index_pos[1])

            if prev_x is None:
                prev_x, prev_y = x, y

            raw_vx     = x - prev_x
            raw_vy     = y - prev_y
            hand_speed = math.hypot(raw_vx, raw_vy)

            x = int(prev_x * SMOOTH + x * (1 - SMOOTH))
            y = int(prev_y * SMOOTH + y * (1 - SMOOTH))
            prev_x, prev_y = x, y
            cursor = (x / PANEL_W, y / PANEL_H)

            tilt  = get_hand_tilt_vector(landmarks)
            swipe = motion.detect_swipe(landmarks)
            hand_mesh.update_from_landmarks(landmarks, PANEL_W, PANEL_H)
            recommendation_mode_active = recommendation_mode.update(swipe)
            if recommendation_mode_active:
                # RECOMMENDATION owns all hand input; no gesture/action code
                # for drawing, selecting, moving, deleting, or depth runs here.
                renderer.set_candidate_hand_state(
                    x / PANEL_W, y / PANEL_H, is_closed_fist(landmarks)
                )
                snapping.clear_preview()
                renderer.set_snap_preview(None)
                delete_candidate = None
                delete_start_time = None
                pinch_lost_frames = 0
                near_object = False
                delete_progress = 0.0
            else:
                renderer.set_candidate_hand_state(None)
                near_object, delete_progress = process_normal_hand(
                    landmarks, x, y, tilt, swipe, hand_speed
                )

        else:
            recommendation_mode_active = recommendation_mode.update(None)
            renderer.set_candidate_hand_state(None)
            snapping.clear_preview()
            renderer.set_snap_preview(None)
            # A visible recommendation panel consumes input even while the
            # hand tracker temporarily loses landmarks.
            if (not recommendation_mode_active and capture.is_paused()
                    and capture.pause_expired()):
                if capture.has_points():
                    try_convert_stroke()

        if converting:
            elapsed = time.time() - convert_start
            if elapsed >= CONVERT_DURATION:
                converting = False

        stroke_points = (capture.get_current()
                         if capture.is_drawing() or capture.is_paused() else [])
        renderer.set_drawing_state(
            points=stroke_points,
            drawing_enabled=drawing_enabled and not recommendation_mode_active,
            cursor=cursor,
            near_object=near_object,
            delete_progress=delete_progress,
            hint_visible=not recommendation_mode_active,
            stroke_color=(capture.current.color or workspace_settings.stroke_color),
        )

        selected = obj_selection.get_selected()
        shape_class = getattr(selected, 'shape_class', None) if selected else None
        renderer.set_hud_text(
            f"Shape: {shape_class.kind} ({shape_class.confidence:.0%})"
            if drawing_enabled and shape_class is not None else None
        )
        renderer.render()

    tracker.release()


def _launch_application_and_run():
    """Open the combined drawing/Three.js WebView and run the Python
    camera and gesture loop on a background thread.
    """
    import webview
    class _Api:
        """JS-callable surface for the single application window."""

        def __init__(self):
            self._window = None

        def exit(self):
            if self._window is not None:
                self._window.destroy()

        def confirm_candidate(self, candidate_id):
            return candidate_placement.confirm(candidate_id)

        def search_candidates(self, query):
            return renderer.set_candidate_search_query(query)

        def cancel_candidate_panel(self):
            candidate_placement.cancel()
            return True

        def clear_scene(self):
            if recommendation_mode.active:
                return False
            _clear_scene_requested.set()
            return True

        def set_snapping_enabled(self, enabled):
            if recommendation_mode.active:
                return snapping.enabled
            return workspace_settings.set_snapping_enabled(enabled)

        def set_mesh_view(self, enabled):
            renderer.set_workspace_settings(mesh_view=enabled)
            return bool(enabled)

        def set_drawing_enabled(self, enabled):
            global drawing_enabled
            if recommendation_mode.active:
                return drawing_enabled
            drawing_enabled = bool(enabled)
            return drawing_enabled

        def set_stroke_color(self, color):
            return workspace_settings.set_stroke_color(color)

    api = _Api()

    window = create_application_window(
        webview, api, renderer, _stop_event.set
    )
    api._window = window

    worker = threading.Thread(target=run_app, daemon=True)
    worker.start()
    webview.start(http_server=True, debug=False)

    # webview.start() returns once the window closes; give run_app's
    # loop a moment to see _stop_event and release the webcam cleanly before
    # the process exits.
    _stop_event.set()
    worker.join(timeout=2.0)
    renderer.close_bridge()


if __name__ == "__main__":
    _launch_application_and_run()

import cv2
import math
import numpy as np
import time

from sketch.sketch_panel          import SketchPanel
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
from shapes.stroke_processing     import StrokeProcessor
from shapes.shape_3d_factory      import Shape3DFactory
from render.renderer              import Renderer
from interaction.object_selection import ObjectSelection
from interaction.manipulation     import Manipulation
from interaction.snapping         import Snapping

INDEX_TIP = 8
THUMB_TIP = 4
WRIST     = 0
PANEL_W   = 640
PANEL_H   = 480

# ── Initialise ────────────────────────────────────────────────────────────────

tracker          = HandTracker()
stroke_processor = StrokeProcessor()
factory          = Shape3DFactory(panel_width=PANEL_W, panel_height=PANEL_H)
renderer         = Renderer(PANEL_W, PANEL_H)
obj_selection    = ObjectSelection(renderer)
manipulation     = Manipulation()
snapping         = Snapping()
stabilizer       = GestureStabilizer(buffer_size=4)
cooldown         = GestureCooldown(cooldown_time=0.3)
motion           = GestureMotion()
scale_detector   = PinchScale()
timer            = GestureTimer()
state_machine    = GestureStateMachine()
panel            = SketchPanel(width=PANEL_W, height=PANEL_H)

gesture          = None
state            = None
prev_x           = None
prev_y           = None
SMOOTH           = 0.45
converting       = False
convert_start    = 0.0
CONVERT_DURATION = 1.2
manipulating     = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def enter_manipulation(obj):
    global manipulating
    manipulation.set_object(obj)
    manipulating = True


def exit_manipulation():
    global manipulating
    selected = obj_selection.get_selected()
    if selected:
        snapping.confirm_snap(selected)
    manipulation.stop_move()
    manipulation.stop_rotate()
    manipulation.stop_scale()
    obj_selection.deselect_all()
    manipulating = False


def try_convert_stroke():
    global converting, convert_start
    pts, closed = stroke_processor.finish_stroke()
    panel.finish_stroke()
    if pts:
        mesh = factory.create_from_stroke(pts, closed=closed)
        if mesh:
            renderer.add_object(mesh)
        panel.stroke_layer[:] = 0
        converting    = True
        convert_start = time.time()


def draw_cursor(display, x, y, near_object=False):
    color  = (0, 255, 150) if near_object else (200, 200, 200)
    radius = 12 if near_object else 8
    cv2.circle(display, (x, y), radius, color, 2)
    cv2.circle(display, (x, y), 2,      color, -1)


# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    frame, landmarks = tracker.get_frame()

    if frame is None:
        break

    # safe defaults — always defined regardless of landmarks
    tilt        = (0.0, 0.0)
    swipe       = None
    cursor_x    = prev_x or PANEL_W // 2
    cursor_y    = prev_y or PANEL_H // 2
    near_object = False

    if landmarks is not None:

        index_pos = landmarks[INDEX_TIP]
        thumb_pos = landmarks[THUMB_TIP]
        wrist_pos = landmarks[WRIST]

        x = int(index_pos[0])
        y = int(index_pos[1])

        if prev_x is None:
            prev_x, prev_y = x, y

        x = int(prev_x * SMOOTH + x * (1 - SMOOTH))
        y = int(prev_y * SMOOTH + y * (1 - SMOOTH))
        prev_x, prev_y = x, y
        cursor_x, cursor_y = x, y

        # ── Compute tilt and swipe first ──────────────────────────────
        tilt  = get_hand_tilt_vector(landmarks)
        swipe = motion.detect_swipe(landmarks)

        # ── Detect all gestures this frame ────────────────────────────
        pinching   = is_pinch(landmarks)
        index_only = is_index_only(landmarks) and not pinching
        peace      = is_peace_sign(landmarks)
        fist       = is_closed_fist(landmarks)
        open_palm  = is_open_palm(landmarks)

        # ── Raw gesture priority ──────────────────────────────────────
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
        gesture        = cooldown.update(stable_gesture)

        # ── Panel activation ──────────────────────────────────────────
        if gesture == "OPEN PALM":
            if timer.check("OPEN PALM", 1.5):
                panel.active = True

        # ── Clear all ─────────────────────────────────────────────────
        if gesture == "FIST":
            if timer.check("FIST", 2):
                renderer.clear_objects()
                panel.stroke_layer[:] = 0
                panel.completed_strokes.clear()
                stroke_processor.finish_stroke()
                converting   = False
                manipulating = False
                obj_selection.deselect_all()

        # ── Near object check ─────────────────────────────────────────
        for obj in renderer.objects:
            bounds = obj.get_bounds()
            if bounds is None:
                continue
            cx = obj.position[0] + bounds['center'][0] + PANEL_W / 2
            cy = -(obj.position[1] + bounds['center'][1]) + PANEL_H / 2
            if math.hypot(x - cx, y - cy) < 60:
                near_object = True
                break

        # ══════════════════════════════════════════════════════════════
        # MANIPULATION MODE
        # ══════════════════════════════════════════════════════════════
        if manipulating:

            if peace and not manipulation.in_scale_mode:
                manipulation.enter_scale_mode()

            if pinching and manipulation.in_scale_mode:
                manipulation.exit_scale_mode()

            if pinching and not manipulation.in_scale_mode:
                manipulation.update_move((x, y), PANEL_W, PANEL_H)
                selected = obj_selection.get_selected()
                snapping.update(selected, renderer.objects)

            if manipulation.in_scale_mode:
                spread = get_peace_spread(landmarks)
                manipulation.update_scale_peace(spread)

            manipulation.update_rotate_free(tilt)

            if swipe in ('UP', 'DOWN'):
                manipulation.update_depth(swipe)

            if not pinching and not manipulation.in_scale_mode:
                exit_manipulation()

        # ══════════════════════════════════════════════════════════════
        # SELECTION
        # ══════════════════════════════════════════════════════════════
        elif pinching and panel.active and not converting:

            if timer.check("PINCH", 1.0):
                hit = obj_selection.update((x, y), PANEL_W, PANEL_H)
                if hit:
                    enter_manipulation(hit)

        # ══════════════════════════════════════════════════════════════
        # DRAWING MODE
        # ══════════════════════════════════════════════════════════════
        elif panel.active and not converting and not manipulating:

            if index_only:

                if stroke_processor.is_paused():
                    stroke_processor.resume_stroke()
                elif not stroke_processor.is_drawing():
                    stroke_processor.start_stroke()

                panel.draw_stroke((x, y))
                stroke_processor.add_point((x, y))

            else:

                if stroke_processor.is_drawing():
                    stroke_processor.pause_stroke()
                    panel.finish_stroke()

                elif stroke_processor.is_paused():
                    if stroke_processor.pause_expired():
                        if stroke_processor.has_points():
                            try_convert_stroke()

    else:
        # no hand — expire pause if needed
        if stroke_processor.is_paused() and stroke_processor.pause_expired():
            if stroke_processor.has_points():
                try_convert_stroke()

    # ── Compose display ───────────────────────────────────────────────
    display = panel.render(frame)

    if renderer.objects:
        display = renderer.composite_onto(display)

    draw_cursor(display, cursor_x, cursor_y, near_object)

    # ── Converting flash ──────────────────────────────────────────────
    if converting:
        elapsed = time.time() - convert_start
        if elapsed < CONVERT_DURATION:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (PANEL_W, PANEL_H),
                          (20, 10, 10), -1)
            display = cv2.addWeighted(overlay, 0.35, display, 0.65, 0)
            progress = elapsed / CONVERT_DURATION
            bar_w    = int(PANEL_W * 0.4)
            bar_x    = (PANEL_W - bar_w) // 2
            bar_y    = PANEL_H // 2 + 30
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + 8), (60, 60, 80), -1)
            cv2.rectangle(display, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 8),
                          (100, 200, 255), -1)
            cv2.putText(display, "Converting to 3D...",
                        (PANEL_W // 2 - 110, PANEL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (100, 200, 255), 2)
        else:
            converting = False

    # ── HUD ───────────────────────────────────────────────────────────
    if panel.active:
        if manipulating:
            mode = "SCALE MODE" if manipulation.in_scale_mode else "MOVE MODE"
        elif stroke_processor.is_drawing():
            mode = "DRAWING"
        elif stroke_processor.is_paused():
            mode = "PAUSED..."
        else:
            mode = "READY"
        cv2.putText(display, f"State: {mode}", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 180), 1)

    if manipulating:
        if manipulation.in_scale_mode:
            hint = "Spread fingers: Scale  |  Pinch: Back to Move"
        else:
            hint = "Pinch+Move  |  Peace:Scale  |  Tilt:Rotate  |  Swipe U/D:Depth"
        cv2.putText(display, hint, (12, PANEL_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 200, 255), 1)
    elif panel.active and not converting:
        cv2.putText(display,
                    "Index finger: Draw  |  Pinch on shape: Select",
                    (12, PANEL_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 180), 1)

    cv2.imshow("2D to 3D", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('w'):
        renderer.toggle_wireframe()

tracker.release()
cv2.destroyAllWindows()
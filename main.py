import cv2
import numpy as np
import time

from sketch.sketch_panel          import SketchPanel
from vision.hand_tracking         import HandTracker
from vision.landmark_utils        import (
    is_pinch, is_open_palm, is_closed_fist, palm_center
)
from gestures.gesture_stabilizer  import GestureStabilizer
from gestures.gesture_motion      import GestureMotion, PinchScale
from gestures.gesture_timer       import GestureTimer
from gestures.gesture_cooldown    import GestureCooldown
from gestures.gesture_state       import GestureStateMachine
from shapes.stroke_processing     import StrokeProcessor
from shapes.shape_3d_factory      import Shape3DFactory
from render.renderer              import Renderer
# from interaction.object_selection import ObjectSelection
# from interaction.manipulation     import Manipulation
# from interaction.snapping         import Snapping

INDEX_TIP = 8
PANEL_W   = 1280
PANEL_H   = 720

# ── Initialise modules ────────────────────────────────────────────────────────

tracker          = HandTracker()
stroke_processor = StrokeProcessor()
factory          = Shape3DFactory(panel_width=PANEL_W, panel_height=PANEL_H)
renderer         = Renderer(PANEL_W, PANEL_H)
# obj_selection    = ObjectSelection(renderer)
# manipulation     = Manipulation()
# snapping         = Snapping()
stabilizer       = GestureStabilizer(buffer_size=5)
cooldown         = GestureCooldown(cooldown_time=0.5)
motion           = GestureMotion()
scale            = PinchScale()
timer            = GestureTimer()
state_machine    = GestureStateMachine()
panel            = SketchPanel(width=PANEL_W, height=PANEL_H)

# ── Frame-level state ─────────────────────────────────────────────────────────

gesture          = None
state            = None
prev_x           = None
prev_y           = None
SMOOTH           = 0.7
converting       = False
convert_start    = 0.0
CONVERT_DURATION = 1.2

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:

    frame, landmarks = tracker.get_frame()

    if frame is None:
        break

    raw_gesture = None

    if landmarks is not None:

        if is_pinch(landmarks):
            raw_gesture = "PINCH"
        elif is_closed_fist(landmarks):
            raw_gesture = "FIST"
        elif is_open_palm(landmarks):
            raw_gesture = "OPEN PALM"

        stable_gesture = stabilizer.update(raw_gesture)
        gesture        = cooldown.update(stable_gesture)

        if gesture == "OPEN PALM":
            if timer.check("OPEN PALM", 3):
                panel.active = True

        elif gesture == "PINCH":
            timer.check("PINCH", 1)

        elif gesture == "FIST":
            if timer.check("FIST", 2):
                renderer.clear_objects()
                panel.stroke_layer[:] = 0
                panel.completed_strokes.clear()
                converting = False

        state = state_machine.update(gesture)

        motion.detect_swipe(landmarks)
        scale.detect_scale(landmarks)

        index_point = landmarks[INDEX_TIP]
        x = int(index_point[0])
        y = int(index_point[1])

        if prev_x is None:
            prev_x, prev_y = x, y

        x = int(prev_x * SMOOTH + x * (1 - SMOOTH))
        y = int(prev_y * SMOOTH + y * (1 - SMOOTH))
        prev_x, prev_y = x, y

        if panel.active and not converting:

            if state == "DRAWING":

                if not stroke_processor.is_drawing():
                    stroke_processor.start_stroke()

                panel.draw_stroke((x, y))
                stroke_processor.add_point((x, y))

            else:

                if stroke_processor.is_drawing():

                    # ── closed flag comes from StrokeProcessor now ────
                    pts, closed = stroke_processor.finish_stroke()
                    panel.finish_stroke()

                    if pts:
                        mesh = factory.create_from_stroke(pts, closed=closed)
                        if mesh:
                            renderer.add_object(mesh)

                        panel.stroke_layer[:] = 0
                        converting    = True
                        convert_start = time.time()

        center = palm_center(landmarks)
        cv2.circle(frame, tuple(center), 6, (255, 255, 0), -1)

    # ── Compose display ───────────────────────────────────────────────
    display = panel.render(frame)

    if renderer.objects:
        display = renderer.composite_onto(display)

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

            cv2.rectangle(display,
                          (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                          (60, 60, 80), -1)
            cv2.rectangle(display,
                          (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 8),
                          (100, 200, 255), -1)

            cv2.putText(display, "Converting to 3D...",
                        (PANEL_W//2 - 140, PANEL_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (100, 200, 255), 2)
        else:
            converting = False

    if panel.active:
        status = f"State: {state or 'IDLE'}"
        cv2.putText(display, status, (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 180), 1)

    cv2.imshow("2D to 3D", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('w'):
        renderer.toggle_wireframe()

tracker.release()
cv2.destroyAllWindows()
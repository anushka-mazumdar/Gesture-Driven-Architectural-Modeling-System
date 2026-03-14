import cv2

from sketch.sketch_panel import SketchPanel
from vision.hand_tracking import HandTracker
from vision.landmark_utils import (
    is_pinch,
    is_open_palm,
    is_closed_fist,
    palm_center
)

from gestures.gesture_stabilizer import GestureStabilizer
from gestures.gesture_motion import GestureMotion, PinchScale
from gestures.gesture_timer import GestureTimer
from gestures.gesture_cooldown import GestureCooldown
from gestures.gesture_state import GestureStateMachine


INDEX_TIP = 8


tracker = HandTracker()

stabilizer = GestureStabilizer(buffer_size=5)
cooldown = GestureCooldown(cooldown_time=0.5)
motion = GestureMotion()
scale = PinchScale()
timer = GestureTimer()
state_machine = GestureStateMachine()

panel = SketchPanel()


gesture = None
raw_gesture = None
state = None


while True:

    frame, landmarks = tracker.get_frame()

    if frame is None:
        break

    raw_gesture = None

    if landmarks is not None:

        # -------- Base Gesture Detection --------

        if is_pinch(landmarks):
            raw_gesture = "PINCH"

        elif is_closed_fist(landmarks):
            raw_gesture = "FIST"

        elif is_open_palm(landmarks):
            raw_gesture = "OPEN PALM"


        # -------- Stabilize Gesture --------

        stable_gesture = stabilizer.update(raw_gesture)


        # -------- Cooldown Filter --------

        gesture = cooldown.update(stable_gesture)


        # -------- Gesture Timing --------

        if gesture == "OPEN PALM":
            if timer.check("OPEN PALM", 3):
                cv2.putText(frame, "OPEN PALM (3s)", (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        elif gesture == "PINCH":
            if timer.check("PINCH", 1):
                cv2.putText(frame, "PINCH HOLD", (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        elif gesture == "FIST":
            if timer.check("FIST", 2):
                cv2.putText(frame, "FIST (DELETE)", (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        # -------- Gesture State Machine --------

        state = state_machine.update(gesture)


        # -------- Motion Gestures --------

        swipe = motion.detect_swipe(landmarks)
        scale_event = scale.detect_scale(landmarks)

        if swipe:
            cv2.putText(frame, swipe, (30,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        if scale_event:
            cv2.putText(frame, scale_event, (30,140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)


        # -------- Drawing Logic --------

        index_point = landmarks[INDEX_TIP]

        x = int(index_point[0] * 2)
        y = int(index_point[1] * 2)

        if state == "DRAWING":
            panel.draw_stroke((x, y))
        else:
             panel.finish_stroke()


        # -------- Palm Center --------

        center = palm_center(landmarks)
        cv2.circle(frame, tuple(center), 6, (255,255,0), -1)


    # -------- Display State --------

    if state:
        cv2.putText(frame, f"STATE: {state}", (30,180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


    # -------- Render Panel --------

    display = panel.render(frame)

    cv2.imshow("Sketch Panel", display)


    if cv2.waitKey(1) & 0xFF == 27:
        break


tracker.release()
cv2.destroyAllWindows()
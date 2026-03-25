import cv2

from sketch.sketch_panel import SketchPanel
from vision.hand_tracking import HandTracker
from vision.landmark_utils import (
    is_pinch,
    is_open_palm,
    is_closed_fist,
    palm_center
)

# from shapes.shape_2d_classifier import ShapeClassifier
from gestures.gesture_stabilizer import GestureStabilizer
from gestures.gesture_motion import GestureMotion, PinchScale
from gestures.gesture_timer import GestureTimer
from gestures.gesture_cooldown import GestureCooldown
from gestures.gesture_state import GestureStateMachine
from shapes.contour_processing import ContourProcessor
from shapes.stroke_processing import StrokeProcessor

INDEX_TIP = 8


tracker = HandTracker()
processor = ContourProcessor()
# classifier = ShapeClassifier()
stroke_processor = StrokeProcessor()
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


prev_x = None
prev_y = None
smooth_factor = 0.7


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


        # -------- Stabilization --------

        stable_gesture = stabilizer.update(raw_gesture)


        # -------- Cooldown --------

        gesture = cooldown.update(stable_gesture)


        # -------- Gesture Timing --------

        if gesture == "OPEN PALM":
            if timer.check("OPEN PALM", 3):
                panel.active = True

        elif gesture == "PINCH":
            timer.check("PINCH", 1)

        elif gesture == "FIST":
            timer.check("FIST", 2)


        # -------- State Machine --------

        state = state_machine.update(gesture)


        # -------- Motion Detection --------

        motion.detect_swipe(landmarks)
        scale.detect_scale(landmarks)


        # -------- Drawing Logic --------

        index_point = landmarks[INDEX_TIP]

        x = int(index_point[0])
        y = int(index_point[1])


        # smoothing

        if prev_x is None:
            prev_x, prev_y = x, y

        x = int(prev_x * smooth_factor + x * (1 - smooth_factor))
        y = int(prev_y * smooth_factor + y * (1 - smooth_factor))

        prev_x, prev_y = x, y


        # draw only when system active

        if panel.active:

            if state == "DRAWING":
                panel.draw_stroke((x, y))
            else:
                panel.finish_stroke()


        # -------- Palm Center --------

        center = palm_center(landmarks)
        cv2.circle(frame, tuple(center), 6, (255,255,0), -1)


    # -------- Render Panel --------

    display = panel.render(frame)


    # -------- Contour + Shape Detection --------

    if panel.completed_strokes:

        stroke = panel.completed_strokes[-1]

        simplified = processor.simplify_stroke(stroke)

        if simplified:

            for p in simplified:
                cv2.circle(display, p, 6, (0,255,0), -1)

            # shape = classifier.classify(simplified)

            # if shape:
            #     cv2.putText(
            #         display,
            #         shape,
            #         (simplified[0][0], simplified[0][1] - 20),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.8,
            #         (0,255,0),
            #         2
            #     )


    cv2.imshow("Sketch Panel", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break


tracker.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_draw = mp.solutions.drawing_utils

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmarks = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []

            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                landmarks.append(np.array([int(lm.x * w), int(lm.y * h)]))

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

        return frame, landmarks

    def release(self):
        self.cap.release()

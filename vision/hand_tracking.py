import os
import time

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task"
)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    # Legacy solutions API (mp.solutions.hands) is not built into mediapipe
    # wheels for newer Python versions, so use the Tasks API instead.
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        HAS_TASKS_API = os.path.exists(MODEL_PATH)
    except ImportError:
        HAS_TASKS_API = False
except ImportError:
    HAS_MEDIAPIPE = False
    HAS_TASKS_API = False


class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mock_mode = not (HAS_MEDIAPIPE and HAS_TASKS_API)

        if not self.mock_mode:
            base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._start_time = time.time()
            self._last_timestamp_ms = -1

        self.mock_hand_pos = [320, 240]  # Center of screen
        self.mock_frame_count = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)  # Mirror

        if self.mock_mode:
            # Mock hand tracking - simulate hand moving in circles
            self.mock_frame_count += 1
            center_x = 320
            center_y = 240
            radius = 100
            angle = self.mock_frame_count * 0.05
            self.mock_hand_pos = [
                int(center_x + radius * np.cos(angle)),
                int(center_y + radius * np.sin(angle))
            ]

            # Mock landmarks - just return index finger position
            mock_landmarks = np.zeros((21, 2))
            mock_landmarks[8] = self.mock_hand_pos  # INDEX_TIP
            mock_landmarks[4] = [self.mock_hand_pos[0] - 20, self.mock_hand_pos[1] - 20]  # THUMB_TIP
            mock_landmarks[0] = [self.mock_hand_pos[0], self.mock_hand_pos[1] + 40]  # WRIST

            return frame, mock_landmarks

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = [[lm.x * 640, lm.y * 480] for lm in result.hand_landmarks[0]]
            return frame, np.array(landmarks)
        else:
            return frame, None

    def release(self):
        self.cap.release()
        if not self.mock_mode:
            self.landmarker.close()

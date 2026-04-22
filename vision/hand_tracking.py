import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    # Try the new API first
    try:
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
    except AttributeError:
        # Try tasks API
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            HAS_MEDIAPIPE = 'tasks'
        except ImportError:
            HAS_MEDIAPIPE = False
except ImportError:
    HAS_MEDIAPIPE = False

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if HAS_MEDIAPIPE == 'tasks':
            # Use tasks API
            self.hand_landmarker = None  # Would need proper initialization
        elif HAS_MEDIAPIPE:
            # Use solutions API
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            # Mock mode
            self.hands = None

        self.mock_mode = not HAS_MEDIAPIPE
        self.mock_hand_pos = [320, 240]  # Center of screen
        self.mock_frame_count = 0

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

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
        elif HAS_MEDIAPIPE == 'tasks':
            # Tasks API implementation would go here
            return frame, None
        else:
            # Solutions API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.append([lm.x * 640, lm.y * 480])
                return frame, np.array(landmarks)
            else:
                return frame, None

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
        elif HAS_MEDIAPIPE == 'tasks':
            # Tasks API implementation would go here
            return frame, None
        else:
            # Solutions API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.append([lm.x * 640, lm.y * 480])
                return frame, np.array(landmarks)
            else:
                return frame, None

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

        return frame, landmarks

    def release(self):
        self.cap.release()

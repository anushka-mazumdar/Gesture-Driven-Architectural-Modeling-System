import numpy as np
from vision.landmark_utils import distance

INDEX_TIP = 8
THUMB_TIP = 4


class GestureMotion:

    def __init__(self):
        self.prev_index = None

    def detect_swipe(self, landmarks, threshold=60):

        index = np.array(landmarks[INDEX_TIP])

        if self.prev_index is None:
            self.prev_index = index
            return None

        dx = index[0] - self.prev_index[0]
        dy = index[1] - self.prev_index[1]

        self.prev_index = index

        if abs(dx) > threshold and abs(dx) > abs(dy):

            if dx > 0:
                return "SWIPE_RIGHT"
            else:
                return "SWIPE_LEFT"

        if abs(dy) > threshold and abs(dy) > abs(dx):

            if dy > 0:
                return "SWIPE_DOWN"
            else:
                return "SWIPE_UP"

        return None


class PinchScale:

    def __init__(self):
        self.prev_distance = None

    def detect_scale(self, landmarks, threshold=8):

        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        d = distance(thumb, index)

        if self.prev_distance is None:
            self.prev_distance = d
            return None

        change = d - self.prev_distance
        self.prev_distance = d

        if change > threshold:
            return "PINCH_OPEN"

        if change < -threshold:
            return "PINCH_CLOSE"

        return None
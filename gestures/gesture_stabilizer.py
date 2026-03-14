from collections import deque


class GestureStabilizer:

    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.gesture_buffer = deque(maxlen=buffer_size)

    def update(self, gesture):

        self.gesture_buffer.append(gesture)

        if len(self.gesture_buffer) < self.buffer_size:
            return None

        first = self.gesture_buffer[0]

        for g in self.gesture_buffer:
            if g != first:
                return None

        return first
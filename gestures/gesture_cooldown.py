import time


class GestureCooldown:

    def __init__(self, cooldown_time=0.5):
        self.cooldown_time = cooldown_time
        self.last_gesture_time = 0
        self.current_gesture = None


    def update(self, gesture):

        current_time = time.time()

        if gesture is None:
            return None

        if self.current_gesture != gesture:
            if current_time - self.last_gesture_time < self.cooldown_time:
                return self.current_gesture

            self.current_gesture = gesture
            self.last_gesture_time = current_time

        return self.current_gesture
import time


class GestureTimer:

    def __init__(self):
        self.start_time = None
        self.active_gesture = None

    def check(self, gesture, hold_time):

        if gesture != self.active_gesture:

            self.active_gesture = gesture
            self.start_time = time.time()
            return False

        if self.start_time is None:
            return False

        elapsed = time.time() - self.start_time

        if elapsed >= hold_time:
            return True

        return False
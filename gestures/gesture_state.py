class GestureState:

    IDLE = "IDLE"
    SKETCH_MODE = "SKETCH_MODE"
    DRAWING = "DRAWING"
    OBJECT_SELECTED = "OBJECT_SELECTED"

class GestureStateMachine:

    def __init__(self):
        self.state = GestureState.IDLE


    def update(self, gesture):

        if self.state == GestureState.IDLE:

            if gesture == "OPEN PALM":
                self.state = GestureState.SKETCH_MODE


        elif self.state == GestureState.SKETCH_MODE:

            if gesture == "PINCH":
                self.state = GestureState.DRAWING


        elif self.state == GestureState.DRAWING:

            if gesture is None:
                self.state = GestureState.SKETCH_MODE


        if gesture == "FIST":
            self.state = GestureState.IDLE


        return self.state
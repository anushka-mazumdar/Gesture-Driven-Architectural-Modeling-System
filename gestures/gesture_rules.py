class GestureRules:

    def __init__(self):
        pass


    def interpret(self, gesture, state, panel_active):

        """
        Convert detected gestures into system actions.

        Parameters
        ----------
        gesture : str
            Detected gesture (PINCH, OPEN PALM, FIST, SWIPE_LEFT, SWIPE_RIGHT)

        state : str
            Current system state

        panel_active : bool
            Whether sketch panel is activated

        Returns
        -------
        action : str or None
            Action for the system to execute
        """


        if not panel_active:

            if gesture == "OPEN PALM":
                return "ACTIVATE_PANEL"

            return None



        if state == "IDLE":

            if gesture == "OPEN PALM":
                return "ENTER_SKETCH_MODE"



        if state == "SKETCH_MODE":

            if gesture == "PINCH":
                return "START_DRAWING"

            if gesture == "FIST":
                return "CLEAR_CANVAS"



        if state == "DRAWING":

            if gesture != "PINCH":
                return "FINISH_STROKE"



        if state == "SELECTION_MODE":

            if gesture == "SWIPE_LEFT":
                return "PREVIOUS_OPTION"

            if gesture == "SWIPE_RIGHT":
                return "NEXT_OPTION"

            if gesture == "PINCH":
                return "CONFIRM_OBJECT"


        return None
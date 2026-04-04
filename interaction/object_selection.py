import numpy as np
import math


class ObjectSelection:

    def __init__(self, renderer):
        self.renderer        = renderer
        self.selected_object = None
        self.select_radius   = 80.0   # px — how close finger must be to object centre


    def update(self, finger_pos_2d, panel_w, panel_h):
        """
        Call every frame with the index finger screen position.
        Finds the closest object within select_radius and selects it.
        """

        # convert screen pos to scene coords
        sx =  (finger_pos_2d[0] - panel_w / 2.0)
        sy = -(finger_pos_2d[1] - panel_h / 2.0)

        best      = None
        best_dist = float('inf')

        for obj in self.renderer.objects:

            bounds = obj.get_bounds()
            if bounds is None:
                continue

            cx = obj.position[0] + bounds['center'][0]
            cy = obj.position[1] + bounds['center'][1]

            dist = math.hypot(sx - cx, sy - cy)

            if dist < self.select_radius and dist < best_dist:
                best_dist = dist
                best      = obj

        # deselect previous
        if self.selected_object and self.selected_object is not best:
            self.selected_object.selected = False

        self.selected_object = best

        if best:
            best.selected = True

        return best


    def deselect_all(self):
        for obj in self.renderer.objects:
            obj.selected = False
        self.selected_object = None


    def get_selected(self):
        return self.selected_object
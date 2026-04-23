import numpy as np
import math


class Manipulation:

    def __init__(self, snapping=None):
        """
        Pass the shared Snapping instance so manipulation can propagate
        moves/rotations/scales to the whole SnapGroup automatically.
        """
        self.snapping        = snapping   # injected from main.py

        self.active_object   = None

        self._prev_move_pos  = None
        self._prev_tilt      = None
        self._prev_spread    = None
        self.scale_min       = 0.15
        self.scale_max       = 6.0
        self.in_scale_mode   = False

        self.move_sensitivity   = 1.0
        self.rotate_sensitivity = 120.0
        self.scale_sensitivity  = 0.012


    def set_object(self, obj):
        self.active_object  = obj
        self._prev_move_pos = None
        self._prev_tilt     = None
        self._prev_spread   = None
        self.in_scale_mode  = False


    def enter_scale_mode(self):
        self.in_scale_mode = True
        self._prev_spread  = None


    def exit_scale_mode(self):
        self.in_scale_mode = False
        self._prev_spread  = None



    def update_move(self, finger_pos_2d, panel_w, panel_h):

        if self.active_object is None:
            return

        sx =  (finger_pos_2d[0] - panel_w / 2.0)
        sy = -(finger_pos_2d[1] - panel_h / 2.0)

        if self._prev_move_pos is not None:
            dx = (sx - self._prev_move_pos[0]) * self.move_sensitivity
            dy = (sy - self._prev_move_pos[1]) * self.move_sensitivity

            self.active_object.position[0] += dx
            self.active_object.position[1] += dy

            if self.snapping:
                self.snapping.propagate_move(self.active_object, dx, dy)

        self._prev_move_pos = (sx, sy)


    def stop_move(self):
        self._prev_move_pos = None



    def update_rotate_free(self, tilt_vector):

        if self.active_object is None:
            return

        tx, ty = tilt_vector

        if self._prev_tilt is not None:
            ptx, pty    = self._prev_tilt
            delta_x = (ty - pty) * self.rotate_sensitivity
            delta_y = (tx - ptx) * self.rotate_sensitivity

            self.active_object.rotation[0] += delta_x
            self.active_object.rotation[1] += delta_y

            if self.snapping:
                self.snapping.propagate_rotate(self.active_object, delta_x, delta_y)

        self._prev_tilt = (tx, ty)


    def stop_rotate(self):
        self._prev_tilt = None



    def update_scale_peace(self, spread):

        if self.active_object is None or not self.in_scale_mode:
            return

        if self._prev_spread is not None:
            delta = (spread - self._prev_spread) * self.scale_sensitivity
            cur   = float(self.active_object.scale[0])
            new_s = max(self.scale_min, min(self.scale_max, cur + delta))

            self.active_object.scale = np.array(
                [new_s, new_s, new_s], dtype=np.float32
            )

            if self.snapping:
                self.snapping.propagate_scale(self.active_object, new_s)

        self._prev_spread = spread


    def stop_scale(self):
        self._prev_spread  = None
        self.in_scale_mode = False



    def update_depth(self, swipe_direction):

        if self.active_object is None:
            return

        step = 20.0
        if swipe_direction == 'UP':
            dz = step
        elif swipe_direction == 'DOWN':
            dz = -step
        else:
            return

        self.active_object.position[2] += dz

        if self.snapping:
            self.snapping.propagate_depth(self.active_object, dz)

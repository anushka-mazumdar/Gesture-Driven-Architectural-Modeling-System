

import numpy as np


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]


class HandMesh:

    def __init__(self):

        # 🔥 Required for system compatibility
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.selected    = False
        self.highlighted = False

        self.kind = 'hand'

        # actual data
        self.landmarks = None
        self._prev = None


    def update_from_landmarks(self, landmarks_2d, panel_w, panel_h):

        if landmarks_2d is None or len(landmarks_2d) != 21:
            return

        pts = []
        scale = 0.18

        for lm in landmarks_2d:

            if len(lm) == 3:
                x, y, z = lm
            else:
                x, y = lm
                z = 0.0

            # normalize
            nx = x / panel_w
            ny = y / panel_h

            # map to world
            wx = (nx - 0.5) * panel_w * scale
            wy = (0.5 - ny) * panel_h * scale
            wz = -z * 120.0   # reduced depth (stable)

            # position offset (bottom-right)
            wx += panel_w * 0.25
            wy -= panel_h * 0.25

            pts.append([wx, wy, wz])

        pts = np.array(pts, dtype=np.float32)

        
        if self._prev is None:
            self._prev = pts
            self.landmarks = pts
            return

        alpha = 0.75
        smoothed = alpha * self._prev + (1 - alpha) * pts

        # update AFTER smoothing
        self._prev = smoothed

        

        self.landmarks = smoothed

        self.position[:] = 0
        self.rotation[:] = 0
        self.scale[:] = 1


    def get_bounds(self):
        return None
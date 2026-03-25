import math
import numpy as np


class Shape2DClassifier:

    def __init__(self, angle_threshold=35, min_corner_dist=15):
        self.angle_threshold  = angle_threshold
        self.min_corner_dist  = min_corner_dist


    def classify(self, stroke_points):
        """
        Returns a solid type string:
            'sphere'        — circle / ellipse
            'tetrahedron'   — triangle
            'cuboid'        — rectangle / square
            'prism_N'       — N-sided polygon (N >= 5)
        """
        pts = stroke_points

        if len(pts) < 6:
            return 'sphere'

        if self._is_circular(pts):
            return 'sphere'

        corners = self._detect_corners(pts)
        n       = len(corners)

        if n <= 2:
            return 'sphere'
        elif n == 3:
            return 'tetrahedron'
        elif n == 4:
            return 'cuboid'
        else:
            return f'prism_{n}'


    def _is_circular(self, pts):

        arr    = np.array(pts, dtype=np.float32)
        centre = arr.mean(axis=0)
        dists  = np.linalg.norm(arr - centre, axis=1)
        mean_r = dists.mean()

        if mean_r < 1:
            return False

        variance = dists.std() / mean_r
        return variance < 0.28


    def _detect_corners(self, pts):

        corners = []

        for i in range(1, len(pts) - 1):

            a1 = math.atan2(pts[i][1]   - pts[i-1][1],
                            pts[i][0]   - pts[i-1][0])
            a2 = math.atan2(pts[i+1][1] - pts[i][1],
                            pts[i+1][0] - pts[i][0])

            diff = abs(math.degrees(a2 - a1))
            if diff > 180:
                diff = 360 - diff

            if diff > self.angle_threshold:

                if corners:
                    spacing = math.hypot(
                        pts[i][0] - corners[-1][0],
                        pts[i][1] - corners[-1][1]
                    )
                    if spacing < self.min_corner_dist:
                        continue

                corners.append(pts[i])

        return corners
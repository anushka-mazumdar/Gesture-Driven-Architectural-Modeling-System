import math


class ClosureDetector:
    """Stage: open/closed determination + corner-snapping.

    A stroke counts as closed when the gap between its first and last point
    is small enough to "snap" the loop shut.
    """

    def __init__(self, snap_threshold=50):
        self.snap_threshold = snap_threshold

    def detect(self, points):
        if len(points) < 2:
            return False
        gap = math.hypot(points[-1][0] - points[0][0],
                         points[-1][1] - points[0][1])
        return gap <= self.snap_threshold

    def snap(self, points):
        """Return (points, closed). When closed, seal the gap by appending
        the start point so downstream contour analysis (corner counting,
        polygon approximation) sees a genuine closed loop instead of a
        polyline with a stray gap."""
        closed = self.detect(points)
        pts = list(points)
        if closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts, closed
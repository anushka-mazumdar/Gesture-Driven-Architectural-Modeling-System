import math


class StrokeProcessor:

    def __init__(self, min_dist=4, smoothing=0.4,
                 min_points=5, snap_threshold=30):

        self.min_dist       = min_dist
        self.smoothing      = smoothing
        self.min_points     = min_points
        self.snap_threshold = snap_threshold

        self.current_stroke    = []
        self.completed_strokes = []
        self._prev_smooth      = None
        self._drawing          = False


    def start_stroke(self):
        self.current_stroke = []
        self._prev_smooth   = None
        self._drawing       = True


    def add_point(self, raw_point):

        if not self._drawing:
            return

        if self._prev_smooth is None:
            self._prev_smooth = raw_point
            self.current_stroke.append(raw_point)
            return

        sx = self.smoothing * raw_point[0] + (1 - self.smoothing) * self._prev_smooth[0]
        sy = self.smoothing * raw_point[1] + (1 - self.smoothing) * self._prev_smooth[1]
        smooth = (int(sx), int(sy))

        dist = math.hypot(smooth[0] - self._prev_smooth[0],
                          smooth[1] - self._prev_smooth[1])

        if dist < self.min_dist:
            return

        self._prev_smooth = smooth
        self.current_stroke.append(smooth)


    def finish_stroke(self):

        self._drawing = False

        if len(self.current_stroke) < self.min_points:
            self.current_stroke = []
            self._prev_smooth   = None
            return None, False

        pts    = list(self.current_stroke)
        closed = False

        # check if start and end are close enough to snap shut
        gap = math.hypot(pts[-1][0] - pts[0][0],
                         pts[-1][1] - pts[0][1])

        if gap <= self.snap_threshold:
            pts.append(pts[0])   # close the loop
            closed = True

        self.completed_strokes.append(pts)
        self.current_stroke = []
        self._prev_smooth   = None

        return pts, closed


    def is_drawing(self):
        return self._drawing


    def get_current_stroke(self):
        return list(self.current_stroke)
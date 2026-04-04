import math
import time


class StrokeProcessor:

    def __init__(self, min_dist=6, smoothing=0.5,
                 min_points=5, snap_threshold=40,
                 resume_window=1.0, exit_buffer=6):

        self.min_dist      = min_dist
        self.smoothing     = smoothing
        self.min_points    = min_points
        self.snap_threshold= snap_threshold
        self.resume_window = resume_window
        self.exit_buffer   = exit_buffer   # points to trim on pause

        self.current_stroke    = []
        self.completed_strokes = []
        self._prev_smooth      = None
        self._drawing          = False
        self._paused           = False
        self._pause_start      = None


    def start_stroke(self):
        self.current_stroke = []
        self._prev_smooth   = None
        self._drawing       = True
        self._paused        = False
        self._pause_start   = None


    def resume_stroke(self):
        self._drawing     = True
        self._paused      = False
        self._pause_start = None
        self._prev_smooth = None


    def pause_stroke(self):
        """
        Index folded — trim exit buffer points to remove
        unwanted lines drawn while hand was transitioning,
        then start pause timer.
        """
        self._drawing = False
        self._paused  = True

        # trim last N points — drawn while finger was folding
        if len(self.current_stroke) > self.exit_buffer:
            self.current_stroke = self.current_stroke[:-self.exit_buffer]

        self._pause_start = time.time()


    def pause_expired(self):
        if not self._paused or self._pause_start is None:
            return False
        return (time.time() - self._pause_start) >= self.resume_window


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

        self._drawing     = False
        self._paused      = False
        self._pause_start = None

        if len(self.current_stroke) < self.min_points:
            self.current_stroke = []
            self._prev_smooth   = None
            return None, False

        pts    = list(self.current_stroke)
        closed = False

        gap = math.hypot(pts[-1][0] - pts[0][0],
                         pts[-1][1] - pts[0][1])
        if gap <= self.snap_threshold:
            pts.append(pts[0])
            closed = True

        self.completed_strokes.append(pts)
        self.current_stroke = []
        self._prev_smooth   = None

        return pts, closed


    def is_drawing(self):
        return self._drawing

    def is_paused(self):
        return self._paused

    def has_points(self):
        return len(self.current_stroke) > 0

    def get_current_stroke(self):
        return list(self.current_stroke)
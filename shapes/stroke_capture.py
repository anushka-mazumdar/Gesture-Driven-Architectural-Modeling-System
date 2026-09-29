import math
import time
from dataclasses import dataclass, field


@dataclass
class StrokeRecord:
    """A completed (or in-progress) stroke with its raw data preserved.

    Raw unfiltered points + per-point timestamps are kept so any later
    stage (analysis, classification, re-fit) can re-run without needing to
    re-capture. `points` is the smoothed/decimated version used for meshing.
    The remaining fields are filled by StrokePipeline.convert().
    """
    raw_points: list = field(default_factory=list)     # every cursor point fed to add_point
    points: list = field(default_factory=list)         # smoothed, min-distance-decimated
    timestamps: list = field(default_factory=list)     # capture-clock time per raw point
    start_time: float = 0.0
    end_time: float = 0.0

    closed: bool = None
    features: object = None       # StrokeFeatures
    shape_class: object = None    # ShapeClass
    recommendation: object = None # MeshRecommendation


class StrokeCapture:
    """Stage 1 of the stroke pipeline: raw point capture + session lifecycle.

    Deliberately does NOT detect open/closed or analyse the stroke — that is
    the job of ClosureDetector / StrokeAnalysis. finish() returns the raw
    StrokeRecord untouched by interpretation.
    """

    def __init__(self, min_dist=5, smoothing=0.68,
                 min_points=5, exit_buffer=6,
                 resume_window=1.0):

        self.min_dist      = min_dist
        self.smoothing     = smoothing
        self.min_points    = min_points
        self.exit_buffer   = exit_buffer   # points to trim on pause
        self.resume_window = resume_window

        self.current    = StrokeRecord()
        self.completed  = []               # finished StrokeRecords (raw preserved)
        self._prev_smooth = None
        self._prev_raw    = None
        self._drawing     = False
        self._paused      = False
        self._pause_start = None
        self._clock       = 0.0            # monotonically increasing point time

    # --- lifecycle ---------------------------------------------------

    def start_stroke(self):
        self.current    = StrokeRecord(start_time=time.time())
        self._prev_smooth = None
        self._prev_raw    = None
        self._drawing     = True
        self._paused      = False
        self._pause_start = None

    def resume_stroke(self):
        self._drawing     = True
        self._paused      = False
        self._pause_start = None
        self._prev_smooth = None

    def pause_stroke(self):
        """Index folded — trim exit-buffer points from both smooth and raw
        lists (lines drawn while the finger was folding), then start the
        pause timer."""
        self._drawing = False
        self._paused  = True

        if len(self.current.points) > self.exit_buffer:
            keep_s = -self.exit_buffer
            keep_r = max(1, len(self.current.raw_points) - self.exit_buffer)
            self.current.points      = self.current.points[:keep_s]
            self.current.raw_points  = self.current.raw_points[:keep_r]
            self.current.timestamps  = self.current.timestamps[:keep_r]

        self._pause_start = time.time()

    def pause_expired(self):
        if not self._paused or self._pause_start is None:
            return False
        return (time.time() - self._pause_start) >= self.resume_window

    def finish(self):
        """Close the session and return the StrokeRecord (or None if too
        few points). No interpretation — closure detection is downstream."""
        self._drawing     = False
        self._paused      = False
        self._pause_start = None

        if len(self.current.points) < self.min_points:
            self._prev_smooth = None
            self._prev_raw    = None
            return None

        self.current.end_time = time.time()
        record = self.current
        self.completed.append(record)

        self._prev_smooth = None
        self._prev_raw    = None
        return record

    # --- point capture ----------------------------------------------

    def add_point(self, raw_point):
        if not self._drawing:
            return

        self._clock += 1.0
        self.current.raw_points.append(raw_point)
        self.current.timestamps.append(self._clock)

        if self._prev_smooth is None:
            self._prev_smooth = raw_point
            self._prev_raw    = raw_point
            self.current.points.append(raw_point)
            return

        sx = self.smoothing * self._prev_smooth[0] + (1 - self.smoothing) * raw_point[0]
        sy = self.smoothing * self._prev_smooth[1] + (1 - self.smoothing) * raw_point[1]
        smooth = (int(sx), int(sy))

        dist = math.hypot(smooth[0] - self._prev_smooth[0],
                          smooth[1] - self._prev_smooth[1])

        if dist < self.min_dist:
            return

        self._prev_smooth = smooth
        self._prev_raw    = raw_point
        self.current.points.append(smooth)

    # --- state -------------------------------------------------------

    def is_drawing(self):
        return self._drawing

    def is_paused(self):
        return self._paused

    def has_points(self):
        return len(self.current.points) > 0

    def get_current(self):
        return list(self.current.points)
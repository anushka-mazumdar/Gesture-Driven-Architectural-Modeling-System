import math
import statistics


DEFAULT_SNAP_THRESHOLD = 48
MAX_SNAP_THRESHOLD = 48
RELATIVE_SNAP_THRESHOLD = 0.045
MIN_SNAP_THRESHOLD = 5.0
ENDPOINT_SAMPLE_MULTIPLIER = 1.5
MIN_NORMALIZED_ENCLOSED_AREA = 0.005


class ClosureDetector:
    """Decide whether a stroke has enough geometric evidence to be sealed.

    Endpoint distance is constrained by the drawing's overall scale and by
    the local point spacing at both ends. A skipped, visibly open section is
    therefore not closed merely because it falls under a large pixel cutoff.
    """

    def __init__(self, snap_threshold=DEFAULT_SNAP_THRESHOLD):
        try:
            threshold = float(snap_threshold)
        except (TypeError, ValueError):
            threshold = DEFAULT_SNAP_THRESHOLD
        self.snap_threshold = (min(MAX_SNAP_THRESHOLD, threshold)
                               if math.isfinite(threshold) and threshold >= 0
                               else DEFAULT_SNAP_THRESHOLD)

    @staticmethod
    def _geometry(points):
        try:
            pts = [(float(point[0]), float(point[1])) for point in points]
            if len(pts) < 4 or not all(math.isfinite(v) for p in pts for v in p):
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if diagonal <= 1e-9:
                return None
            gap = math.dist(pts[0], pts[-1])
            area = abs(sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(pts, pts[1:] + pts[:1])
            )) / 2.0
            area_ratio = area / (diagonal * diagonal)
            endpoint_steps = [math.dist(a, b) for a, b in zip(pts[:4], pts[1:4])]
            endpoint_steps += [math.dist(a, b) for a, b in zip(pts[-4:-1], pts[-3:])]
            endpoint_steps = [step for step in endpoint_steps if step > 1e-9]
            if not endpoint_steps:
                return None
            local_spacing = statistics.median(endpoint_steps)
            return pts, diagonal, gap, local_spacing, area_ratio
        except (TypeError, ValueError, IndexError, OverflowError):
            return None

    def threshold_for(self, points):
        """Return the effective pixel threshold after scale and sampling caps."""
        geometry = self._geometry(points)
        if geometry is None:
            return MIN_SNAP_THRESHOLD
        _, diagonal, _, local_spacing, _ = geometry
        scale_limit = max(MIN_SNAP_THRESHOLD, diagonal * RELATIVE_SNAP_THRESHOLD)
        sampling_limit = local_spacing * ENDPOINT_SAMPLE_MULTIPLIER
        return max(0.0, min(self.snap_threshold, scale_limit, sampling_limit))

    def detect(self, points):
        geometry = self._geometry(points)
        if geometry is None:
            return False
        _, diagonal, gap, local_spacing, area_ratio = geometry
        # Keep both normalized and sample-relative evidence explicit here.
        # threshold_for also applies the absolute cap and safeguards for size.
        return (
            gap <= self.threshold_for(points)
            and gap / diagonal <= RELATIVE_SNAP_THRESHOLD
            and gap <= local_spacing * ENDPOINT_SAMPLE_MULTIPLIER
            and area_ratio >= MIN_NORMALIZED_ENCLOSED_AREA
        )

    def snap(self, points):
        """Seal only a stroke that passed the geometric closure checks."""
        closed = self.detect(points)
        pts = list(points)
        if closed and pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts, closed

import math
from dataclasses import dataclass, field


@dataclass
class StrokeFeatures:
    """Geometric features of a stroke, computed by StrokeAnalysis.

    Bounds, perimeter, area, aspect_ratio and roundness stay meaningful on
    raw pixel points. Convexity / solidity / polygon approximation are
    contour descriptors that are scale-invariant — for maximum robustness
    prefer to compute them on normalized/resampled points (see
    stroke_normalization.preprocess and StrokePipeline).
    """
    points: list = field(default_factory=list)   # the exact points analyze() was given
    length: float = 0.0          # path length (sum of segment lengths)
    bbox: tuple = field(default_factory=lambda: (0, 0, 0, 0))  # (xmin, ymin, xmax, ymax)
    width: float = 0.0
    height: float = 0.0
    aspect_ratio: float = 1.0    # width / height (>= 1)
    endpoint_gap: float = 0.0    # distance first -> last point
    perimeter: float = 0.0       # closed loop length (gap-sealed)
    area: float = 0.0            # shoelace area (absolute)
    roundness: float = 0.0       # circularity = 4*pi*area / perimeter^2 (1.0 == circle)
    bbox_fill: float = 0.0       # area / (width*height)     (1.0 == axis-aligned quad)
    straightness: float = 1.0    # endpoint_gap / length     (1.0 == straight line)
    center: tuple = field(default_factory=lambda: (0.0, 0.0))  # bbox centre
    centroid: tuple = field(default_factory=lambda: (0.0, 0.0))  # geometric centroid
    point_count: int = 0

    # -- contour descriptors (Phase 1.2) ------------------------------
    convex_hull: list = field(default_factory=list)      # CCW hull vertices
    hull_area: float = 0.0
    hull_perimeter: float = 0.0
    convexity: float = 1.0          # hull_perimeter / perimeter  (<=1; 1 == convex)
    solidity: float = 1.0           # area / hull_area            (<=1; 1 == convex fill)
    approx_points: list = field(default_factory=list)    # Douglas-Peucker key vertices
    approx_vertices: int = 0
    approx_epsilon: float = 0.0


def analyze(points, approx_rel=0.035, area_rel=0.02):
    """Compute StrokeFeatures for a list of (x, y) points.

    `approx_rel` is kept for API compatibility but no longer drives corner
    extraction (see `area_rel`). `area_rel` is the Visvalingam-Whyatt corner
    significance threshold, as a fraction of the bounding-box diagonal
    squared — resolution-independent like `approx_rel` was, but scored by
    enclosed triangle area rather than perpendicular distance. Area-based
    scoring is far less sensitive to hand-tracking jitter: a single noisy
    point next to two real edge points still encloses a tiny triangle, so it
    gets pruned, whereas Douglas-Peucker's distance-to-chord test flags it as
    a "corner" the moment the jitter exceeds epsilon.
    """
    feats = StrokeFeatures(point_count=len(points))
    if len(points) < 2:
        return feats

    feats.points = [tuple(p) for p in points]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    feats.bbox = (min(xs), min(ys), max(xs), max(ys))
    feats.width  = feats.bbox[2] - feats.bbox[0]
    feats.height = feats.bbox[3] - feats.bbox[1]
    feats.center = ((feats.bbox[0] + feats.bbox[2]) / 2.0,
                    (feats.bbox[1] + feats.bbox[3]) / 2.0)
    smaller = max(min(feats.width, feats.height), 1e-9)
    feats.aspect_ratio = max(feats.width, feats.height) / smaller

    # path
    seg_lengths = [
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    ]
    feats.length = sum(seg_lengths)
    feats.endpoint_gap = math.hypot(points[-1][0] - points[0][0],
                                    points[-1][1] - points[0][1])
    if feats.length > 1e-9:
        feats.straightness = min(feats.endpoint_gap / feats.length, 1.0)

    # closed-loop perimeter: seal the first->last gap
    feats.perimeter = feats.length + feats.endpoint_gap

    signed_area = _signed_area(points)
    feats.area = abs(signed_area)
    feats.centroid = _polygon_centroid(points, signed_area)
    feats.bbox_fill = (min(feats.area / (feats.width * feats.height), 1.0)
                       if feats.width * feats.height > 1e-9 else 0.0)
    feats.roundness = (4.0 * math.pi * feats.area / (feats.perimeter ** 2)
                       if feats.perimeter > 1e-9 else 0.0)

    # -- contour descriptors -----------------------------------------
    hull = _convex_hull(points)
    feats.convex_hull = [tuple(h) for h in hull]
    feats.hull_area = abs(_signed_area(hull))
    feats.hull_perimeter = _perimeter(hull)
    if feats.perimeter > 1e-9:
        feats.convexity = feats.hull_perimeter / feats.perimeter
    if feats.hull_area > 1e-9:
        feats.solidity = min(feats.area / feats.hull_area, 1.0)

    diag = math.hypot(feats.width, feats.height)
    feats.approx_epsilon = approx_rel * (diag if diag > 1e-9 else 1.0)
    closed_loop = len(points) >= 3 and points[0] == points[-1]
    feats.approx_points = _visvalingam_whyatt(
        points, area_rel * (diag * diag if diag > 1e-9 else 1.0), closed_loop)
    feats.approx_vertices = len(feats.approx_points)

    return feats


def _signed_area(points):
    n = len(points)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return 0.5 * s


def _polygon_centroid(points, signed_area):
    n = len(points)
    if n < 3 or abs(signed_area) < 1e-9:
        if not n:
            return (0.0, 0.0)
        return (sum(p[0] for p in points) / n,
                sum(p[1] for p in points) / n)
    cx = cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = points[i][0] * points[j][1] - points[j][0] * points[i][1]
        cx += (points[i][0] + points[j][0]) * cross
        cy += (points[i][1] + points[j][1]) * cross
    return (cx / (6 * signed_area), cy / (6 * signed_area))


def _perimeter(points):
    if len(points) < 2:
        return 0.0
    p = sum(math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points, points[1:]))
    return p + math.hypot(points[-1][0] - points[0][0],
                          points[-1][1] - points[0][1])


def _convex_hull(points):
    """Andrew's monotone chain. Returns CCW hull vertices (unique points)."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _triangle_area(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def _vw_reduce(points, closed, min_area=None, target_count=None):
    """Shared Visvalingam-Whyatt-style reduction core.

    Repeatedly drops the point that contributes the least enclosed-triangle
    area (with its current neighbours). Unlike Douglas-Peucker's
    chord-distance test, a single noisy point still scores low here (its
    triangle with its two neighbours stays tiny even if it deviates from
    the chord), so jittery hand-tracking input doesn't get misread as extra
    corners.

    Exactly one of `min_area` (stop once every remaining vertex clears this
    area) or `target_count` (stop once exactly this many vertices remain,
    regardless of area) drives the stopping rule. Returns
    (points, removed_area) — removed_area is the sum of the enclosed
    triangle areas of every vertex pruned to reach the result, i.e. how
    much the simplification actually altered the shape. A caller forcing a
    smaller target_count than the data naturally supports can use this to
    tell "discarded near-collinear noise" from "discarded a real corner".

    `points` may carry a duplicate closing point (first == last) for a
    sealed loop; that duplicate is stripped before simplifying and
    reattached on the way out. `closed` controls whether the two endpoints
    are themselves eligible for removal (a closed loop has no fixed
    endpoint; an open polyline keeps its two ends fixed).
    """
    pts = [tuple(p) for p in points]
    dup_closed = closed and len(pts) >= 2 and pts[0] == pts[-1]
    if dup_closed:
        pts = pts[:-1]

    n = len(pts)
    if n < 3:
        return [tuple(p) for p in points], 0.0

    min_keep = 3 if closed else 2
    if target_count is not None:
        min_keep = max(min_keep, target_count)
    alive = list(range(n))
    removed_area = 0.0

    while len(alive) > min_keep:
        m = len(alive)
        best_area = None
        best_pos = None
        span = range(m) if closed else range(1, m - 1)
        for pos in span:
            prev_i = alive[pos - 1]
            next_i = alive[(pos + 1) % m]
            area = _triangle_area(pts[prev_i], pts[alive[pos]], pts[next_i])
            if best_area is None or area < best_area:
                best_area = area
                best_pos = pos
        if best_pos is None:
            break
        if target_count is None and best_area >= min_area:
            break
        del alive[best_pos]
        removed_area += best_area

    result = [pts[i] for i in alive]
    if dup_closed:
        result.append(result[0])
    return result, removed_area


def _visvalingam_whyatt(points, min_area, closed):
    """Visvalingam-Whyatt simplification — the key vertices of the shape."""
    result, _ = _vw_reduce(points, closed, min_area=min_area)
    return result


def simplify_to_count(points, k, closed):
    """Force an already-simplified corner list down to exactly `k`
    vertices (or fewer, if it's already smaller), continuing the same
    least-area-first removal past the area-significance floor.

    Used by the classifier to test "does this contour look like a clean
    k-gon" even when the natural corner count came out slightly higher —
    air-drawn strokes routinely pick up one stray corner along an
    otherwise-straight edge. Returns (points, removed_area); a small
    removed_area relative to the shape's total area means the discarded
    vertices were near-collinear noise (the forced fit is trustworthy), a
    large one means real corners were merged away (distrust it).
    """
    return _vw_reduce(points, closed, target_count=k)
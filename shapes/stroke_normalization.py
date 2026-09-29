import math


def resample(points, n=64):
    """Resample a polyline to exactly `n` points, equidistant along its arc
    length. Fixes the uneven point density that results from hand speed
    varying during capture, so later measures don't depend on stroke tempo.

    Returns float-coordinate points. Degenerate (single-point) strokes are
    replicated; short strokes are upsampled, long strokes downsampled.
    """
    pts = list(points)
    if len(pts) < 2 or n < 2:
        return list(pts)

    dists = [0.0]
    for p, q in zip(pts, pts[1:]):
        dists.append(dists[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    total = dists[-1]

    if total <= 1e-9:
        return [tuple(pts[0]) for _ in range(n)]

    out = []
    step = total / (n - 1)
    idx = 1
    for i in range(n):
        target = i * step
        while idx < len(dists) - 1 and dists[idx] < target:
            idx += 1
        seg_len = dists[idx] - dists[idx - 1]
        t = 0.0 if seg_len <= 1e-9 else (target - dists[idx - 1]) / seg_len
        t = max(0.0, min(1.0, t))
        x = pts[idx - 1][0] + t * (pts[idx][0] - pts[idx - 1][0])
        y = pts[idx - 1][1] + t * (pts[idx][1] - pts[idx - 1][1])
        out.append((x, y))

    out[0] = tuple(pts[0])
    out[-1] = tuple(pts[-1])
    return out


def centroid(points):
    """Simple vertex mean centroid (robust for open and closed strokes)."""
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def normalize(points, target_span=1.0):
    """Centroid-centre then scale so max(width, height) == target_span.

    Translation- and scale-invariant: two captures of the same shape at
    different sizes/positions produce identical normalized geometry. Aspect
    ratio is preserved (uniform scaling).
    """
    if not points:
        return []
    cx, cy = centroid(points)
    xs = [p[0] - cx for p in points]
    ys = [p[1] - cy for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    s = target_span / span
    return [(x * s, y * s) for x, y in zip(xs, ys)]


def preprocess(points, n=64, target_span=1.0):
    """Canonical form for analysis: resample to uniform density, then
    normalize. Use this when comparing strokes across speed/size/location —
    analysis of these points yields robust, resolution-independent features.
    """
    return normalize(resample(points, n), target_span)
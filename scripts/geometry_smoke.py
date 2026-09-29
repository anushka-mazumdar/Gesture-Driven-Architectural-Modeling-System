"""
Phase 1.2 smoke test — Geometry preprocessing & contour features.

Exercises the new normalization/resampling and the extended StrokeFeatures
(centroid, convex hull, solidity, convexity, Douglas-Peucker polygon
approximation). Goes through StrokePipeline so we also prove classification
is robust to resampling (density) and normalization (scale/position).

Run:  python scripts/geometry_smoke.py
Exit code 0 means all checks passed.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapes.stroke_normalization import resample, normalize, preprocess
from shapes.stroke_analysis       import analyze
from shapes.stroke_pipeline       import StrokePipeline
from shapes.shape_3d_factory      import Shape3DFactory


def circle(cx, cy, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


# near-rectangle (float coords -> clean ratios)
RECT = [(100, 100), (100, 140), (100, 180), (140, 180), (180, 180),
        (180, 140), (180, 100), (140, 100), (102, 100)]
# concave L-shape: solidity < 1 (area < hull area), convexity > 1
L = [(40, 40), (180, 40), (180, 100), (110, 100), (110, 180), (40, 180)]
# straight diagonal
LINE = [(i * 2.0, i * 2.0) for i in range(60)]

checks = []


def check(name, condition, detail=""):
    ok = bool(condition)
    checks.append((name, ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def main():
    # 1. Resampling -------------------------------------------------------
    r5 = resample(LINE, 10)
    check("resample produces exactly n points", len(r5) == 10, f"n={len(r5)}")
    check("resample preserves endpoints",
          r5[0] == LINE[0] and r5[-1] == LINE[-1],
          f"{r5[0]} -> {LINE[0]} | {r5[-1]} -> {LINE[-1]}")
    segs = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(r5, r5[1:])]
    spread = max(segs) - min(segs)
    check("resample spacing is (nearly) uniform", spread < 1e-6,
          f"spread={spread:.2e}")
    check("resample upsamples short strokes",
          len(resample([(0, 0), (10, 10)], 64)) == 64)

    # 2. Normalization (scale & translation invariance) -------------------
    big   = [(x * 3.0 + 200, y * 3.0 + 50) for x, y in RECT]
    small = [(x * 0.5 - 400, y * 0.5 + 300) for x, y in RECT]
    n_orig = normalize(RECT)
    n_big  = normalize(big)
    n_small = normalize(small)
    same = all(abs(a - b) < 1e-9 for p, q in zip(n_orig, n_big)
               for a, b in zip(p, q))
    same_s = all(abs(a - b) < 1e-9 for p, q in zip(n_orig, n_small)
                 for a, b in zip(p, q))
    check("normalize invariant to scale+translation", same and same_s)
    xspan = max(p[0] for p in n_orig) - min(p[0] for p in n_orig)
    yspan = max(p[1] for p in n_orig) - min(p[1] for p in n_orig)
    check("normalize sets span to target", max(xspan, yspan) == 1.0,
          f"span=({xspan:.3f},{yspan:.3f})")
    n0 = normalize(RECT)
    gx = sum(p[0] for p in n0) / len(n0)
    gy = sum(p[1] for p in n0) / len(n0)
    check("normalize centres centroid on (0,0)", abs(gx) < 1e-9 and abs(gy) < 1e-9,
          f"centroid=({gx},{gy})")

    # 3. Contour features (raw and normalized) ----------------------------
    fr = analyze(RECT)          # pixel coords
    fn = analyze(preprocess(RECT))  # resampled + normalized
    check("features: bounds correct",
          fr.bbox == (100, 100, 180, 180), f"bbox={fr.bbox}")
    check("features: centroid inside bbox",
          100 < fr.centroid[0] < 180 and 100 < fr.centroid[1] < 180,
          f"centroid={fr.centroid}")
    check("features: aspect_ratio >= 1", fr.aspect_ratio >= 1.0,
          f"ar={fr.aspect_ratio:.3f}")
    check("features: rectangle near-solid (solidity ~ 1)",
          fr.solidity > 0.95, f"solidity={fr.solidity:.3f}")
    check("features: rectangle near-convex (convexity ~ 1)",
          fr.convexity < 1.05, f"convexity={fr.convexity:.3f}")

    # concave L-shape -> lower solidity AND lower convexity (hull_perim /
    # contour_perim < 1) than the rectangle
    fL = analyze(L)
    check("features: concave shape is less solid than rectangle",
          fL.solidity < fr.solidity, f"L solidity={fL.solidity:.3f} vs rect {fr.solidity:.3f}")
    check("features: concave shape convexity < 1 and < rect's",
          fL.convexity < 0.995 and fL.convexity < fr.convexity,
          f"L convexity={fL.convexity:.3f} vs rect {fr.convexity:.3f}")

    # circularity on circle
    fc = analyze(circle(320, 240, 70))
    check("features: circle circularity (roundness) > 0.9",
          fc.roundness > 0.9, f"roundness={fc.roundness:.3f}")

    # 4. Polygon approximation (Douglas-Peucker) --------------------------
    fa = analyze(preprocess(RECT))
    check("polygon approx: rectangle -> ~4 vertices",
          4 <= fa.approx_vertices <= 6, f"vertices={fa.approx_vertices} ({fa.approx_points})")
    fc_approx = analyze(preprocess(circle(0, 0, 1.0), 64))
    check("polygon approx: circle down-sampled well below 64",
          4 <= fc_approx.approx_vertices <= 20,
          f"vertices={fc_approx.approx_vertices}")
    check("polygon approx: preprocess resamples to 64 by default",
          len(preprocess(LINE, 64)) == 64)

    # 5. Pipeline robustness ---------------------------------------------
    pipe = StrokePipeline(factory=Shape3DFactory(panel_width=640, panel_height=480))
    closed, f_raw, f_norm, cls, rec = pipe.classify_stroke(RECT)
    # bbox is 80x80 (a square), so the richer classifier now correctly
    # names it "square" rather than the old generic "rectangle"
    check("pipeline: classifies rectangle on normalized features",
          closed and cls.kind == "square", f"kind={cls.kind}")
    _, _, _, cls_l, _ = pipe.classify_stroke(LINE)
    check("pipeline: classifies straight open as line",
          cls_l.kind == "line", f"kind={cls_l.kind}")
    # recommendation sizing still uses PHYSICAL pixels, not normalized units
    _, _, _, cls_c, rec_c = pipe.classify_stroke(circle(320, 240, 70))
    check("pipeline: circle -> closed family -> polygon recommendation",
          rec_c.kind == "polygon" and cls_c.kind == "circle",
          f"kind={rec_c.kind}, cls={cls_c.kind}")
    check("pipeline: recommender keeps physical extrude (>= 20)",
          rec_c.extrude_depth >= 20.0, f"extrude={rec_c.extrude_depth:.1f}")

    failures = [name for name, ok in checks if not ok]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
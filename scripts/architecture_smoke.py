"""
Phase 1.1 smoke test — Architecture separation of the stroke pipeline.

Drives the staged pipeline (capture -> analysis -> closure -> classify ->
recommend -> build) that main.py uses, entirely offline (no webcam/GUI).
Proves each stage is separated and that raw stroke data is preserved.

Run:  python scripts/architecture_smoke.py
Exit code 0 means all checks passed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapes.stroke_capture    import StrokeCapture
from shapes.stroke_analysis   import analyze
from shapes.closure_detector  import ClosureDetector
from shapes.stroke_classifier import StrokeClassifier
from shapes.shape_recommender import ShapeRecommender
from shapes.stroke_pipeline   import StrokePipeline
from shapes.shape_3d_factory  import Shape3DFactory

PANEL_W, PANEL_H = 640, 480

# Synthetic strokes ----------------------------------------------------

def circle_points(cx, cy, r, n=48):
    import math
    return [(int(cx + r * math.cos(2 * math.pi * i / n)),
             int(cy + r * math.sin(2 * math.pi * i / n))) for i in range(n)]

# near-perfect rectangle: bbox_fill ~1, roundness ~0.69
RECT = [(100, 100), (100, 140), (100, 180), (140, 180), (180, 180),
        (180, 140), (180, 100), (140, 100), (105, 100)]
# sampled circle: roundness ~1, bbox_fill ~0.78
CIRC = circle_points(320, 240, 70)
# generic closed polygon (concave L-shape): low roundness -> polygon
POLY = [(40, 40), (180, 40), (180, 100), (110, 100), (110, 180), (40, 180)]
# straight open stroke
LINE = [(40, 200), (40 + 30, 200), (40 + 60, 200), (40 + 90, 200), (40 + 120, 200)]
# curved open stroke
CURVE = [(40, 300), (90, 260), (140, 250), (190, 265), (240, 300)]

checks = []


def check(name, condition, detail=""):
    ok = bool(condition)
    checks.append((name, ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def main():
    # 1. Capture preserves raw data --------------------------------------
    cap = StrokeCapture()
    cap.start_stroke()
    raw_fed = []
    for i in range(40):
        p = (10 + i * 3, 200 + i)      # steady diagonal
        raw_fed.append(p)
        cap.add_point(p)
    rec = cap.finish()
    check("capture finish returns a record", rec is not None)
    check("capture preserves raw points", rec.raw_points == raw_fed,
          f"raw={len(rec.raw_points)} fed={len(raw_fed)}")
    check("smoothed points kept separately & decimated",
          rec.points and rec.points != rec.raw_points and len(rec.points) <= len(rec.raw_points))
    check("timestamps recorded per raw point", len(rec.timestamps) == len(rec.raw_points))
    check("raw preserved in completed history",
          cap.completed and cap.completed[0] is rec)

    # 2. Closure detection -----------------------------------------------
    closure = ClosureDetector()
    check("closed rectangle -> True", closure.detect(RECT))
    check("closed circle -> True", closure.detect(CIRC))
    check("open line -> False", not closure.detect(LINE))
    check("open curve -> False", not closure.detect(CURVE))

    # 3. Analysis --------------------------------------------------------
    fr = analyze(RECT)
    fc = analyze(CIRC)
    fl = analyze(LINE)
    check("analysis: rectangle bbox_fill ~ 1", 0.9 < fr.bbox_fill <= 1.0,
          f"bbox_fill={fr.bbox_fill:.2f}")
    check("analysis: circle roundness > 0.85", fc.roundness > 0.85,
          f"roundness={fc.roundness:.2f}")
    check("analysis: line straightness > 0.98", fl.straightness > 0.98,
          f"straightness={fl.straightness:.2f}")
    check("analysis: line aspect ratio >= 1", fl.aspect_ratio >= 1.0)
    check("analysis: line has no polygon area", fl.area < 1e-6,
          f"area={fl.area:.2f}")

    # 4. Classification --------------------------------------------------
    # Classify on sealed (closure.snap'd) points, same as the real pipeline
    # feeds StrokeClassifier — a raw near-duplicate closing point would
    # otherwise read as a spurious extra corner.
    rect_sealed, _ = closure.snap(RECT)
    fr_sealed = analyze(rect_sealed)
    clf = StrokeClassifier()
    rc  = clf.classify(rect_sealed, fr_sealed, True)
    cc  = clf.classify(CIRC, fc, True)
    pc  = clf.classify(POLY, analyze(POLY), True)
    lc  = clf.classify(LINE, fl, False)
    cv  = clf.classify(CURVE, analyze(CURVE), False)
    # bbox is 80x80 (a square), so the richer classifier now correctly
    # names it "square" rather than the old generic "rectangle"
    check("classify: rectangle -> 'square'", rc.kind == "square",
          f"got {rc.kind}")
    check("classify: circle -> 'circle'", cc.kind == "circle", f"got {cc.kind}")
    check("classify: closed blob -> 'polygon'", pc.kind == "polygon",
          f"got {pc.kind}")
    check("classify: straight open -> 'line'", lc.kind == "line",
          f"got {lc.kind}")
    check("classify: curved open -> 'curve'", cv.kind == "curve",
          f"got {cv.kind}")
    check("classify: confidence in (0,1]", all(0 < c.confidence <= 1.0
                                               for c in (rc, cc, pc, lc, cv)))

    # 5. Recommendation --------------------------------------------------
    recommender = ShapeRecommender()
    rr = recommender.recommend(RECT, fr, rc)
    rl = recommender.recommend(LINE, fl, lc)
    check("recommend: closed family -> polygon", rr.kind == "polygon",
          f"got {rr.kind}")
    check("recommend: open family -> ribbon", rl.kind == "ribbon",
          f"got {rl.kind}")
    check("recommend: polygon extruded from bbox", rr.extrude_depth >= 20.0)
    check("recommend: ribbon thickness scales with length (min 12)",
          rl.thickness == max(fl.length * 0.06, 12.0),
          f"thickness={rl.thickness}")

    # 6. End-to-end pipeline ---------------------------------------------
    pipe = StrokePipeline(factory=Shape3DFactory(panel_width=PANEL_W,
                                                 panel_height=PANEL_H))
    cp = StrokeCapture()
    cp.start_stroke()
    for p in RECT:
        cp.add_point(p)
    crec = cp.finish()
    res = pipe.convert(crec) if crec else None
    check("pipeline.convert returns a result", res is not None)
    check("pipeline builds closed stroke -> polygon mesh",
          res.closed and res.mesh is not None and res.mesh.kind == "polygon",
          f"kind={getattr(res.mesh, 'kind', None)}")
    # open stroke through the pipeline
    op = StrokeCapture()
    op = StrokeCapture()
    op.start_stroke()
    for p in LINE:
        op.add_point(p)
    orec = op.finish()
    ores = pipe.convert(orec) if orec else None
    check("pipeline open stroke -> ribbon mesh",
          ores is not None and not ores.closed and ores.mesh is not None
          and ores.mesh.kind == "ribbon", f"kind={getattr(ores and ores.mesh, 'kind', None)}")
    check("pipeline preserves+enriches record metadata",
          crec.closed is True and crec.features is not None
          and crec.shape_class is not None and crec.recommendation is not None)

    # 7. Factory compatibility (regression guard) ------------------------
    factory = Shape3DFactory(panel_width=PANEL_W, panel_height=PANEL_H)
    poly = factory.create_from_stroke(RECT, closed=True)
    rib  = factory.create_from_stroke(LINE, closed=False)
    check("factory.create_from_stroke(closed=True) -> polygon",
          poly is not None and poly.kind == "polygon")
    check("factory.create_from_stroke(open) -> ribbon",
          rib is not None and rib.kind == "ribbon")
    # default closure detection, no hint
    auto = factory.create_from_stroke(CIRC)
    check("factory.create_from_stroke auto-closes circle -> polygon",
          auto is not None and auto.kind == "polygon")

    failures = [name for name, ok in checks if not ok]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
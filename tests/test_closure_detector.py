import unittest

from shapes.closure_detector import ClosureDetector
from shapes.stroke_classifier import ShapeClass
from shapes.stroke_capture import StrokeRecord
from shapes.stroke_capture import StrokeCapture
from shapes.stroke_pipeline import StrokePipeline


class _ClosedShapeClassifier:
    def __init__(self):
        self.received_closed = None

    def classify(self, points, features, closed):
        self.received_closed = closed
        return ShapeClass("polygon" if closed else "polyline", 1.0)


class _KindFactory:
    def build(self, recommendation, points):
        return type("BuiltMesh", (), {"kind": recommendation.kind})()


class ClosureDetectorTests(unittest.TestCase):
    def test_pause_trimming_keeps_a_near_start_endpoint_from_the_exit_buffer(self):
        capture = StrokeCapture(exit_buffer=3)
        capture.start_stroke()
        capture.current.points = [
            (0, 0), (100, 0), (100, 100), (0, 100),
            (12, 8), (130, 150), (145, 165), (160, 180),
        ]
        capture.current.raw_points = list(capture.current.points)
        capture.current.timestamps = list(range(len(capture.current.points)))

        capture.pause_stroke()

        self.assertEqual(capture.current.points[-1], (12, 8))
        self.assertTrue(ClosureDetector().detect(capture.current.points))

    def test_nearby_endpoints_snap_to_start_and_mark_stroke_closed(self):
        points = [(0, 0), (100, 0), (100, 100), (0, 100), (60, 0)]
        snapped, closed = ClosureDetector().snap(points)

        self.assertTrue(closed)
        self.assertEqual(snapped[:-1], points)
        self.assertEqual(snapped[-1], points[0])

    def test_gap_at_threshold_closes_but_gap_over_threshold_stays_open(self):
        detector = ClosureDetector(snap_threshold=80)
        at_limit = [(0, 0), (50, 0), (80, 0)]
        over_limit = [(0, 0), (50, 0), (81, 0)]

        self.assertTrue(detector.detect(at_limit))
        self.assertFalse(detector.detect(over_limit))
        snapped, closed = detector.snap(over_limit)
        self.assertFalse(closed)
        self.assertEqual(snapped, over_limit)

    def test_pipeline_uses_snapped_closure_for_closed_3d_build(self):
        classifier = _ClosedShapeClassifier()
        pipeline = StrokePipeline(classifier=classifier, factory=_KindFactory())
        points = [(0, 0), (100, 0), (100, 100), (0, 100), (60, 0)]

        result = pipeline.convert(StrokeRecord(raw_points=list(points), points=list(points)))

        self.assertTrue(result.closed)
        self.assertTrue(classifier.received_closed)
        self.assertEqual(result.shape_class.kind, "polygon")
        self.assertEqual(result.mesh.kind, "polygon")
        self.assertEqual(result.record.points[-1], points[0])


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from shapes.closure_detector import ClosureDetector
from shapes.stroke_capture import StrokeRecord, StrokeCapture
from shapes.stroke_classifier import ShapeClass, StrokeClassifier
from shapes.stroke_pipeline import StrokePipeline
from shapes.shape_3d_factory import Shape3DFactory


OPEN_LABELS = {"straight_line", "polyline", "arc", "curve", "freeform_path"}


def circle_stroke(radius=140, count=72):
    return [
        (320 + radius * math.cos(2 * math.pi * index / count),
         240 + radius * math.sin(2 * math.pi * index / count))
        for index in range(count + 1)
    ]


def rectangle_stroke(gap=4):
    points = []
    for start, end in (((150, 100), (450, 100)),
                       ((450, 100), (450, 300)),
                       ((450, 300), (150, 300)),
                       ((150, 300), (150, 100 + gap))):
        length = math.dist(start, end)
        steps = max(1, round(length / 10))
        points.extend((round(start[0] + (end[0] - start[0]) * i / steps),
                       round(start[1] + (end[1] - start[1]) * i / steps))
                      for i in range(steps))
    points.append((150, 100 + gap))
    return points


class RecordingClassifier(StrokeClassifier):
    def __init__(self):
        super().__init__(use_ml_model=False)
        self.closed_arguments = []

    def classify(self, points, features, closed):
        self.closed_arguments.append(closed)
        return super().classify(points, features, closed)


class ClosureDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = ClosureDetector()

    def test_large_gap_open_rectangle_does_not_snap(self):
        points = [(100, 100), (400, 100), (400, 300), (100, 300), (100, 170)]
        gap = math.dist(points[0], points[-1])
        diagonal = math.hypot(300, 200)
        self.assertEqual(gap, 70)
        self.assertLess(gap, 100)  # the former pixel-only rule closed this stroke
        self.assertGreater(gap / diagonal, 0.045)
        self.assertFalse(self.detector.detect(points))
        self.assertEqual(self.detector.snap(points), (points, False))

    def test_small_but_visible_gap_stays_open_by_scale_and_sampling_evidence(self):
        points = [(100, 100), (400, 100), (400, 300), (100, 300), (100, 120)]
        gap = math.dist(points[0], points[-1])
        self.assertEqual(gap, 20)
        self.assertLess(gap, 100)  # visibly open despite being under the old cutoff
        self.assertGreater(gap / math.hypot(300, 200), 0.045)
        self.assertFalse(self.detector.detect(points))

    def test_open_arc_is_not_closed_by_its_curvature(self):
        points = [
            (320 + 130 * math.cos(math.radians(angle)),
             240 + 130 * math.sin(math.radians(angle)))
            for angle in range(0, 341, 10)
        ]
        gap = math.dist(points[0], points[-1])
        diagonal = math.hypot(
            max(point[0] for point in points) - min(point[0] for point in points),
            max(point[1] for point in points) - min(point[1] for point in points),
        )
        self.assertLess(gap, 100)
        self.assertGreater(gap / diagonal, 0.045)
        self.assertFalse(self.detector.detect(points))

    def test_endpoint_match_without_enclosed_area_is_not_a_valid_closed_shape(self):
        retraced_line = [(0, 0), (100, 0), (10, 0), (90, 0), (0, 0)]
        self.assertEqual(math.dist(retraced_line[0], retraced_line[-1]), 0)
        self.assertFalse(self.detector.detect(retraced_line))

    def test_closed_circle_passes_scale_and_local_sampling_checks(self):
        points = circle_stroke()
        self.assertLessEqual(math.dist(points[0], points[-2]), self.detector.threshold_for(points))
        self.assertTrue(self.detector.detect(points))
        snapped, closed = self.detector.snap(points[:-1] + [
            (points[-2][0] + 1, points[-2][1])
        ])
        self.assertTrue(closed)
        self.assertEqual(snapped[-1], snapped[0])

    def test_closed_rectangle_with_small_endpoint_error_remains_closed(self):
        points = rectangle_stroke(gap=4)
        self.assertLessEqual(math.dist(points[0], points[-1]), self.detector.threshold_for(points))
        snapped, closed = self.detector.snap(points)
        self.assertTrue(closed)
        self.assertEqual(snapped[-1], points[0])

    def test_pause_trim_only_preserves_tail_that_passes_closure_evidence(self):
        capture = StrokeCapture(exit_buffer=3)
        capture.start_stroke()
        capture.current.points = [
            (0, 0), (100, 0), (100, 100), (0, 100),
            (4, 3), (130, 150), (145, 165), (160, 180),
        ]
        capture.current.raw_points = list(capture.current.points)
        capture.current.timestamps = list(range(len(capture.current.points)))

        capture.pause_stroke()

        self.assertEqual(capture.current.points[-1], (4, 3))
        self.assertTrue(self.detector.detect(capture.current.points))

    def test_visible_near_start_tail_does_not_rewind_an_open_stroke(self):
        capture = StrokeCapture(exit_buffer=3)
        capture.start_stroke()
        capture.current.points = [
            (0, 0), (300, 0), (300, 200), (0, 200),
            (0, 180), (0, 80), (0, 40), (0, 20),
        ]
        capture.current.raw_points = list(capture.current.points)
        capture.current.timestamps = list(range(len(capture.current.points)))

        capture.pause_stroke()

        self.assertEqual(capture.current.points[-1], (0, 180))
        self.assertFalse(self.detector.detect(capture.current.points))

    def test_pipeline_routes_open_strokes_to_open_classifier_path_only(self):
        classifier = RecordingClassifier()
        pipeline = StrokePipeline(classifier=classifier,
                                  factory=Shape3DFactory(panel_width=640, panel_height=480))
        strokes = {
            "large_gap": [(100, 100), (400, 100), (400, 300), (100, 300), (100, 170)],
            "small_gap": [(100, 100), (400, 100), (400, 300), (100, 300), (100, 120)],
            "arc": [(320 + 130 * math.cos(math.radians(a)),
                     240 + 130 * math.sin(math.radians(a))) for a in range(0, 341, 10)],
            "pen": [(150, 100), (450, 100), (450, 300), (150, 300), (150, 120)],
        }
        for name, points in strokes.items():
            with self.subTest(stroke=name):
                result = pipeline.convert(StrokeRecord(points=list(points), raw_points=list(points)))
                self.assertFalse(result.closed)
                self.assertIn(result.shape_class.kind, OPEN_LABELS)
                self.assertFalse(classifier.closed_arguments[-1])
                self.assertEqual(result.mesh.kind, "ribbon")

    def test_genuine_closed_circle_and_rectangle_route_to_closed_classifier(self):
        classifier = RecordingClassifier()
        pipeline = StrokePipeline(classifier=classifier,
                                  factory=Shape3DFactory(panel_width=640, panel_height=480))
        circle = pipeline.convert(StrokeRecord(points=circle_stroke(), raw_points=circle_stroke()))
        rectangle = rectangle_stroke(gap=4)
        rect_result = pipeline.convert(StrokeRecord(points=rectangle, raw_points=rectangle))

        self.assertTrue(circle.closed)
        self.assertTrue(rect_result.closed)
        self.assertEqual(classifier.closed_arguments, [True, True])
        self.assertEqual(circle.shape_class.kind, "circle")
        self.assertEqual(circle.mesh.candidate_id, "solid.sphere")
        self.assertIn(rect_result.shape_class.kind, {"square", "rectangle"})
        self.assertEqual(rect_result.mesh.kind, "polygon")

    def test_open_preview_mesh_has_separate_ends_and_no_end_to_start_faces(self):
        points = [(100, 100), (400, 100), (400, 300), (100, 300), (100, 120)]
        pipeline = StrokePipeline(classifier=RecordingClassifier(),
                                  factory=Shape3DFactory(panel_width=640, panel_height=480))
        result = pipeline.convert(StrokeRecord(points=list(points), raw_points=list(points)))
        mesh = result.mesh
        section_count = len(mesh.stroke_points)
        radial_count = mesh.TUBE_SIDES
        first = set(range(radial_count))
        last_start = (section_count - 1) * radial_count
        last = set(range(last_start, last_start + radial_count))
        self.assertGreater(math.dist(points[0], points[-1]), 0)
        self.assertGreater(section_count, 2)
        for offset in range(0, len(mesh.indices), 3):
            triangle = set(mesh.indices[offset:offset + 3])
            self.assertFalse(triangle & first and triangle & last)


if __name__ == "__main__":
    unittest.main()

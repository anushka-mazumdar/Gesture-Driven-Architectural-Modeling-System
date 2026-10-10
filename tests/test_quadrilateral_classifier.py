import math
import unittest

from shapes.stroke_analysis import analyze
from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_normalization import preprocess
from shapes.stroke_pipeline import StrokePipeline
from shapes.stroke_capture import StrokeRecord


def closed_box(width, height, angle=0.0):
    center = (320.0, 240.0)
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    corners = [(-width / 2, -height / 2),
               (width / 2, -height / 2),
               (width / 2, height / 2),
               (-width / 2, height / 2)]
    points = [
        (center[0] + x * cosine - y * sine,
         center[1] + x * sine + y * cosine)
        for x, y in corners
    ]
    return points + [points[0]]


class QuadrilateralClassifierTests(unittest.TestCase):
    def test_square_is_recognized_even_when_rotated_and_model_is_enabled(self):
        points = closed_box(200, 200, angle=31)
        classifier = StrokeClassifier(use_ml_model=True)
        features = analyze(preprocess(points, n=64))

        result = classifier.classify(points, features, closed=True)

        self.assertEqual(result.kind, "square")
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_rectangle_uses_side_ratio_not_axis_aligned_bbox(self):
        points = closed_box(280, 150, angle=38)
        classifier = StrokeClassifier(use_ml_model=True)
        features = analyze(preprocess(points, n=64))

        result = classifier.classify(points, features, closed=True)

        self.assertEqual(result.kind, "rectangle")
        self.assertGreaterEqual(result.confidence, 0.70)

    def test_pipeline_returns_candidate_recommendations_for_square_and_rectangle(self):
        pipeline = StrokePipeline()
        for label, points in (("square", closed_box(200, 200, 24)),
                              ("rectangle", closed_box(300, 160, 24))):
            with self.subTest(shape=label):
                result = pipeline.convert(StrokeRecord(points=points,
                                                       raw_points=list(points)))
                self.assertTrue(result.closed)
                self.assertEqual(result.shape_class.kind, label)
                self.assertEqual(result.candidates.status, "ok")
                self.assertTrue(result.candidates.candidate_ids)


if __name__ == "__main__":
    unittest.main()

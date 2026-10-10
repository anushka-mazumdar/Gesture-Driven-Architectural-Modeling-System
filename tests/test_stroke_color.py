import unittest

from shapes.stroke_capture import StrokeCapture, StrokeRecord
from shapes.stroke_classifier import StrokeClassifier
from shapes.stroke_pipeline import StrokePipeline
from shapes.shape_3d_factory import Shape3DFactory


class StrokeColorTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = StrokePipeline(
            classifier=StrokeClassifier(use_ml_model=False),
            factory=Shape3DFactory(panel_width=640, panel_height=480),
        )

    def test_capture_keeps_the_color_selected_when_each_stroke_begins(self):
        capture = StrokeCapture(min_points=5, smoothing=0.0, min_dist=1)
        red = (239 / 255.0, 83 / 255.0, 80 / 255.0)
        yellow = (244 / 255.0, 211 / 255.0, 94 / 255.0)

        capture.start_stroke(color=red)
        for point in ((100, 100), (150, 100), (150, 150), (100, 150), (101, 101)):
            capture.add_point(point)
        first = capture.finish()
        self.assertEqual(first.color, red)

        capture.start_stroke(color=yellow)
        for point in ((200, 100), (250, 100), (250, 150), (200, 150), (201, 101)):
            capture.add_point(point)
        second = capture.finish()
        self.assertEqual(second.color, yellow)
        self.assertEqual(first.color, red)

    def test_pipeline_colors_new_meshes_without_recoloring_existing_meshes(self):
        square = [(220, 140), (420, 140), (420, 340), (220, 340), (220, 140)]
        red = (239 / 255.0, 83 / 255.0, 80 / 255.0)
        yellow = (244 / 255.0, 211 / 255.0, 94 / 255.0)
        red_result = self.pipeline.convert(StrokeRecord(
            raw_points=list(square), points=list(square), color=red
        ))
        self.assertEqual(tuple(red_result.mesh.color), red)

        next_square = [(180, 120), (440, 120), (440, 380), (180, 380), (180, 120)]
        yellow_result = self.pipeline.convert(StrokeRecord(
            raw_points=list(next_square), points=list(next_square), color=yellow
        ))
        self.assertEqual(tuple(yellow_result.mesh.color), yellow)
        self.assertEqual(tuple(red_result.mesh.color), red)
        self.assertIsNot(red_result.mesh, yellow_result.mesh)


if __name__ == "__main__":
    unittest.main()

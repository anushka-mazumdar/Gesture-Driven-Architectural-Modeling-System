import numpy as np
from render.primitives import RibbonMesh, PolygonMesh


class Shape3DFactory:

    def __init__(self, panel_width=640, panel_height=480,
                 scene_scale=1.0):

        self.panel_width  = panel_width
        self.panel_height = panel_height
        self.scene_scale  = scene_scale


    def build(self, recommendation, stroke_points):
        """Assemble geometry from a MeshRecommendation.

        Pure construction — all sizing decisions live in the recommender.
        """
        if len(stroke_points) < 2:
            return None

        if recommendation.kind == "polygon":
            return self._create_polygon(stroke_points,
                                        extrude_depth=recommendation.extrude_depth)
        if recommendation.kind == "ribbon":
            return self._create_ribbon(stroke_points,
                                       thickness=recommendation.thickness,
                                       extrude_depth=recommendation.extrude_depth)
        return None


    def create_from_stroke(self, stroke_points, closed=False):
        """Compatibility entry point: full staged conversion of a stroke.

        Runs closure -> classify -> recommend -> build so the factory no
        longer embeds decisions. Returns the built mesh (or None).
        """
        from shapes.stroke_pipeline import StrokePipeline
        from shapes.stroke_analysis import analyze
        pipeline = StrokePipeline(factory=self)

        if len(stroke_points) < 2:
            return None

        if closed is True:
            # caller already knows it is closed — force an extrudable shape
            from shapes.stroke_classifier import ShapeClass
            if stroke_points and stroke_points[0] != stroke_points[-1]:
                stroke_points = list(stroke_points) + [stroke_points[0]]
            shape_class = ShapeClass("polygon", 1.0)
        else:
            # derive closure the normal way
            stroke_points, closed = pipeline.closure.snap(stroke_points)
            shape_class = pipeline.classifier.classify(
                stroke_points, analyze(stroke_points), closed)

        features = analyze(stroke_points)
        rec = pipeline.recommender.recommend(stroke_points, features, shape_class)
        return self.build(rec, stroke_points)


    def _create_polygon(self, stroke_points, extrude_depth=None):

        pts = np.array(stroke_points, dtype=np.float32)

        # compute bounding box in screen space
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())

        # depth = 1/3 of average dimension (default heuristic retained)
        if extrude_depth is None:
            avg_size    = (w + h) / 2.0
            extrude_depth = max(avg_size * 0.33, 20.0)

        scene_pts = self._to_scene(stroke_points, centre=True)
        return PolygonMesh(scene_pts, extrude_depth)


    def _create_ribbon(self, stroke_points, thickness=None, extrude_depth=None):

        pts = np.array(stroke_points, dtype=np.float32)

        # estimate stroke length
        diffs  = np.diff(pts, axis=0)
        length = float(np.linalg.norm(diffs, axis=1).sum())

        # tube thickness and depth proportional to length (defaults retained)
        if thickness is None:
            thickness = max(length * 0.06, 12.0)
        if extrude_depth is None:
            extrude_depth = thickness   # square cross-section feels like a tube

        scene_pts = self._to_scene(stroke_points, centre=True)
        return RibbonMesh(scene_pts, thickness, extrude_depth)


    def _to_scene(self, stroke_points, centre=False):

        pts = np.array(stroke_points, dtype=np.float32)

        pts[:, 0] -= self.panel_width  / 2.0
        pts[:, 1] -= self.panel_height / 2.0
        pts[:, 1] *= -1
        pts        *= self.scene_scale

        if centre:
            pts -= pts.mean(axis=0)

        return pts.tolist()
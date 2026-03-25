import math
import numpy as np
from render.primitives     import RibbonMesh, SolidMesh
from shapes.shape_2d_classifier import Shape2DClassifier


class Shape3DFactory:

    def __init__(self, panel_width=1280, panel_height=720,
                 scene_scale=1.0, ribbon_thickness=18.0, extrude_depth=28.0):

        self.panel_width      = panel_width
        self.panel_height     = panel_height
        self.scene_scale      = scene_scale
        self.ribbon_thickness = ribbon_thickness
        self.extrude_depth    = extrude_depth
        self.classifier       = Shape2DClassifier()


    def create_from_stroke(self, stroke_points, closed=False):

        if len(stroke_points) < 2:
            return None

        if closed:
            return self._create_solid(stroke_points)
        else:
            return self._create_ribbon(stroke_points)


    # ── open stroke → ribbon ──────────────────────────────────────────

    def _create_ribbon(self, stroke_points):

        pts = self._normalise(stroke_points)
        return RibbonMesh(pts, self.ribbon_thickness, self.extrude_depth)


    # ── closed stroke → solid ─────────────────────────────────────────

    def _create_solid(self, stroke_points):

        solid_type = self.classifier.classify(stroke_points)
        pts        = np.array(stroke_points, dtype=np.float32)

        # compute size from stroke bounding box
        w = pts[:, 0].max() - pts[:, 0].min()
        h = pts[:, 1].max() - pts[:, 1].min()
        size = max(float(np.mean([w, h])) * 0.5, 30.0)

        # centre position in scene coords
        cx = float(pts[:, 0].mean()) - self.panel_width  / 2.0
        cy = -(float(pts[:, 1].mean()) - self.panel_height / 2.0)

        mesh = SolidMesh(solid_type, size)
        mesh.position = np.array([cx, cy, 0.0], dtype=np.float32)

        return mesh


    # ── coordinate normalisation (ribbon only) ────────────────────────

    def _normalise(self, stroke_points):

        pts = np.array(stroke_points, dtype=np.float32)
        pts[:, 0] -= self.panel_width  / 2.0
        pts[:, 1] -= self.panel_height / 2.0
        pts[:, 1] *= -1
        pts       *= self.scene_scale
        pts       -= pts.mean(axis=0)

        return pts.tolist()
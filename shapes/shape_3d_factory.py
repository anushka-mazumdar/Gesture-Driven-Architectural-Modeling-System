import numpy as np
from render.primitives import RibbonMesh, PolygonMesh


class Shape3DFactory:

    def __init__(self, panel_width=640, panel_height=480,
                 scene_scale=1.0, ribbon_thickness=12.0,
                 extrude_depth=30.0):

        self.panel_width      = panel_width
        self.panel_height     = panel_height
        self.scene_scale      = scene_scale
        self.ribbon_thickness = ribbon_thickness
        self.extrude_depth    = extrude_depth


    def create_from_stroke(self, stroke_points, closed=False):

        if len(stroke_points) < 2:
            return None

        if closed:
            return self._create_polygon(stroke_points)
        else:
            return self._create_ribbon(stroke_points)


    def _create_ribbon(self, stroke_points):

        pts = self._to_scene(stroke_points, centre=True)
        return RibbonMesh(pts, self.ribbon_thickness, self.extrude_depth)


    def _create_polygon(self, stroke_points):

        pts = self._to_scene(stroke_points, centre=True)
        return PolygonMesh(pts, self.extrude_depth)


    def _to_scene(self, stroke_points, centre=False):

        pts = np.array(stroke_points, dtype=np.float32)

        pts[:, 0] -= self.panel_width  / 2.0
        pts[:, 1] -= self.panel_height / 2.0
        pts[:, 1] *= -1
        pts        *= self.scene_scale

        if centre:
            pts -= pts.mean(axis=0)

        return pts.tolist()
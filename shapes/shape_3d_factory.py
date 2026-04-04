import numpy as np
from render.primitives import RibbonMesh, PolygonMesh


class Shape3DFactory:

    def __init__(self, panel_width=640, panel_height=480,
                 scene_scale=1.0):

        self.panel_width  = panel_width
        self.panel_height = panel_height
        self.scene_scale  = scene_scale


    def create_from_stroke(self, stroke_points, closed=False):

        if len(stroke_points) < 2:
            return None

        if closed:
            return self._create_polygon(stroke_points)
        else:
            return self._create_ribbon(stroke_points)


    def _create_polygon(self, stroke_points):

        pts = np.array(stroke_points, dtype=np.float32)

        # compute bounding box in screen space
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())

        # depth = 1/3 of average dimension
        avg_size    = (w + h) / 2.0
        extrude_depth = max(avg_size * 0.33, 20.0)

        scene_pts = self._to_scene(stroke_points, centre=True)
        return PolygonMesh(scene_pts, extrude_depth)


    def _create_ribbon(self, stroke_points):

        pts = np.array(stroke_points, dtype=np.float32)

        # estimate stroke length
        diffs  = np.diff(pts, axis=0)
        length = float(np.linalg.norm(diffs, axis=1).sum())

        # tube thickness and depth proportional to length
        thickness     = max(length * 0.06, 12.0)
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
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# RibbonMesh  —  open stroke
# ─────────────────────────────────────────────────────────────────────────────

class RibbonMesh:

    def __init__(self, stroke_points, thickness=20.0, extrude_depth=30.0):

        self.stroke_points = stroke_points
        self.thickness     = thickness
        self.extrude_depth = extrude_depth
        self.vertices      = []
        self.normals       = []
        self.indices       = []
        self.position      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale         = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected      = False
        self.color         = (0.4, 0.75, 1.0)
        self.kind          = 'ribbon'

        self._build()


    def _build(self):

        pts = self.stroke_points
        if len(pts) < 2:
            return

        verts = []
        idxs  = []

        for i, (x, y) in enumerate(pts):

            if i == 0:
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
            elif i == len(pts) - 1:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
            else:
                dx = pts[i+1][0] - pts[i-1][0]
                dy = pts[i+1][1] - pts[i-1][1]

            length     = max((dx*dx + dy*dy) ** 0.5, 0.001)
            nx, ny     = -dy / length, dx / length
            half       = self.thickness / 2.0
            zf, zb     = self.extrude_depth / 2, -self.extrude_depth / 2

            verts.append([x + nx*half, y + ny*half, zf])
            verts.append([x - nx*half, y - ny*half, zf])
            verts.append([x + nx*half, y + ny*half, zb])
            verts.append([x - nx*half, y - ny*half, zb])

        for i in range(len(pts) - 1):
            b = i * 4
            idxs += [b,   b+1, b+5, b,   b+5, b+4]
            idxs += [b+2, b+6, b+7, b+2, b+7, b+3]
            idxs += [b,   b+4, b+6, b,   b+6, b+2]
            idxs += [b+1, b+3, b+7, b+1, b+7, b+5]

        self.vertices = np.array(verts,  dtype=np.float32)
        self.indices  = idxs
        self._compute_normals()


    def _compute_normals(self):

        self.normals = np.zeros_like(self.vertices)
        for i in range(0, len(self.indices), 3):
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            n  = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
            self.normals[i0] += n
            self.normals[i1] += n
            self.normals[i2] += n

        lens = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals /= np.where(lens == 0, 1, lens)


    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0) + self.position
        maxs = self.vertices.max(axis=0) + self.position
        return {'min': mins, 'max': maxs, 'center': (mins + maxs) / 2}


# ─────────────────────────────────────────────────────────────────────────────
# PolygonMesh  —  closed stroke extruded exactly as drawn
# ─────────────────────────────────────────────────────────────────────────────

class PolygonMesh:

    def __init__(self, stroke_points, extrude_depth=40.0):

        self.stroke_points = stroke_points
        self.extrude_depth = extrude_depth
        self.vertices      = []
        self.normals       = []
        self.indices       = []
        self.position      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale         = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected      = False
        self.color         = (0.55, 1.0, 0.65)
        self.kind          = 'polygon'

        self._build()


    def _build(self):

        pts = self.stroke_points
        if len(pts) < 3:
            return

        # remove duplicate closing point if present
        if pts[0] == pts[-1]:
            pts = pts[:-1]

        n    = len(pts)
        zf   =  self.extrude_depth / 2.0   # front face Z
        zb   = -self.extrude_depth / 2.0   # back face Z

        verts = []
        idxs  = []

        # front ring (indices 0..n-1) and back ring (indices n..2n-1)
        for (x, y) in pts:
            verts.append([x, y, zf])
        for (x, y) in pts:
            verts.append([x, y, zb])

        # centroid vertices for cap fan triangulation
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n

        front_center = len(verts)
        verts.append([cx, cy, zf])

        back_center = len(verts)
        verts.append([cx, cy, zb])

        # side walls — quad per edge
        for i in range(n):
            nxt  = (i + 1) % n
            f0, f1 = i,     nxt
            b0, b1 = i + n, nxt + n
            idxs += [f0, f1, b0,
                     f1, b1, b0]

        # front cap — fan from front_center
        for i in range(n):
            nxt = (i + 1) % n
            idxs += [front_center, i, nxt]

        # back cap — fan from back_center (reversed winding)
        for i in range(n):
            nxt = (i + 1) % n
            idxs += [back_center, nxt + n, i + n]

        self.vertices = np.array(verts, dtype=np.float32)
        self.indices  = idxs
        self._compute_normals()


    def _compute_normals(self):

        self.normals = np.zeros_like(self.vertices)
        for i in range(0, len(self.indices), 3):
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            n  = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
            self.normals[i0] += n
            self.normals[i1] += n
            self.normals[i2] += n

        lens = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals /= np.where(lens == 0, 1, lens)


    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0) + self.position
        maxs = self.vertices.max(axis=0) + self.position
        return {'min': mins, 'max': maxs, 'center': (mins + maxs) / 2}
import numpy as np
import math


def _signed_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p, a, b, c):
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _ear_clip_triangulate(pts):
    """Triangulate a simple 2D polygon (convex or concave) via ear clipping.

    Returns a list of (i, j, k) index triples into `pts`, CCW-wound. Used
    for extrusion caps instead of a naive centroid fan, which only produces
    correct geometry for convex outlines — a hand-drawn stroke routinely
    isn't. Degenerate/self-intersecting input degrades gracefully: clipping
    stops early and whatever ears were already found are returned.
    """
    n = len(pts)
    if n < 3:
        return []

    order = list(range(n))
    if _signed_area(pts) < 0:
        order.reverse()

    triangles = []
    guard = 0
    while len(order) > 3 and guard < n * n + 8:
        guard += 1
        clipped = False
        m = len(order)
        for i in range(m):
            ia, ib, ic = order[(i - 1) % m], order[i], order[(i + 1) % m]
            a, b, c = pts[ia], pts[ib], pts[ic]
            if _cross(a, b, c) <= 0:   # reflex/degenerate at b — not an ear
                continue
            if any(_point_in_triangle(pts[order[j]], a, b, c)
                   for j in range(m) if order[j] not in (ia, ib, ic)):
                continue
            triangles.append((ia, ib, ic))
            order.pop(i)
            clipped = True
            break
        if not clipped:
            break   # self-intersecting/degenerate remainder — stop here
    if len(order) == 3:
        triangles.append(tuple(order))
    return triangles




class RibbonMesh:

    TUBE_SIDES = 8   # number of faces around the tube circumference

    def __init__(self, stroke_points, thickness=20.0, extrude_depth=20.0):

        # for tube, radius = thickness / 2
        self.stroke_points = stroke_points
        self.radius        = max(thickness / 2.0, 6.0)

        self.vertices = []
        self.normals  = []
        self.indices  = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected    = False
        self.highlighted = False
        self.color       = (0.4, 0.75, 1.0)
        self.kind        = 'ribbon'

        self._build()


    def _build(self):

        pts = self.stroke_points
        if len(pts) < 2:
            return

        n      = self.TUBE_SIDES
        verts  = []
        idxs   = []

        # precompute tangent, normal, binormal at each point
        frames = self._compute_frames(pts)

        for i, (x, y, z) in enumerate(
                [(p[0], p[1], 0.0) for p in pts]):

            T, N, B = frames[i]

            # ring of vertices around tube circumference
            for j in range(n):
                angle = 2.0 * math.pi * j / n
                offset = (math.cos(angle) * N +
                          math.sin(angle) * B) * self.radius
                verts.append([x + offset[0],
                               y + offset[1],
                               z + offset[2]])

        # connect rings with quads
        for i in range(len(pts) - 1):
            for j in range(n):
                a  = i * n + j
                b  = i * n + (j + 1) % n
                c  = (i + 1) * n + (j + 1) % n
                d  = (i + 1) * n + j
                idxs += [a, b, d,
                          b, c, d]

        # cap start
        cx_s = sum(v[0] for v in verts[:n]) / n
        cy_s = sum(v[1] for v in verts[:n]) / n
        cz_s = sum(v[2] for v in verts[:n]) / n
        start_center = len(verts)
        verts.append([cx_s, cy_s, cz_s])
        for j in range(n):
            a = j
            b = (j + 1) % n
            idxs += [start_center, b, a]

        # cap end
        end_base = (len(pts) - 1) * n
        cx_e = sum(verts[end_base + j][0] for j in range(n)) / n
        cy_e = sum(verts[end_base + j][1] for j in range(n)) / n
        cz_e = sum(verts[end_base + j][2] for j in range(n)) / n
        end_center = len(verts)
        verts.append([cx_e, cy_e, cz_e])
        for j in range(n):
            a = end_base + j
            b = end_base + (j + 1) % n
            idxs += [end_center, a, b]

        self.vertices = np.array(verts, dtype=np.float32)
        self.indices  = idxs
        self._compute_normals()


    def _compute_frames(self, pts):
        """
        Compute tangent/normal/binormal at each point
        using parallel transport for a smooth tube frame.
        """
        count  = len(pts)
        frames = []

        # initial tangent
        def tangent_at(i):
            if i == 0:
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
            elif i == count - 1:
                dx = pts[-1][0] - pts[-2][0]
                dy = pts[-1][1] - pts[-2][1]
            else:
                dx = pts[i+1][0] - pts[i-1][0]
                dy = pts[i+1][1] - pts[i-1][1]
            ln = math.hypot(dx, dy)
            if ln < 0.001:
                return np.array([1.0, 0.0, 0.0])
            return np.array([dx/ln, dy/ln, 0.0])

        T0 = tangent_at(0)

        # pick initial normal perpendicular to T0
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(T0, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        N0 = np.cross(T0, up)
        N0 /= max(np.linalg.norm(N0), 0.001)
        B0  = np.cross(T0, N0)

        frames.append((T0, N0, B0))

        # parallel transport
        for i in range(1, count):
            T1 = tangent_at(i)
            axis = np.cross(frames[-1][0], T1)
            s    = np.linalg.norm(axis)
            c    = np.dot(frames[-1][0], T1)

            if s < 0.001:
                N1 = frames[-1][1]
                B1 = frames[-1][2]
            else:
                axis /= s
                angle = math.atan2(s, c)
                # Rodrigues rotation
                def rot(v, ax, ang):
                    return (v * math.cos(ang) +
                            np.cross(ax, v) * math.sin(ang) +
                            ax * np.dot(ax, v) * (1 - math.cos(ang)))
                N1 = rot(frames[-1][1], axis, angle)
                B1 = rot(frames[-1][2], axis, angle)

            frames.append((T1, N1, B1))

        return frames


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




class PolygonMesh:

    def __init__(self, stroke_points, extrude_depth=60.0):

        self.stroke_points = stroke_points
        self.extrude_depth = extrude_depth

        self.vertices = []
        self.normals  = []
        self.indices  = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected    = False
        self.highlighted = False
        self.color       = (0.55, 1.0, 0.65)
        self.kind        = 'polygon'

        self._build()


    def _build(self):

        pts = self.stroke_points
        if len(pts) < 3:
            return

        if pts[0] == pts[-1]:
            pts = pts[:-1]

        n  = len(pts)
        zf =  self.extrude_depth / 2.0
        zb = -self.extrude_depth / 2.0

        verts = []
        idxs  = []

        for (x, y) in pts:
            verts.append([x, y, zf])
        for (x, y) in pts:
            verts.append([x, y, zb])

        # side walls
        for i in range(n):
            nxt  = (i + 1) % n
            f0, f1 = i,     nxt
            b0, b1 = i + n, nxt + n
            idxs += [f0, f1, b0,
                     f1, b1, b0]

        # caps: ear-clip triangulation handles concave outlines correctly —
        # a naive centroid fan folds over itself on anything non-convex,
        # which is most hand-drawn strokes
        ears = _ear_clip_triangulate(pts)
        for a, b, c in ears:
            idxs += [a, b, c]              # front (already CCW)
            idxs += [c + n, b + n, a + n]  # back (winding reversed)

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
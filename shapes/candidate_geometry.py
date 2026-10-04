"""Deterministic procedural geometry primitives for 3D candidate families.

Local object-space convention (scene units, same scale as the existing
PolygonMesh/RibbonMesh):
  * +Y is the solid's vertical axis.
  * A 2D sketch is read as a plan-view footprint: sketch x -> X, sketch y -> Z.
    Footprints, profiles, and sweep paths therefore lie in the XZ plane.
  * Every finished mesh is recentred so its bounding-box centre is the origin,
    like the centred meshes Shape3DFactory already produces.

Output is (vertices Nx3 float32, normals Nx3 float32, flat index list) — the
exact fields render/mesh_bridge.serialize_mesh forwards to Three.js.
Pure numpy; no UI, gesture, classifier, or placement logic.
"""

import math

import numpy as np

from render.primitives import _ear_clip_triangulate


MAX_VERTICES = 200_000
_EPS = 1e-9
_UP = np.array([0.0, 1.0, 0.0])
_MITER_LIMIT = 4.0


class GeometryError(ValueError):
    """Raised when a construction cannot produce valid, bounded geometry."""


def _newell_normal(points):
    normal = np.zeros(3)
    count = len(points)
    for i in range(count):
        c, d = points[i], points[(i + 1) % count]
        normal[0] += (c[1] - d[1]) * (c[2] + d[2])
        normal[1] += (c[2] - d[2]) * (c[0] + d[0])
        normal[2] += (c[0] - d[0]) * (c[1] + d[1])
    return normal


class MeshBuilder:
    """Accumulates vertices/normals/triangles with a hard vertex budget.

    Triangles are oriented so their geometric normal agrees with their
    vertices' normals (outward), and zero-area triangles (lathe poles, cone
    apexes) are dropped, so every constructor emits consistent winding.
    """

    def __init__(self, max_vertices=MAX_VERTICES):
        self.max_vertices = max_vertices
        self._verts = []
        self._norms = []
        self._idx = []

    def vertex(self, position, normal):
        if len(self._verts) >= self.max_vertices:
            raise GeometryError("construction exceeds the vertex budget")
        normal = np.asarray(normal, dtype=float)
        length = np.linalg.norm(normal)
        self._verts.append(np.asarray(position, dtype=float))
        self._norms.append(normal / length if length > _EPS else normal)
        return len(self._verts) - 1

    def triangle(self, a, b, c):
        pa, pb, pc = self._verts[a], self._verts[b], self._verts[c]
        geometric = np.cross(pb - pa, pc - pa)
        if np.linalg.norm(geometric) <= _EPS:
            return
        if np.dot(geometric, self._norms[a] + self._norms[b] + self._norms[c]) < 0:
            b, c = c, b
        self._idx.extend((a, b, c))

    def flat_polygon(self, points, outward_hint, triangles=None):
        """Add one flat face; default fan triangulation assumes convexity."""
        pts = np.asarray(points, dtype=float)
        normal = _newell_normal(pts)
        length = np.linalg.norm(normal)
        if length <= _EPS:
            return
        normal /= length
        if np.dot(normal, outward_hint) < 0:
            normal = -normal
        ids = [self.vertex(p, normal) for p in pts]
        if triangles is None:
            triangles = [(0, k, k + 1) for k in range(1, len(pts) - 1)]
        for a, b, c in triangles:
            self.triangle(ids[a], ids[b], ids[c])

    def finish(self, scale=(1.0, 1.0, 1.0)):
        if not self._idx:
            raise GeometryError("construction produced no triangles")
        scale = np.asarray(scale, dtype=float)
        verts = np.array(self._verts, dtype=float) * scale
        norms = np.array(self._norms, dtype=float) / scale   # inverse-transpose
        lengths = np.linalg.norm(norms, axis=1, keepdims=True)
        norms /= np.where(lengths == 0, 1.0, lengths)
        verts -= (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        if not (np.all(np.isfinite(verts)) and np.all(np.isfinite(norms))):
            raise GeometryError("construction produced non-finite values")
        return verts.astype(np.float32), norms.astype(np.float32), list(self._idx)


# ---- 2D footprint helpers --------------------------------------------------

def signed_area(poly):
    return 0.5 * sum(x1 * z2 - x2 * z1
                     for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]))


def area_centroid(poly):
    area = signed_area(poly)
    if abs(area) <= _EPS:
        return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    cx = cz = 0.0
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        cross = x1 * z2 - x2 * z1
        cx += (x1 + x2) * cross
        cz += (z1 + z2) * cross
    return (cx / (6.0 * area), cz / (6.0 * area))


def triangulate(poly):
    triangles = _ear_clip_triangulate([tuple(p) for p in poly])
    if len(triangles) != len(poly) - 2:
        raise GeometryError("footprint could not be triangulated")
    return triangles


def _edge_outward(p0, p1, ccw):
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return np.array([dz, 0.0, -dx]) if ccw else np.array([-dz, 0.0, dx])


def regular_polygon(sides, radius):
    """CCW regular polygon in XZ, first vertex on +Z (pointing 'forward')."""
    return [(radius * math.sin(2 * math.pi * k / sides),
             radius * math.cos(2 * math.pi * k / sides)) for k in range(sides)]


def isosceles_triangle(width, depth):
    return [(-width / 2.0, depth / 2.0), (width / 2.0, depth / 2.0), (0.0, -depth / 2.0)]


# ---- solids of revolution --------------------------------------------------

def lathe(builder, pieces, segments):
    """Revolve profile pieces (lists of (r, y, nr, ny)) about +Y.

    Each piece is smooth internally; separate pieces give hard edges.
    """
    for piece in pieces:
        rings = []
        for r, y, nr, ny in piece:
            ring = []
            for j in range(segments + 1):
                t = 2.0 * math.pi * j / segments
                c, s = math.cos(t), math.sin(t)
                ring.append(builder.vertex((r * c, y, r * s), (nr * c, ny, nr * s)))
            rings.append(ring)
        for i in range(len(rings) - 1):
            for j in range(segments):
                a0, a1 = rings[i][j], rings[i][j + 1]
                b0, b1 = rings[i + 1][j], rings[i + 1][j + 1]
                builder.triangle(a0, a1, b1)
                builder.triangle(a0, b1, b0)


def _disc(radius, y, up):
    ny = 1.0 if up else -1.0
    return [(radius, y, 0.0, ny), (0.0, y, 0.0, ny)]


def sphere(builder, radius, width_segments, height_segments):
    piece = []
    for k in range(height_segments + 1):
        phi = -math.pi / 2 + math.pi * k / height_segments
        piece.append((radius * math.cos(phi), radius * math.sin(phi),
                      math.cos(phi), math.sin(phi)))
    lathe(builder, [piece], width_segments)


def hemisphere(builder, radius, width_segments, height_segments):
    piece = []
    for k in range(height_segments + 1):
        phi = (math.pi / 2) * k / height_segments
        piece.append((radius * math.cos(phi), radius * math.sin(phi),
                      math.cos(phi), math.sin(phi)))
    lathe(builder, [piece, _disc(radius, 0.0, up=False)], width_segments)


def cylinder(builder, radius, height, segments):
    h = height / 2.0
    side = [(radius, -h, 1.0, 0.0), (radius, h, 1.0, 0.0)]
    lathe(builder, [side, _disc(radius, h, up=True), _disc(radius, -h, up=False)],
          segments)


def cone(builder, radius, height, segments):
    h = height / 2.0
    slope = math.hypot(height, radius)
    nr, ny = height / slope, radius / slope
    side = [(radius, -h, nr, ny), (0.0, h, nr, ny)]
    lathe(builder, [side, _disc(radius, -h, up=False)], segments)


def torus(builder, major_radius, minor_radius, radial_segments, tubular_segments):
    piece = []
    for k in range(radial_segments + 1):
        a = 2.0 * math.pi * k / radial_segments
        piece.append((major_radius + minor_radius * math.cos(a),
                      minor_radius * math.sin(a), math.cos(a), math.sin(a)))
    lathe(builder, [piece], tubular_segments)


# ---- polyhedra from footprints ----------------------------------------------

def convex_solid(builder, faces):
    """Flat-shaded convex solid; outward = away from the solid's centre."""
    centre = np.mean([p for face in faces for p in face], axis=0)
    for face in faces:
        builder.flat_polygon(face, np.mean(face, axis=0) - centre)


def box(builder, width, height, depth):
    x, y, z = width / 2.0, height / 2.0, depth / 2.0
    c = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
         (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    convex_solid(builder, [[c[i] for i in f] for f in faces])


def wedge(builder, width, height, depth):
    """Ramp: full-height back face at -Z sloping down to a front edge at +Z."""
    x, y, z = width / 2.0, height / 2.0, depth / 2.0
    left = [(-x, -y, -z), (-x, -y, z), (-x, y, -z)]
    right = [(x, py, pz) for _, py, pz in left]
    faces = [left, right,
             [left[0], left[1], right[1], right[0]],     # bottom
             [left[0], left[2], right[2], right[0]],     # back
             [left[1], left[2], right[2], right[1]]]     # slope
    convex_solid(builder, faces)


def prism(builder, poly, height):
    triangles = triangulate(poly)
    ccw = signed_area(poly) > 0
    h = height / 2.0
    bottom = [(x, -h, z) for x, z in poly]
    top = [(x, h, z) for x, z in poly]
    builder.flat_polygon(bottom, -_UP, triangles)
    builder.flat_polygon(top, _UP, triangles)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        builder.flat_polygon([bottom[i], bottom[j], top[j], top[i]],
                             _edge_outward(poly[i], poly[j], ccw))


def pyramid(builder, poly, height, top_scale=0.0):
    """Pyramid over a footprint; top_scale > 0 truncates it into a frustum/loft."""
    triangles = triangulate(poly)
    ccw = signed_area(poly) > 0
    cx, cz = area_centroid(poly)
    h = height / 2.0
    bottom = [(x, -h, z) for x, z in poly]
    builder.flat_polygon(bottom, -_UP, triangles)
    if top_scale > 0.0:
        top = [(cx + (x - cx) * top_scale, h, cz + (z - cz) * top_scale) for x, z in poly]
        builder.flat_polygon(top, _UP, triangles)
    apex = (cx, h, cz)
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        hint = _edge_outward(poly[i], poly[j], ccw)
        if top_scale > 0.0:
            builder.flat_polygon([bottom[i], bottom[j], top[j], top[i]], hint)
        else:
            builder.flat_polygon([bottom[i], bottom[j], apex], hint)


def bipyramid(builder, poly, height):
    ccw = signed_area(poly) > 0
    cx, cz = area_centroid(poly)
    h = height / 2.0
    equator = [(x, 0.0, z) for x, z in poly]
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        hint = _edge_outward(poly[i], poly[j], ccw)
        builder.flat_polygon([equator[i], equator[j], (cx, h, cz)], hint + 0.1 * _UP)
        builder.flat_polygon([equator[i], equator[j], (cx, -h, cz)], hint - 0.1 * _UP)


# ---- sweeps ---------------------------------------------------------------

def circle_section(radius, segments):
    ring, cap = [], []
    for j in range(segments):
        t = 2.0 * math.pi * j / segments
        u, v = radius * math.cos(t), radius * math.sin(t)
        ring.append((u, v, math.cos(t), math.sin(t)))
        cap.append((u, v))
    strips = [(j, (j + 1) % segments) for j in range(segments)]
    return ring, strips, cap


def rect_section(u_size, v_size):
    """Rectangle: u along the path's horizontal normal, v along +Y."""
    u, v = u_size / 2.0, v_size / 2.0
    corners = [(-u, -v), (u, -v), (u, v), (-u, v)]
    edge_normals = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
    ring, strips = [], []
    for k in range(4):
        (u0, v0), (u1, v1), (nu, nv) = corners[k], corners[(k + 1) % 4], edge_normals[k]
        ring += [(u0, v0, nu, nv), (u1, v1, nu, nv)]
        strips.append((2 * k, 2 * k + 1))
    return ring, strips, corners


def _unit(v):
    length = np.linalg.norm(v)
    return v / length if length > _EPS else None


def sweep(builder, path, section, closed=False):
    """Sweep a cross-section along a planar XZ path (mitred joints, capped ends)."""
    pts = []
    for x, z in path:
        p = np.array([x, 0.0, z], dtype=float)
        if not pts or np.linalg.norm(p - pts[-1]) > 1e-6:
            pts.append(p)
    if closed and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= 1e-6:
        pts.pop()
    count = len(pts)
    if count < (3 if closed else 2):
        raise GeometryError("sweep path needs distinct points")

    frames = []
    for i in range(count):
        d_in = _unit(pts[i] - pts[i - 1]) if (closed or i > 0) else None
        d_out = _unit(pts[(i + 1) % count] - pts[i]) if (closed or i < count - 1) else None
        if d_in is None or d_out is None:
            tangent, miter = (d_out if d_in is None else d_in), 1.0
        else:
            tangent = _unit(d_in + d_out)
            if tangent is None:          # 180-degree reversal
                tangent, miter = d_in, 1.0
            else:
                miter = min(1.0 / max(np.dot(tangent, d_in), _EPS), _MITER_LIMIT)
        normal = _unit(np.cross(tangent, _UP))
        frames.append((tangent, normal, miter))

    ring, strips, cap = section
    rings = []
    for p, (_, normal, miter) in zip(pts, frames):
        rings.append([builder.vertex(p + u * miter * normal + v * _UP, nu * normal + nv * _UP)
                      for u, v, nu, nv in ring])
    for i in range(count if closed else count - 1):
        k = (i + 1) % count
        for a, c in strips:
            builder.triangle(rings[i][a], rings[i][c], rings[k][c])
            builder.triangle(rings[i][a], rings[k][c], rings[k][a])

    if not closed:
        for p, (tangent, normal, _), sign in ((pts[0], frames[0], -1.0),
                                               (pts[-1], frames[-1], 1.0)):
            builder.flat_polygon([p + u * normal + v * _UP for u, v in cap], sign * tangent)


def arc_path(radius, sweep_degrees, segments):
    half = math.radians(sweep_degrees) / 2.0
    return [(radius * math.cos(-half + 2 * half * k / segments),
             radius * math.sin(-half + 2 * half * k / segments))
            for k in range(segments + 1)]


def ellipse_path(radius_x, radius_z, segments):
    return [(radius_x * math.cos(2 * math.pi * k / segments),
             radius_z * math.sin(2 * math.pi * k / segments)) for k in range(segments)]


def chaikin(path, iterations):
    """Corner-cutting smoothing of an open path; endpoints are preserved."""
    pts = [tuple(p) for p in path]
    for _ in range(iterations):
        if len(pts) < 3:
            break
        smoothed = [pts[0]]
        for (x0, z0), (x1, z1) in zip(pts, pts[1:]):
            smoothed.append((0.75 * x0 + 0.25 * x1, 0.75 * z0 + 0.25 * z1))
            smoothed.append((0.25 * x0 + 0.75 * x1, 0.25 * z0 + 0.75 * z1))
        smoothed.append(pts[-1])
        pts = smoothed
    return pts

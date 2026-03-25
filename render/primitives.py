import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# RibbonMesh  —  open stroke extruded into a 3D ribbon
# ─────────────────────────────────────────────────────────────────────────────

class RibbonMesh:

    def __init__(self, stroke_points, thickness=20.0, extrude_depth=30.0):

        self.stroke_points = stroke_points
        self.thickness     = thickness
        self.extrude_depth = extrude_depth

        self.vertices = []
        self.normals  = []
        self.indices  = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected = False
        self.color    = (0.4, 0.75, 1.0)
        self.kind     = 'ribbon'

        self._build()


    def _build(self):

        pts = self.stroke_points
        if len(pts) < 2:
            return

        self.vertices = []
        self.indices  = []

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

            length = max((dx*dx + dy*dy) ** 0.5, 0.001)
            nx, ny = -dy / length, dx / length
            half   = self.thickness / 2.0

            self.vertices.append([x + nx*half, y + ny*half,  self.extrude_depth / 2])
            self.vertices.append([x - nx*half, y - ny*half,  self.extrude_depth / 2])
            self.vertices.append([x + nx*half, y + ny*half, -self.extrude_depth / 2])
            self.vertices.append([x - nx*half, y - ny*half, -self.extrude_depth / 2])

        for i in range(len(pts) - 1):
            b = i * 4
            self.indices += [b,   b+1, b+5, b,   b+5, b+4]
            self.indices += [b+2, b+6, b+7, b+2, b+7, b+3]
            self.indices += [b,   b+4, b+6, b,   b+6, b+2]
            self.indices += [b+1, b+3, b+7, b+1, b+7, b+5]

        self.vertices = np.array(self.vertices, dtype=np.float32)
        self._compute_normals()


    def _compute_normals(self):

        self.normals = np.zeros_like(self.vertices)

        for i in range(0, len(self.indices), 3):
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
            self.normals[i0] += n
            self.normals[i1] += n
            self.normals[i2] += n

        lens = np.linalg.norm(self.normals, axis=1, keepdims=True)
        lens = np.where(lens == 0, 1, lens)
        self.normals /= lens


    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0) + self.position
        maxs = self.vertices.max(axis=0) + self.position
        return {'min': mins, 'max': maxs, 'center': (mins + maxs) / 2}


# ─────────────────────────────────────────────────────────────────────────────
# SolidMesh  —  closed stroke converted to a recognisable 3D solid
# ─────────────────────────────────────────────────────────────────────────────

class SolidMesh:

    COLORS = {
        'sphere':      (1.0, 0.45, 0.45),   # warm red
        'tetrahedron': (0.45, 1.0, 0.55),   # green
        'cuboid':      (0.45, 0.65, 1.0),   # blue
        'prism':       (1.0, 0.85, 0.35),   # yellow
    }

    def __init__(self, solid_type, size=80.0):

        self.solid_type = solid_type
        self.size       = size

        self.vertices = np.array([], dtype=np.float32)
        self.normals  = np.array([], dtype=np.float32)
        self.indices  = []

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale    = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.selected = False
        self.kind     = 'solid'

        # pick colour by type
        key = solid_type if solid_type in self.COLORS else 'prism'
        self.color = self.COLORS[key]

        self._build()


    def _build(self):

        t = self.solid_type

        if t == 'sphere':
            self._build_sphere()
        elif t == 'tetrahedron':
            self._build_tetrahedron()
        elif t == 'cuboid':
            self._build_cuboid()
        elif t.startswith('prism_'):
            n = int(t.split('_')[1])
            self._build_prism(n)
        else:
            self._build_sphere()

        self._compute_normals()


    def _build_sphere(self, stacks=14, slices=18):

        r    = self.size / 2.0
        verts = []
        idxs  = []

        for i in range(stacks + 1):
            phi = math.pi * i / stacks
            for j in range(slices + 1):
                theta = 2 * math.pi * j / slices
                x = r * math.sin(phi) * math.cos(theta)
                y = r * math.cos(phi)
                z = r * math.sin(phi) * math.sin(theta)
                verts.append([x, y, z])

        for i in range(stacks):
            for j in range(slices):
                a = i * (slices + 1) + j
                b = a + slices + 1
                idxs += [a, b, a+1, b, b+1, a+1]

        self.vertices = np.array(verts, dtype=np.float32)
        self.indices  = idxs


    def _build_tetrahedron(self):

        s = self.size
        h = s * math.sqrt(2/3)

        self.vertices = np.array([
            [ 0,         h * 0.75,  0       ],
            [-s / 2,    -h * 0.25,  s / 3   ],
            [ s / 2,    -h * 0.25,  s / 3   ],
            [ 0,        -h * 0.25, -s * 2/3 ],
        ], dtype=np.float32)

        self.indices = [
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            1, 3, 2,
        ]


    def _build_cuboid(self):

        h = self.size / 2.0

        self.vertices = np.array([
            [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
        ], dtype=np.float32)

        self.indices = [
            0,1,2, 0,2,3,   # back
            4,6,5, 4,7,6,   # front
            0,4,1, 4,5,1,   # bottom
            2,6,3, 6,7,3,   # top
            0,3,4, 3,7,4,   # left
            1,5,2, 5,6,2,   # right
        ]


    def _build_prism(self, n):

        r = self.size / 2.0
        d = self.size / 2.0

        verts = []
        idxs  = []

        # bottom ring, top ring
        for sign, z in [(-1, -d), (1, d)]:
            for i in range(n):
                angle = 2 * math.pi * i / n
                verts.append([r * math.cos(angle), r * math.sin(angle), z])

        # side faces
        for i in range(n):
            nxt = (i + 1) % n
            b0, b1 = i, nxt
            t0, t1 = i + n, nxt + n
            idxs += [b0, b1, t0, b1, t1, t0]

        # bottom cap
        for i in range(1, n - 1):
            idxs += [0, i, i + 1]

        # top cap
        base = n
        for i in range(1, n - 1):
            idxs += [base, base + i + 1, base + i]

        self.vertices = np.array(verts, dtype=np.float32)
        self.indices  = idxs


    def _compute_normals(self):

        if len(self.vertices) == 0:
            return

        self.normals = np.zeros_like(self.vertices)

        for i in range(0, len(self.indices), 3):
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            n = np.cross(v1 - v0, v2 - v0)
            ln = np.linalg.norm(n)
            if ln > 0:
                n /= ln
            self.normals[i0] += n
            self.normals[i1] += n
            self.normals[i2] += n

        lens = np.linalg.norm(self.normals, axis=1, keepdims=True)
        lens = np.where(lens == 0, 1, lens)
        self.normals /= lens


    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        mins = self.vertices.min(axis=0) + self.position
        maxs = self.vertices.max(axis=0) + self.position
        return {'min': mins, 'max': maxs, 'center': (mins + maxs) / 2}
"""Stable, local-space connection anchors for indexed triangle meshes."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SnapAnchor:
    """A local-space mesh connection point with a stable mesh-local ID."""

    anchor_id: str
    kind: str                         # center | vertex | edge | face
    local_position: tuple
    local_normal: tuple = None
    source_vertices: tuple = ()       # quantized local-space vertex keys

    def world_position(self, obj):
        return transform_point(obj, self.local_position)

    def world_normal(self, obj):
        if self.local_normal is None:
            return None
        scale = _vector3(getattr(obj, "scale", (1, 1, 1)), (1, 1, 1))
        # Normals transform by inverse-transpose; zero scale components safely
        # contribute zero rather than producing infinities.
        local = np.asarray(self.local_normal, dtype=float)
        unscaled = np.divide(local, scale, out=np.zeros(3), where=np.abs(scale) > 1e-12)
        world = _rotation_matrix(getattr(obj, "rotation", (0, 0, 0))) @ unscaled
        length = float(np.linalg.norm(world))
        return tuple((world / length).tolist()) if length > 1e-12 else None


def _vector3(value, default):
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=float)
    if not np.all(np.isfinite(arr)):
        return np.asarray(default, dtype=float)
    return arr


def _rotation_matrix(rotation_degrees):
    # Three.js renderer stores degrees and uses Euler XYZ (Rz * Ry * Rx).
    angles = np.radians(_vector3(rotation_degrees, (0, 0, 0)))
    sx, sy, sz = np.sin(angles)
    cx, cy, cz = np.cos(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def transform_point(obj, local_position):
    """Apply the WebView's scale, Euler rotation, and translation to a point."""
    scale = _vector3(getattr(obj, "scale", (1, 1, 1)), (1, 1, 1))
    rotation = _rotation_matrix(getattr(obj, "rotation", (0, 0, 0)))
    position = _vector3(getattr(obj, "position", (0, 0, 0)), (0, 0, 0))
    return rotation @ (np.asarray(local_position, dtype=float) * scale) + position


def _vertex_key(point, tolerance):
    return tuple(int(round(float(component) / tolerance)) for component in point)


def build_snap_anchors(vertices, indices, tolerance=1e-5):
    """Generate deterministic center, unique vertex/edge, and triangle-face anchors.

    Anchors stay in local mesh coordinates, so object transforms can be applied
    at query time without rebuilding or changing the geometry.
    """
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be a positive finite number")
    try:
        points = np.asarray(vertices, dtype=float).reshape((-1, 3))
        raw_indices = np.asarray(indices).reshape(-1)
    except (TypeError, ValueError):
        return ()
    if not len(points) or not np.all(np.isfinite(points)):
        return ()

    key_to_point = {}
    index_to_key = {}
    for index, point in enumerate(points):
        key = _vertex_key(point, tolerance)
        key_to_point.setdefault(key, point.copy())
        index_to_key[index] = key

    anchors = []
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) / 2.0
    anchors.append(SnapAnchor("center", "center", tuple(center.tolist())))

    for key in sorted(key_to_point):
        anchors.append(SnapAnchor(
            f"vertex:{key[0]},{key[1]},{key[2]}", "vertex",
            tuple(key_to_point[key].tolist()), source_vertices=(key,),
        ))

    edge_keys = set()
    triangles = []
    for offset in range(0, len(raw_indices) - 2, 3):
        try:
            ids = tuple(int(raw_indices[offset + i]) for i in range(3))
        except (TypeError, ValueError, OverflowError):
            continue
        if any(index < 0 or index >= len(points) for index in ids):
            continue
        keys = tuple(index_to_key[index] for index in ids)
        for left, right in ((keys[0], keys[1]), (keys[1], keys[2]),
                            (keys[2], keys[0])):
            if left != right:
                edge_keys.add(tuple(sorted((left, right))))
        a, b, c = (points[index] for index in ids)
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        face_key = tuple(sorted(keys))
        if length > tolerance and len(set(keys)) == 3:
            triangles.append((face_key, (a + b + c) / 3.0, normal / length, keys))

    for left, right in sorted(edge_keys):
        midpoint = (key_to_point[left] + key_to_point[right]) / 2.0
        anchors.append(SnapAnchor(
            f"edge:{left[0]},{left[1]},{left[2]}|{right[0]},{right[1]},{right[2]}",
            "edge", tuple(midpoint.tolist()), source_vertices=(left, right),
        ))

    seen_faces = set()
    for face_key, centroid, normal, keys in sorted(triangles, key=lambda item: item[0]):
        if face_key in seen_faces:
            continue
        seen_faces.add(face_key)
        anchors.append(SnapAnchor(
            "face:" + "|".join(f"{key[0]},{key[1]},{key[2]}" for key in face_key),
            "face", tuple(centroid.tolist()), tuple(normal.tolist()), face_key,
        ))
    return tuple(anchors)

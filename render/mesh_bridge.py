"""
Serializes an existing project mesh object into a JSON-safe dict that
the Three.js side can turn into a BufferGeometry.

This module only *reads* mesh objects shaped like the ones
render/primitives.py produces (vertices, normals, indices, color,
position, rotation, scale); it does not change how those meshes are
built. Used by ThreeJSRenderer (render/threejs_renderer.py) to push
live scene objects from main.py's application loop.
"""

import numpy as np


def serialize_mesh(mesh):
    """Convert a mesh object into plain Python types for the pywebview
    JS bridge (numpy arrays are not JSON-serializable on their own).

    Mirrors the exact fields render/renderer.py._draw_mesh() reads:
    vertices, normals, indices, color, and the position/rotation/scale
    TRS transform.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.normals, dtype=np.float32)

    return {
        "vertices": vertices.flatten().tolist(),
        "normals": normals.flatten().tolist(),
        "indices": [int(i) for i in mesh.indices],
        "color": [float(c) for c in mesh.color],
        "position": [float(v) for v in mesh.position],
        "rotation": [float(v) for v in mesh.rotation],
        "scale": [float(v) for v in mesh.scale],
        "kind": getattr(mesh, "kind", "mesh"),
    }


def serialize_hand(hand_mesh):
    """Serialize a HandMesh's smoothed 21-point landmark set for the
    Three.js hand overlay (see webview_app/web/main.js:receiveHandFromPython).

    Returns None when no hand is currently tracked, so the JS side
    clears the overlay instead of freezing on stale landmarks. The
    finger/palm connectivity is a fixed constant (mirrored from
    render/renderer.py._draw_hand) and lives in JS — only the tracked
    positions, which are the actual per-frame state, cross the bridge.
    """
    landmarks = getattr(hand_mesh, "landmarks", None)
    if landmarks is None:
        return None
    return np.asarray(landmarks, dtype=np.float32).tolist()

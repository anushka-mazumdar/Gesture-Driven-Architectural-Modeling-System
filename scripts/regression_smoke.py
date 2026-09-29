"""
Phase 0.4e regression smoke test.

Drives the pipeline that main.py's application loop uses — shape
generation, the Three.js renderer's object surface, the mesh-bridge
serialization, object selection, and manipulation transforms — without
opening a webcam or any GUI window. Asserts the same behaviour the old
PyOpenGL backend had, so the Three.js-backend switch stays drop-in.

Run:  python scripts/regression_smoke.py
Exit code 0 means all checks passed.
"""

import json
import os
import sys

import numpy as np

# Make the project root importable regardless of cwd (scripts/ is itself
# put on sys.path when run as `python scripts/regression_smoke.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render.threejs_renderer import ThreeJSRenderer
from render.mesh_bridge       import serialize_mesh, serialize_hand
from render.hand_mesh         import HandMesh
from shapes.shape_3d_factory  import Shape3DFactory
from interaction.object_selection import ObjectSelection
from interaction.manipulation     import Manipulation
from interaction.snapping         import Snapping

PANEL_W, PANEL_H = 640, 480

CLOSED_STROKE = [
    (100, 40), (180, 70), (200, 150),
    (150, 210), (60, 200), (20, 120),
]
OPEN_STROKE = [
    (40, 200), (120, 160), (200, 120), (300, 100),
]

# A 21-point landmark set roughly shaped like an open hand, many frames of
# the same shape so HandMesh's exponential smoothing settles.
HAND_LANDMARKS = [
    [x, y, 0.0] for (x, y) in [
        (320, 240), (310, 210), (300, 190), (295, 175), (292, 165),   # wrist, thumb
        (330, 205), (335, 180), (338, 160), (340, 145),               # index
        (355, 205), (362, 180), (366, 158), (368, 142),               # middle
        (380, 208), (385, 184), (388, 165), (390, 150),               # ring
        (400, 212), (404, 192), (406, 178), (408, 166),               # pinky
    ]
]

checks = []


def check(name, condition, detail=""):
    ok = bool(condition)
    checks.append((name, ok))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    return ok


def main():
    factory = Shape3DFactory(panel_width=PANEL_W, panel_height=PANEL_H)

    # 1. Shape generation — closed -> polygon, open -> ribbon.
    polygon = factory.create_from_stroke(CLOSED_STROKE, closed=True)
    ribbon  = factory.create_from_stroke(OPEN_STROKE, closed=False)
    check("closed stroke -> PolygonMesh", polygon is not None and polygon.kind == "polygon",
          f"got kind={getattr(polygon, 'kind', None)}")
    check("open stroke -> RibbonMesh", ribbon is not None and ribbon.kind == "ribbon",
          f"got kind={getattr(ribbon, 'kind', None)}")
    check("meshes carry geometry", polygon and len(polygon.vertices) > 0 and polygon.get_bounds() is not None)

    # 2. Renderer object surface — the ops main.py / interaction call.
    renderer = ThreeJSRenderer(PANEL_W, PANEL_H)
    renderer.add_object(polygon)
    renderer.add_object(ribbon)
    check("add_object -> objects list", len(renderer.objects) == 2)

    frame = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    out = renderer.render(frame)
    check("render() returns display unchanged (3D lives in WebView)",
          isinstance(out, np.ndarray) and out.shape == frame.shape)
    check("render() does not mutate the input frame", np.array_equal(out, frame))

    # 3. Bridge serialization — JSON-safe, mirrors renderer-visible fields.
    for mesh, name in ((polygon, "polygon"), (ribbon, "ribbon")):
        data = serialize_mesh(mesh)
        re_encoded = json.dumps(data)
        check(f"serialize_mesh({name}) -> JSON-safe", bool(re_encoded))
        check(f"serialize_mesh({name}) -> all fields",
              all(k in data for k in ("vertices", "normals", "indices", "color",
                                      "position", "rotation", "scale")))

    # 4. Hand mesh — tracks landmarks, serializes them; None when idle.
    hand = HandMesh()
    for _ in range(5):
        hand.update_from_landmarks(HAND_LANDMARKS, PANEL_W, PANEL_H)
    check("HandMesh tracks 21 landmarks", hand.landmarks is not None and len(hand.landmarks) == 21)
    hand_payload = serialize_hand(hand)
    check("serialize_hand -> 21 JSON-safe points",
          hand_payload is not None and len(hand_payload) == 21 and json.dumps(hand_payload))
    idle_hand = HandMesh()
    check("serialize_hand(None landmarks) -> None", serialize_hand(idle_hand) is None)

    # 5. Object selection — picks the hovered object from renderer.objects.
    selector = ObjectSelection(renderer)
    # Map the polygon's true scene centre -> screen space and hover that
    # point; ObjectSelection must pick the polygon, not the ribbon beside it.
    p_center = polygon.position + polygon.get_bounds()["center"]
    p_screen = (p_center[0] + PANEL_W / 2.0, -p_center[1] + PANEL_H / 2.0)
    hovered = selector.update((int(p_screen[0]), int(p_screen[1])), PANEL_W, PANEL_H)
    check("ObjectSelection picks the hovered object", hovered is polygon,
          f"got {type(hovered).__name__ if hovered else None}")
    selector.deselect_all()
    check("deselect_all clears selection", selector.get_selected() is None)

    # 6. Manipulation — move/rotate/scale mutate the selected object's TRS.
    manip = Manipulation(snapping=Snapping())
    manip.set_object(polygon)
    before_pos = polygon.position.copy()
    manip.update_move((PANEL_W // 2 + 10, PANEL_H // 2 + 5), PANEL_W, PANEL_H)
    manip.update_move((PANEL_W // 2 + 20, PANEL_H // 2 + 10), PANEL_W, PANEL_H)
    check("manipulation move updates position", not np.array_equal(polygon.position, before_pos))

    before_rot = polygon.rotation.copy()
    manip.update_rotate_free((0.02, -0.03))
    manip.update_rotate_free((0.05, -0.01))
    check("manipulation rotate updates rotation", not np.array_equal(polygon.rotation, before_rot))

    manip.in_scale_mode = True
    before_scale = polygon.scale[0]
    manip.update_scale_peace(120.0)
    manip.update_scale_peace(150.0)
    check("manipulation scale updates scale", abs(polygon.scale[0] - before_scale) > 1e-6)

    # 7. Delete path — remove_object()/clear_objects() (fist-hold -> renderer.remove_object).
    renderer.remove_object(polygon)
    check("remove_object drops polygon from objects", polygon not in renderer.objects and len(renderer.objects) == 1)
    renderer.clear_objects()
    check("clear_objects empties scene", len(renderer.objects) == 0)

    failures = [name for name, ok in checks if not ok]
    print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
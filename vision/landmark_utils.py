import math
import numpy as np

WRIST        = 0
THUMB_TIP    = 4
INDEX_MCP    = 5
INDEX_PIP    = 6
INDEX_TIP    = 8
MIDDLE_MCP   = 9
MIDDLE_PIP   = 10
MIDDLE_TIP   = 12
RING_MCP     = 13
RING_TIP     = 16
PINKY_MCP    = 17
PINKY_TIP    = 20
THUMB_IP     = 3


def is_pinch(landmarks, threshold=40):
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    d = math.hypot(thumb[0]-index[0], thumb[1]-index[1])
    return d < threshold


def is_index_only(landmarks):
   
    index_up   = landmarks[INDEX_TIP][1]  < landmarks[INDEX_MCP][1]
    middle_down= landmarks[MIDDLE_TIP][1] > landmarks[MIDDLE_PIP][1]
    ring_down  = landmarks[RING_TIP][1]   > landmarks[RING_MCP][1]
    pinky_down = landmarks[PINKY_TIP][1]  > landmarks[PINKY_MCP][1]

    # thumb must not be pinching index
    thumb_clear = math.hypot(
        landmarks[THUMB_TIP][0] - landmarks[INDEX_TIP][0],
        landmarks[THUMB_TIP][1] - landmarks[INDEX_TIP][1]
    ) > 35

    return index_up and middle_down and ring_down and pinky_down and thumb_clear


def is_open_palm(landmarks):
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [2, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return all(landmarks[t][1] < landmarks[m][1]
               for t, m in zip(tips, mcps))


def is_closed_fist(landmarks):
    tips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return all(landmarks[t][1] > landmarks[m][1]
               for t, m in zip(tips, mcps))


def is_peace_sign(landmarks):
   
    index_up   = landmarks[INDEX_TIP][1]  < landmarks[INDEX_MCP][1]
    middle_up  = landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_MCP][1]
    ring_down  = landmarks[RING_TIP][1]   > landmarks[RING_MCP][1]
    pinky_down = landmarks[PINKY_TIP][1]  > landmarks[PINKY_MCP][1]
    thumb_clear = math.hypot(
        landmarks[THUMB_TIP][0] - landmarks[INDEX_TIP][0],
        landmarks[THUMB_TIP][1] - landmarks[INDEX_TIP][1]
    ) > 40
    return index_up and middle_up and ring_down and pinky_down and thumb_clear


def get_hand_tilt_vector(landmarks):
    
    wx, wy = landmarks[WRIST][0],      landmarks[WRIST][1]
    mx, my = landmarks[MIDDLE_MCP][0], landmarks[MIDDLE_MCP][1]
    dx, dy = mx - wx, my - wy
    length = math.hypot(dx, dy)
    if length < 0.001:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def get_peace_spread(landmarks):
    
    ix, iy = landmarks[INDEX_TIP][0],  landmarks[INDEX_TIP][1]
    mx, my = landmarks[MIDDLE_TIP][0], landmarks[MIDDLE_TIP][1]
    return math.hypot(ix - mx, iy - my)


def palm_center(landmarks):
    cx = int((landmarks[WRIST][0] + landmarks[MIDDLE_MCP][0]) / 2)
    cy = int((landmarks[WRIST][1] + landmarks[MIDDLE_MCP][1]) / 2)
    return (cx, cy)
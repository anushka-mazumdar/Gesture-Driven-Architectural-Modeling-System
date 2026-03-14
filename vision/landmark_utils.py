import numpy as np
import math
import mediapipe as mp


# ---------- Landmark Indices ----------
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
WRIST = 0
INDEX_TIP = 8
INDEX_PIP = 6

MIDDLE_TIP = 12
MIDDLE_PIP = 10

RING_TIP = 16
RING_PIP = 14

PINKY_TIP = 20
PINKY_PIP = 18

# ---------- Utility ----------

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def palm_center(landmarks):
    return np.mean([
        landmarks[0],
        landmarks[5],
        landmarks[9],
        landmarks[13],
        landmarks[17]
    ], axis=0).astype(int)

# ---------- Gesture Detection ----------

def is_pinch(landmarks, threshold=0.35):
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]

    thumb_index_dist = distance(thumb, index)

    palm_size = distance(wrist, index_mcp)

    if palm_size == 0:
        return False

    ratio = thumb_index_dist / palm_size

    return ratio < threshold

def is_open_palm(landmarks):

    fingers_extended = 0

    if landmarks[INDEX_TIP][1] < landmarks[INDEX_PIP][1]:
        fingers_extended += 1

    if landmarks[MIDDLE_TIP][1] < landmarks[MIDDLE_PIP][1]:
        fingers_extended += 1

    if landmarks[RING_TIP][1] < landmarks[RING_PIP][1]:
        fingers_extended += 1

    if landmarks[PINKY_TIP][1] < landmarks[PINKY_PIP][1]:
        fingers_extended += 1

    return fingers_extended >= 3

def is_closed_fist(landmarks):

    fingers_folded = 0

    if landmarks[INDEX_TIP][1] > landmarks[INDEX_PIP][1]:
        fingers_folded += 1

    if landmarks[MIDDLE_TIP][1] > landmarks[MIDDLE_PIP][1]:
        fingers_folded += 1

    if landmarks[RING_TIP][1] > landmarks[RING_PIP][1]:
        fingers_folded += 1

    if landmarks[PINKY_TIP][1] > landmarks[PINKY_PIP][1]:
        fingers_folded += 1

    return fingers_folded >= 3
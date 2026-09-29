import cv2
import json
import sys
import time
from pathlib import Path

# Running this file directly sets sys.path[0] to tools/, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.hand_tracking import HandTracker
from vision.landmark_utils import is_index_only, is_open_palm


# ============================================================
# Configuration
# ============================================================

LABEL = "irregular_polygon"
TARGET_SAMPLES = 200

SAVE_DIR = PROJECT_ROOT / "dataset" / "raw" / LABEL
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def get_next_sample_id():
    """Return the next available sample number."""
    existing = list(SAVE_DIR.glob("sample_*.json"))

    if not existing:
        return 1

    numbers = []

    for file in existing:
        try:
            numbers.append(int(file.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass

    return max(numbers, default=0) + 1


def save_stroke(points, sample_id):
    """Save one raw stroke as JSON."""
    data = {
        "label": LABEL,
        "sample_id": sample_id,
        "points": [[int(x), int(y)] for x, y in points],
        "timestamp": time.time(),
    }

    output_path = SAVE_DIR / f"sample_{sample_id:04d}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path


# ============================================================
# Main
# ============================================================

def main():
    tracker = HandTracker()

    sample_id = get_next_sample_id()

    # If some samples already exist, continue from there.
    completed = sample_id - 1

    drawing = False
    stroke_points = []

    window_name = "Irregular Polygon Dataset Collector"

    print("=" * 50)
    print("IRREGULAR POLYGON DATASET COLLECTOR")
    print("=" * 50)
    print(f"Saving to: {SAVE_DIR}")
    print(f"Target: {TARGET_SAMPLES}")
    print()
    print("Controls:")
    print("  Hold up your index finger to draw")
    print("  Open your palm to save the stroke")
    print("  Press R to discard the current stroke")
    print("  Press Q to quit")
    print("=" * 50)

    while True:
        frame, landmarks = tracker.get_frame()

        if frame is None:
            print("Failed to read webcam frame.")
            break

        # ----------------------------------------------------
        # Hand tracking
        # ----------------------------------------------------

        index_point = None
        index_gesture = False
        open_palm = False

        if landmarks is not None and len(landmarks) == 21:
            index_gesture = is_index_only(landmarks)
            open_palm = is_open_palm(landmarks)

            if index_gesture:
                index_point = (int(landmarks[8][0]), int(landmarks[8][1]))

        if index_gesture and index_point is not None:
            if not drawing:
                drawing = True
                stroke_points = []

            stroke_points.append(index_point)
            cv2.circle(frame, index_point, 8, (0, 255, 0), -1)

        elif open_palm and drawing:
            if len(stroke_points) >= 10:
                save_stroke(stroke_points, sample_id)
                completed += 1
                sample_id += 1
                print(f"Saved irregular polygon {completed}/{TARGET_SAMPLES}")
            else:
                print("Stroke too short; not saved.")

            drawing = False
            stroke_points = []

            if completed >= TARGET_SAMPLES:
                print("Collection complete.")
                break

        # ----------------------------------------------------
        # Draw current stroke
        # ----------------------------------------------------

        if len(stroke_points) > 1:
            for i in range(1, len(stroke_points)):
                cv2.line(
                    frame,
                    stroke_points[i - 1],
                    stroke_points[i],
                    (255, 255, 255),
                    2,
                )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "IRREGULAR POLYGON",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"{completed} / {TARGET_SAMPLES}",
            (
                frame.shape[1] - 180,
                frame.shape[0] - 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "Index finger: draw | Open palm: save | R: discard | Q: quit",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        cv2.imshow(window_name, frame)

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Collection stopped by user.")
            break

        elif key == ord("r"):
            drawing = False
            stroke_points = []
            print("Current stroke discarded.")

    cv2.destroyAllWindows()
    tracker.release()

    print(f"\nSamples saved: {completed}")
    print(f"Location: {SAVE_DIR}")


if __name__ == "__main__":
    main()


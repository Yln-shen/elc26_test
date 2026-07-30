#!/usr/bin/env python3
"""
Interactive pipe calibration tool.

Mark 4 pipe corners, the centre mark, then up to 13 calibration points.
Saves → config/pipe_calib.json  for use by main.py / main_test.py.

Controls
--------
  Click             — place next point
  Backspace / b     — undo last point
  r                 — reset all points
  s                 — save to file
  q / Esc           — quit without saving
"""

import os
import sys
import json
import cv2

# project paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CALIB_PATH = os.path.join(_PROJECT_ROOT, "config", "pipe_calib.json")

# camera
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "vision"))
from camera import Camera

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
POINTS = []          # [(x, y, label)]
LABELS = [
    "TL corner", "TR corner", "BR corner", "BL corner",  # 0–3  corners
    "CENTRE MARK",                                        # 4     0-mm mark
]
CAL_MM_VALUES = []   # mm values entered by user for cal points
NEXT_CAL_MM = None   # current mm being set

FRAME = None
WINDOW = "Pipe Calibration"


# ---------------------------------------------------------------------------
def draw_all():
    """Redraw the frame with all placed points."""
    if FRAME is None:
        return
    canvas = FRAME.copy()
    h, w = canvas.shape[:2]

    # instruction
    total = len(LABELS)
    n = len(POINTS)
    if n < total:
        text = f"Click: {LABELS[n]}  ({n + 1}/{total})"
        color = (0, 255, 255)
    elif NEXT_CAL_MM is not None:
        text = (f"Type mm value + Enter for cal point #{n - total + 1}  "
                f"(current: {NEXT_CAL_MM})")
        color = (0, 255, 0)
    else:
        text = "Press 's' to save, 'r' to reset, 'q' to quit"
        color = (0, 255, 0)
    cv2.putText(canvas, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)
    cv2.putText(canvas, f"Points: {len(POINTS)}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # corners (0–3) — white
    for i in range(min(4, len(POINTS))):
        x, y, _ = POINTS[i]
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(canvas, LABELS[i][:2], (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # centre mark (4) — yellow
    if len(POINTS) > 4:
        x, y, _ = POINTS[4]
        cv2.drawMarker(canvas, (x, y), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    # calibration points (5+) — green
    for i in range(5, len(POINTS)):
        x, y, label = POINTS[i]
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(canvas, label, (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # connect cal points
    cal_pts = [(p[0], p[1]) for p in POINTS[5:]]
    for i in range(1, len(cal_pts)):
        cv2.line(canvas, cal_pts[i - 1], cal_pts[i], (0, 150, 0), 1)

    cv2.imshow(WINDOW, canvas)


# ---------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        n = len(POINTS)
        if n < len(LABELS):
            POINTS.append((x, y, LABELS[n]))
        else:
            # calibration point — prompt for mm value
            POINTS.append((x, y, f"???"))  # placeholder
            prompt_mm_value()
        draw_all()


# ---------------------------------------------------------------------------
def prompt_mm_value():
    """Ask user to type the mm value for the most recent cal point."""
    global NEXT_CAL_MM
    NEXT_CAL_MM = ""
    draw_all()


def set_last_cal_mm(value_str):
    global POINTS, NEXT_CAL_MM
    try:
        mm = float(value_str)
    except ValueError:
        print(f"  Invalid mm: '{value_str}'")
        POINTS.pop()  # remove the placeholder
        NEXT_CAL_MM = None
        draw_all()
        return

    x, y, _ = POINTS[-1]
    POINTS[-1] = (x, y, f"{mm:+.0f} mm")
    print(f"  cal point #{len(POINTS) - 5}:  ({x}, {y})  →  {mm:+.0f} mm")
    NEXT_CAL_MM = None
    draw_all()


# ---------------------------------------------------------------------------
def save():
    """Build and save the calibration file."""
    if len(POINTS) < 6:
        print("  Need at least 4 corners + centre + 1 cal point to save")
        return

    corners = [POINTS[i][:2] for i in range(4)]     # tl, tr, br, bl
    centre = POINTS[4][:2]

    cal = []
    for i in range(5, len(POINTS)):
        x, y, label = POINTS[i]
        # parse mm from label like "+50 mm" or "-100 mm"
        try:
            mm = float(label.replace(" mm", ""))
        except ValueError:
            continue
        cal.append({"x": x, "y": y, "mm": mm})

    data = {
        "pipe_length_mm": 234.0,
        "corners": [list(c) for c in corners],
        "center_mark": list(centre),
        "cal_points": cal,
    }

    os.makedirs(os.path.dirname(_CALIB_PATH), exist_ok=True)
    with open(_CALIB_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved → {_CALIB_PATH}")
    print(f"  {len(cal)} calibration points")


# ---------------------------------------------------------------------------
def main():
    global FRAME

    cam = Camera(index=0, width=640, height=480, fps=30)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print("=" * 55)
    print("Pipe Calibration Tool")
    print("  Click:  TL corner → TR → BR → BL → CENTRE MARK")
    print("  Then click cal points (type mm when prompted)")
    print("  Backspace: undo   s: save   q: quit")
    print("=" * 55)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        FRAME = frame
        draw_all()

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:        # q / Esc
            break
        elif key == ord('s'):
            save()
        elif key == ord('r'):
            POINTS.clear()
            print("  reset")
        elif key == 8 or key == ord('b'):       # Backspace / b
            if POINTS:
                removed = POINTS.pop()
                print(f"  removed: {removed[2]}")
                NEXT_CAL_MM = None
            draw_all()
        elif key == 13:                          # Enter
            if NEXT_CAL_MM is not None and NEXT_CAL_MM:
                set_last_cal_mm(NEXT_CAL_MM)
        elif key != 255:                         # printable
            if NEXT_CAL_MM is not None:
                ch = chr(key)
                if ch in "0123456789.+-":
                    NEXT_CAL_MM += ch
                    draw_all()

    cam.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

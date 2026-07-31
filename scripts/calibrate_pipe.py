#!/usr/bin/env python3
"""
Keyboard-driven pipe calibration tool (no mouse required).

Arrow keys move a crosshair cursor.  Space / Enter places a point.

Controls
--------
  ↑ ↓ ← →        — move cursor (hold Shift = 10 px step)
  Space / Enter   — place current point
  Backspace       — undo last point
  r               — reset all
  s               — save to config/pipe_calib.json
  q / Esc         — quit
"""

import os
import sys
import json
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIB_PATH = os.path.join(_PROJECT_ROOT, "config", "pipe_calib.json")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "vision"))
from camera import Camera

# ---- pre-defined calibration-point mm sequence ----
CAL_MM_ORDER = [0, -10, 10, -50, -60, -40, 50, 40, 60, -122, 114, -80, 80]

# ---- state ----
POINTS = []          # [(x, y, label)]
LABELS = [
    "TL corner", "TR corner", "BR corner", "BL corner",
    "CENTRE MARK",
]
FRAME = None
CURSOR = (320, 240)  # current crosshair position
WINDOW = "Pipe Calibration"


# ---------------------------------------------------------------------------
def draw_all():
    global FRAME
    if FRAME is None:
        return
    canvas = FRAME.copy()
    n = len(POINTS)
    total = len(LABELS) + len(CAL_MM_ORDER)

    # ---- hint ----
    if n < len(LABELS):
        hint = f">>> [{n + 1}/{len(LABELS)}]  Place cursor on: {LABELS[n]}"
        color = (0, 255, 255)
    else:
        idx = n - len(LABELS)
        if idx < len(CAL_MM_ORDER):
            hint = (f">>> [{idx + 1}/{len(CAL_MM_ORDER)}]  "
                    f"Ball at  {CAL_MM_ORDER[idx]:+d} mm  →  move cursor + Space")
            color = (0, 255, 0)
        else:
            hint = "ALL DONE — 's' to save, 'q' to quit"
            color = (0, 255, 255)
    cv2.putText(canvas, hint, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1)

    # ---- progress bar ----
    done = min(n, total)
    bar_w = 200
    cv2.rectangle(canvas, (10, 34), (10 + bar_w, 42), (60, 60, 60), -1)
    cv2.rectangle(canvas, (10, 34), (10 + bar_w * done // total, 42),
                  (0, 200, 0), -1)

    # ---- placed points ----
    for i in range(min(4, n)):
        x, y, _ = POINTS[i]
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(canvas, LABELS[i][:2], (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    if n > 4:
        x, y, _ = POINTS[4]
        cv2.drawMarker(canvas, (x, y), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    for i in range(5, n):
        x, y, label = POINTS[i]
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(canvas, label, (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    cal_xy = [(p[0], p[1]) for p in POINTS[5:]]
    for i in range(1, len(cal_xy)):
        cv2.line(canvas, cal_xy[i - 1], cal_xy[i], (0, 100, 0), 1)

    # ---- crosshair cursor ----
    cx, cy = CURSOR
    cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 1)
    cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 1)
    cv2.putText(canvas, f"({cx},{cy})", (cx + 12, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow(WINDOW, canvas)


# ---------------------------------------------------------------------------
def place_point():
    """Place a point at the current cursor position."""
    n = len(POINTS)
    if n < len(LABELS):
        POINTS.append((CURSOR[0], CURSOR[1], LABELS[n]))
    else:
        idx = n - len(LABELS)
        if idx < len(CAL_MM_ORDER):
            mm = CAL_MM_ORDER[idx]
            POINTS.append((CURSOR[0], CURSOR[1], f"{mm:+d} mm"))
            print(f"  [{idx + 1}/{len(CAL_MM_ORDER)}]  "
                  f"({CURSOR[0]}, {CURSOR[1]})  →  {mm:+d} mm")
        else:
            return


def save():
    if len(POINTS) < 6:
        print("[!] Need >= 4 corners + centre + 1 cal point")
        return
    corners = [POINTS[i][:2] for i in range(4)]
    centre = POINTS[4][:2]
    cal = []
    for i in range(5, len(POINTS)):
        x, y, label = POINTS[i]
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
    print(f"\n[OK] Saved {len(cal)} cal points → {_CALIB_PATH}")


# ---------------------------------------------------------------------------
def main():
    global FRAME, CURSOR

    cam = Camera(index=0, width=640, height=480, fps=30)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)

    seq_str = "  →  ".join(f"{m:+d}" for m in CAL_MM_ORDER)
    print("=" * 55)
    print("Pipe Calibration Tool  (keyboard-driven)")
    print("  Arrows = move   Space/Enter = place   Shift+Arrow = fast")
    print("  Backspace = undo   r = reset   s = save   q = quit")
    print(f"  Sequence: {seq_str}")
    print("=" * 55)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        FRAME = frame
        draw_all()

        key = cv2.waitKey(30) & 0xFF

        # ---- modifiers (shift) ----
        # cv2.waitKey doesn't expose Shift for arrow keys cleanly on all
        # platforms, so we use a faster step when a "fast" key combo is held.
        # We detect the raw keycode and check common arrow ranges.
        step = 1

        if key == ord('q') or key == 27:        # q / Esc
            break
        elif key == ord('s'):
            save()
        elif key == ord('r'):
            POINTS.clear()
            print("  [reset]")
        elif key == 8 or key == ord('b'):        # Backspace / b
            if POINTS:
                removed = POINTS.pop()
                print(f"  [undo] removed {removed[2]}")
        elif key == 13 or key == 32:             # Enter / Space
            place_point()
        # ---- arrows (OpenCV keycodes on Linux) ----
        elif key == 81:                          # left
            CURSOR = (max(0, CURSOR[0] - step), CURSOR[1])
        elif key == 82:                          # up
            CURSOR = (CURSOR[0], max(0, CURSOR[1] - step))
        elif key == 83:                          # right
            CURSOR = (min(FRAME.shape[1] - 1, CURSOR[0] + step), CURSOR[1])
        elif key == 84:                          # down
            CURSOR = (CURSOR[0], min(FRAME.shape[0] - 1, CURSOR[1] + step))
        # ---- Shift+arrow (fast move, 10 px) ----
        elif key == 0x61:
            CURSOR = (max(0, CURSOR[0] - 10), CURSOR[1])
        elif key == 0x62:
            CURSOR = (CURSOR[0], max(0, CURSOR[1] - 10))
        elif key == 0x63:
            CURSOR = (min(FRAME.shape[1] - 1, CURSOR[0] + 10), CURSOR[1])
        elif key == 0x64:
            CURSOR = (CURSOR[0], min(FRAME.shape[0] - 1, CURSOR[1] + 10))

    cam.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

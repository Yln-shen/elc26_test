#!/usr/bin/env python3
"""
Interactive pipe calibration tool.

Mark 4 pipe corners + centre mark by clicking, then place calibration
points (type mm value in the *terminal*, not the image window).

Controls (in the IMAGE WINDOW)
--------
  Click       — place next point
  Backspace   — undo last point
  r key       — reset all
  s key       — save to config/pipe_calib.json
  q / Esc     — quit

After clicking a calibration point the terminal asks:
  Enter mm for point #N (e.g. -50 or +75) >
"""

import os
import sys
import json
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIB_PATH = os.path.join(_PROJECT_ROOT, "config", "pipe_calib.json")

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "vision"))
from camera import Camera

# ---------------------------------------------------------------------------
POINTS = []          # [(x, y, label)]
LABELS = [
    "TL corner", "TR corner", "BR corner", "BL corner",
    "CENTRE MARK",
]
FRAME = None
WINDOW = "Pipe Calibration — click to mark"


def draw_all():
    if FRAME is None:
        return
    canvas = FRAME.copy()

    n = len(POINTS)
    if n < len(LABELS):
        hint = f">>> Click: {LABELS[n]}  ({n + 1}/{len(LABELS)})"
        color = (0, 255, 255)
    else:
        hint = ">>> Click cal point, then type mm in TERMINAL (s=save q=quit)"
        color = (0, 255, 0)

    cv2.putText(canvas, hint, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1)
    cv2.putText(canvas, f"Points: {len(POINTS)}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # corners — white
    for i in range(min(4, n)):
        x, y, _ = POINTS[i]
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)
        cv2.putText(canvas, LABELS[i][:2], (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # centre mark — yellow cross
    if n > 4:
        x, y, _ = POINTS[4]
        cv2.drawMarker(canvas, (x, y), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    # cal points — green
    for i in range(5, n):
        x, y, label = POINTS[i]
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(canvas, label, (x + 8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # connect cal points
    cal_xy = [(p[0], p[1]) for p in POINTS[5:]]
    for i in range(1, len(cal_xy)):
        cv2.line(canvas, cal_xy[i - 1], cal_xy[i], (0, 120, 0), 1)

    cv2.imshow(WINDOW, canvas)


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    n = len(POINTS)
    if n < len(LABELS):
        POINTS.append((x, y, LABELS[n]))
    else:
        POINTS.append((x, y, "???"))
    draw_all()


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
    global FRAME

    cam = Camera(index=0, width=640, height=480, fps=30)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    print("=" * 55)
    print("Pipe Calibration Tool")
    print("  1. Click: TL → TR → BR → BL corner → CENTRE MARK")
    print("  2. Click cal points → terminal asks for mm value")
    print("  Backspace=undo  r=reset  s=save  q=quit")
    print("=" * 55)

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        FRAME = frame
        draw_all()

        # ---- handle newly-added cal point (prompt in terminal) ----
        if len(POINTS) > len(LABELS):
            last = POINTS[-1]
            if last[2] == "???":   # not yet labelled
                try:
                    raw = input(
                        f"  Enter mm for cal point #{len(POINTS) - len(LABELS)} "
                        f"at ({last[0]}, {last[1]}): "
                    ).strip()
                    mm = float(raw)
                    POINTS[-1] = (last[0], last[1], f"{mm:+.0f} mm")
                    print(f"    → set to {mm:+.0f} mm")
                except (ValueError, EOFError):
                    POINTS.pop()
                    print("    cancelled")
                draw_all()

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            save()
        elif key == ord('r'):
            POINTS.clear()
            print("  [reset]")
        elif key == 8 or key == ord('b'):
            if POINTS:
                removed = POINTS.pop()
                print(f"  [undo] removed {removed[2]}")
            draw_all()

    cam.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

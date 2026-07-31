import os
import cv2
import time
from detector import Detector
from camera import Camera
from tracker import Tracker
from urat import UARTSender

# ---- mode ----
HEADLESS = False          # True = no display, pure detect+track+uart (max fps)

# ---- paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')
_CALIB = os.path.join(_PROJECT_ROOT, 'config', 'pipe_calib.json')

# ---- ROI + calibration (tune after rig is assembled) ----
PIPE_ROI = (0, 180, 640, 200)

# ---- init ----
try:
    cam = Camera(index=0, width=640, height=480, fps=120)
except Exception:
    cam = Camera(index=1)

# CENTER_OFFSET = (calib_center_mark - image_center)
# calib centre = (299, 256),  image centre = (320, 240) → offset = (-21, +16)
CENTER_OFFSET = (-21, 16)

detector = Detector(model_path=_MODEL, conf_threshold=0.05, roi=PIPE_ROI,
                    center_offset=CENTER_OFFSET)
tracker = Tracker(calib_path=_CALIB, use_kf=True, frame_add=35, Q_base=2.0, R=0.05)

# ---- UART ----
UART_PORT = '/dev/ttyS3'
uart = UARTSender(port=UART_PORT, baud=115200)
uart_ok = uart.open()
if uart_ok:
    uart.start(rate_hz=1000)

print("=" * 60)
print("Steel Ball Tracker — 1D Kalman + mm output")
print(f"Calib:  {tracker.calib.info()}")
print(f"Kalman: Q_base=2.0  R=0.05  frame_add=35")
print(f"UART:   {UART_PORT}  {'1 kHz' if uart_ok else 'CLOSED'}")
print("Press 'q' to quit, 'r' to reset")
print("=" * 60)

# ---- stats ----
fps_last = 0
fps_timer = time.perf_counter()
frame_count = 0
total_latency = 0.0

if not HEADLESS:
    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracker", 640, 480)

while True:
    t0 = time.perf_counter()

    ret, frame = cam.read()
    if not ret:
        break

    # ---- detect → track ----
    ball_center = detector.detect(frame)
    pos_mm = tracker.track(ball_center)

    # ---- UART ----
    if pos_mm is not None:
        uart.update(pos_mm)

    t1 = time.perf_counter()
    total_latency += (t1 - t0) * 1000  # ms
    frame_count += 1

    if time.perf_counter() - fps_timer >= 1.0:
        avg_ms = total_latency / frame_count
        fps_last = frame_count
        print(f"  fps: {fps_last}  |  {avg_ms:5.1f} ms/frame  |  pos: {pos_mm:+.1f} mm" if pos_mm is not None else f"  fps: {fps_last}  |  {avg_ms:5.1f} ms/frame  |  LOST")
        frame_count = 0
        total_latency = 0.0
        fps_timer = time.perf_counter()

    if HEADLESS:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ---- draw ----
    detector.draw(frame)
    tracker.draw_debug(frame)

    cv2.putText(frame, f"FPS: {fps_last}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if pos_mm is not None:
        tag = "PRED" if tracker.is_predicting() else "TRACK"
        color = (0, 200, 255) if tracker.is_predicting() else (0, 255, 0)
        cv2.putText(frame, f"{tag}  {pos_mm:+.1f} mm", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    tx_x = frame.shape[1] - 55
    if uart_ok:
        cv2.circle(frame, (tx_x - 10, 20), 5, (0, 255, 0), -1)
        cv2.putText(frame, "TX 1k", (tx_x, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.circle(frame, (tx_x - 10, 20), 5, (0, 0, 255), -1)
        cv2.putText(frame, "TX OFF", (tx_x, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.imshow("Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracker._reset()
        print("Tracker reset")

cam.cam.release()
uart.stop()
cv2.destroyAllWindows()
# main_test.py — headless benchmark: detect + track + uart, fps to terminal
import os
import time
from detector import Detector
from camera import Camera
from tracker import Tracker
from urat import UARTSender

# ---- paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')
_CALIB = os.path.join(_PROJECT_ROOT, 'config', 'pipe_calib.json')

# ---- ROI + calibration ----
PIPE_ROI = (5, 195, 630, 120)    # ~3x pipe height, trim background

# ---- init ----
try:
    cam = Camera(index=0, width=640, height=480, fps=120)
except Exception:
    cam = Camera(index=1)

detector = Detector(model_path=_MODEL, conf_threshold=0.05, roi=PIPE_ROI)
tracker = Tracker(calib_path=_CALIB, use_kf=True, frame_add=35, Q_base=2.0, R=0.05)

UART_PORT = '/dev/ttyS3'
uart = UARTSender(port=UART_PORT, baud=115200)
uart_ok = uart.open()
if uart_ok:
    uart.start(rate_hz=1000)

print("=" * 50)
print("Headless — detect + track + UART only")
print(f"Model:  [160, 512]  |  UART: {'1 kHz' if uart_ok else 'OFF'}")
print(f"Calib:  {tracker.calib.info()}")
print("Press  Ctrl-C  to stop")
print("=" * 50)

# ---- FPS ----
fps_timer = time.perf_counter()
frame_count = 0
track_count = 0
lost_count = 0

total_latency = 0.0

try:
    while True:
        t0 = time.perf_counter()

        ret, frame = cam.read()
        if not ret:
            break

        ball_center = detector.detect(frame)
        pos_mm = tracker.track(ball_center)

        if pos_mm is not None:
            uart.update(pos_mm)
            track_count += 1
        else:
            lost_count += 1

        t1 = time.perf_counter()
        total_latency += (t1 - t0) * 1000  # ms

        frame_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            avg_ms = total_latency / frame_count
            track_pct = track_count / frame_count * 100 if frame_count else 0
            print(f"  fps: {fps:5.1f}  |  {avg_ms:5.1f} ms/frame  |  "
                  f"track: {track_pct:4.0f}%  |  "
                  f"pos: {pos_mm:+6.1f} mm" if pos_mm is not None else
                  f"  fps: {fps:5.1f}  |  {avg_ms:5.1f} ms/frame  |  "
                  f"track: {track_pct:4.0f}%  |  LOST")
            frame_count = 0
            track_count = 0
            lost_count = 0
            total_latency = 0.0
            fps_timer = time.perf_counter()

except KeyboardInterrupt:
    pass
finally:
    print("\n[DONE]")
    cam.cam.release()
    uart.stop()

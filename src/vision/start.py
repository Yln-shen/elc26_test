# start.py — headless production: detect → track → uart
import os
import time
from detector import Detector
from camera import Camera
from tracker import Tracker
from urat import UARTSender
from gpio import GPIO

# ---- paths ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')
_CALIB = os.path.join(_PROJECT_ROOT, 'config', 'pipe_calib.json')

# ---- ROI + calibration ----
PIPE_ROI = (5, 215, 630, 80)
CENTER_OFFSET = (-21, 16)

# ---- init ----
try:
    cam = Camera(index=0, width=640, height=480, fps=120)
except Exception:
    cam = Camera(index=1)

detector = Detector(model_path=_MODEL, conf_threshold=0.05, roi=PIPE_ROI,
                    center_offset=CENTER_OFFSET)
tracker = Tracker(calib_path=_CALIB, use_kf=True, frame_add=35, Q_base=2.0, R=0.05)

gpio = GPIO(chip_path='/dev/gpiochip4', line_offset=22, consumer='my-led')

# ---- UART ----
UART_PORT = '/dev/ttyS3'
uart = UARTSender(port=UART_PORT, baud=115200)
uart_ok = uart.open()
if uart_ok:
    uart.start(rate_hz=1000)

print("=" * 50)
# print("elc26  headless  production")
print(f"calib: {tracker.calib.info()}")
print(f"uart:  {'1 kHz' if uart_ok else 'OFF'}")
# print("=" * 50)

# ---- main loop ----
fps_timer = time.perf_counter()
frame_count = 0
total_latency = 0.0
pos_mm = None
tx_prev = 0

try:
    while True:
        gpio.on()
        t0 = time.perf_counter()

        ret, frame = cam.read()
        if not ret:
            break

        ball_center = detector.detect(frame)
        pos_mm = tracker.track(ball_center)

        if pos_mm is not None and uart_ok:
            uart.update(pos_mm)

        t1 = time.perf_counter()
        total_latency += (t1 - t0) * 1000
        frame_count += 1

        if time.perf_counter() - fps_timer >= 1.0:
            avg_ms = total_latency / frame_count
            fps = frame_count
            tx_now = uart.tx_count if uart_ok else 0
            tx_rate = tx_now - tx_prev
            tx_prev = tx_now
            if pos_mm is not None:
                print(f"  fps:{fps}  {avg_ms:5.1f}ms  tx:{tx_rate:4d}/s  pos:{pos_mm:+.1f}mm")
            else:
                print(f"  fps:{fps}  {avg_ms:5.1f}ms  tx:{tx_rate:4d}/s  LOST")
            frame_count = 0
            total_latency = 0.0
            fps_timer = time.perf_counter()
finally:
    print("shutting down ...")
    gpio.off()
    gpio.release()
    cam.cam.release()
    uart.stop()
    print("done")

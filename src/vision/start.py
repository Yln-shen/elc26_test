#!/usr/bin/env python3
"""
Production startup script — headless, auto-start, GPIO heartbeat.

- No display / no remote needed
- GPIO LED blinks while tracking (heartbeat)
- Logs to ~/elc26_test/logs/
- Auto-retries camera + UART on failure
- systemd restarts this process if it dies
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path

# project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---- log directory ----
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"start_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("start")

# ---- imports (after log setup so import errors are captured) ----
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "vision"))
from detector import Detector
from camera import Camera
from tracker import Tracker
from urat import UARTSender

# ---- GPIO heartbeat ----
GPIO_CHIP = "/dev/gpiochip4"
GPIO_LINE = 22
_gpio = None

try:
    import gpiod

    class GPIO:
        def __init__(self, chip_path, line_offset):
            self.chip = gpiod.Chip(chip_path)
            settings = gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT
            )
            self.request = self.chip.request_lines(
                config={line_offset: settings},
                consumer="elc26-heartbeat",
            )
            self._offset = line_offset

        def on(self):
            self.request.set_value(self._offset, gpiod.line.Value.ACTIVE)

        def off(self):
            self.request.set_value(self._offset, gpiod.line.Value.INACTIVE)

        def set(self, v):
            self.on() if v else self.off()

        def release(self):
            self.request.release()
            self.chip.close()

    _gpio = GPIO(GPIO_CHIP, GPIO_LINE)
    logger.info(f"GPIO  heartbeat  on  {GPIO_CHIP}:{GPIO_LINE}")
except Exception as e:
    logger.warning(f"GPIO  unavailable: {e}  — heartbeat disabled")


def heartbeat():
    """Blink GPIO LED in background thread."""
    state = False
    while _heartbeat_run:
        state = not state
        try:
            _gpio.set(state)
        except Exception:
            pass
        time.sleep(0.5)


_heartbeat_run = True
_heartbeat_thread = None
if _gpio is not None:
    _heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    _heartbeat_thread.start()

# ---- shutdown handler ----
_running = True


def shutdown(signum=None, frame=None):
    global _running, _heartbeat_run
    logger.info("shutdown  signal received")
    _running = False
    _heartbeat_run = False


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# ====================================================================
# configuration
# ====================================================================
_MODEL = str(_PROJECT_ROOT / "src" / "yolo_det" / "best_rknn_model")
_CALIB = str(_PROJECT_ROOT / "config" / "pipe_calib.json")
PIPE_ROI = (5, 215, 630, 80)
CENTER_OFFSET = (-21, 16)
UART_PORT = "/dev/ttyS3"

CAMERA_RETRY_SEC = 2.0        # retry interval for camera
CAMERA_TIMEOUT_SEC = 30.0     # give up after this long
UART_RETRY_SEC = 1.0
UART_TIMEOUT_SEC = 10.0

# ====================================================================
# init  (with retry)
# ====================================================================


def init_camera():
    deadline = time.time() + CAMERA_TIMEOUT_SEC
    while time.time() < deadline and _running:
        for idx in (0, 1):
            try:
                cam = Camera(index=idx, width=640, height=480, fps=120)
                logger.info(f"camera  open  /dev/video{idx}")
                return cam
            except Exception:
                pass
        logger.warning(f"camera  retry  in  {CAMERA_RETRY_SEC}s ...")
        time.sleep(CAMERA_RETRY_SEC)
    return None


def init_uart():
    deadline = time.time() + UART_TIMEOUT_SEC
    while time.time() < deadline and _running:
        try:
            uart = UARTSender(port=UART_PORT, baud=115200)
            if uart.open():
                uart.start(rate_hz=1000)
                logger.info(f"uart  open  {UART_PORT}  @ 1 kHz")
                return uart
        except Exception:
            pass
        logger.warning(f"uart  retry  in  {UART_RETRY_SEC}s ...")
        time.sleep(UART_RETRY_SEC)
    return None


# ====================================================================
# main
# ====================================================================

logger.info("=" * 45)
logger.info("elc26  start  —  headless production mode")
logger.info(f"model : {_MODEL}")
logger.info(f"calib : {_CALIB}")
logger.info(f"uart  : {UART_PORT}")
logger.info(f"logs  : {_LOG_FILE}")
logger.info("=" * 45)

# --- camera ---
cam = init_camera()
if cam is None:
    logger.error("camera  FAILED  —  exiting")
    sys.exit(1)

# --- detector + tracker ---
detector = Detector(model_path=_MODEL, conf_threshold=0.05,
                    roi=PIPE_ROI, center_offset=CENTER_OFFSET)
tracker = Tracker(calib_path=_CALIB, use_kf=True,
                  frame_add=35, Q_base=2.0, R=0.05)
logger.info(f"calib  {tracker.calib.info()}")

# --- UART ---
uart = init_uart()
if uart is None:
    logger.warning("uart  FAILED  —  continuing without TX")

# --- heartbeat LED: solid ON = running ---
if _gpio is not None:
    _heartbeat_run = False   # stop blinking
    if _heartbeat_thread:
        _heartbeat_thread.join(timeout=1.0)
    _gpio.on()
    logger.info("GPIO  SOLID ON  —  tracking active")

# ====================================================================
# main loop
# ====================================================================
fps_timer = time.perf_counter()
frame_count = 0
total_latency = 0.0
pos_mm = None

logger.info("loop  started")

while _running:
    t0 = time.perf_counter()

    ret, frame = cam.read()
    if not ret:
        logger.error("camera  read  failed")
        break

    ball_center = detector.detect(frame)
    pos_mm = tracker.track(ball_center)

    if pos_mm is not None and uart is not None and uart.is_open:
        uart.update(pos_mm)

    t1 = time.perf_counter()
    total_latency += (t1 - t0) * 1000
    frame_count += 1

    if time.perf_counter() - fps_timer >= 5.0:
        avg_ms = total_latency / frame_count if frame_count else 0
        fps = frame_count / 5.0
        if pos_mm is not None:
            logger.info(f"fps:{fps:5.1f}  avg:{avg_ms:5.1f}ms  pos:{pos_mm:+.1f}mm")
        else:
            logger.info(f"fps:{fps:5.1f}  avg:{avg_ms:5.1f}ms  LOST")
        frame_count = 0
        total_latency = 0.0
        fps_timer = time.perf_counter()

# ====================================================================
# cleanup
# ====================================================================
logger.info("cleaning  up  ...")

if _gpio is not None:
    _gpio.off()
    _gpio.release()

cam.cam.release()
if uart is not None:
    uart.stop()

logger.info("exit  (0)")
sys.exit(0)

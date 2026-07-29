# src/vision/tracker.py
# 1D steel-ball tracker — pixel-space Kalman → mm conversion for motor control
import time
import cv2
from Kalman import AdaptiveEKF1D


def time_diff(last_time=[None]):
    """Seconds since last call (mutable-default singleton)."""
    now = time.time_ns()
    if last_time[0] is None:
        last_time[0] = now
        return 1e-9
    diff = now - last_time[0]
    last_time[0] = now
    return diff / 1e9


class Tracker:
    """
    1D position tracker for a steel ball rolling along a horizontal pipe.

    Pipeline
    --------
    detector  ──→  ball_center_x (px)  ──→  AdaptiveEKF1D  ──→  px → mm
                (or None when missed)

    Features
    --------
    - Kalman-filtered 1D position along the pipe (pixel space)
    - Predict-only extrapolation when detection drops for 1-2 frames
      → motor receives smooth, continuous signal — no sudden jumps
    - Pipe calibration:  mark left & right ends in the image → automatic
      pixel-to-mm conversion (left = positive, right = negative)
    - Zero PnP / 3D dependency — only needs the ball's x-coordinate
    """

    def __init__(self, use_kf=True, frame_add=35,
                 Q_base=2.0, R=0.05):
        """
        Args:
            use_kf:    enable Kalman filtering
            frame_add: how many consecutive missed frames to predict before
                       declaring "lost" and resetting
            Q_base:    process-noise spectral density (px²/s³).
                       Higher → trusts prediction more, faster response.
            R:         measurement-noise variance (px²).
                       Lower  → trusts the detector more.
        """
        self.use_kf = use_kf
        self.frame_add = frame_add

        # ---- single 1D Kalman filter (position along pipe) ----
        self.kf = AdaptiveEKF1D(Q_base=Q_base, R=R, dt=1 / 120.0)

        # ---- pipe calibration ----
        self.pipe_left_px = None
        self.pipe_right_px = None
        self.pipe_center_px = None
        self.mm_per_pixel = None         # scale factor

        # ---- current state ----
        self.position_px = None          # filtered position (pixels)
        self.position_mm = None          # filtered position (mm, left = +)
        self.raw_position_px = None      # raw measurement (pixels)
        self.raw_position_mm = None      # raw measurement (mm)

        # ---- tracking status ----
        self.lost_frames = 0
        self.if_find = False             # output is usable right now
        self.predict = False             # current output is prediction-only

    # ==================================================================
    # Calibration
    # ==================================================================

    def set_pipe_calibration(self, left_px, right_px):
        """
        Register the pipe's pixel-space endpoints.

        The pipe is physically **250 mm** long (25 cm).  After calling
        this, pixel positions are automatically converted to mm with the
        sign convention:

            centre ── left  →  positive mm
            centre ── right →  negative mm

        Args:
            left_px:  x-coordinate of the pipe's left  end in the image
            right_px: x-coordinate of the pipe's right end in the image
        """
        self.pipe_left_px = float(left_px)
        self.pipe_right_px = float(right_px)
        self.pipe_center_px = (left_px + right_px) / 2.0

        span_px = right_px - left_px
        if span_px <= 0:
            raise ValueError(f"right_px ({right_px}) must be > left_px ({left_px})")
        self.mm_per_pixel = 250.0 / span_px

    # ==================================================================
    # Core tracking
    # ==================================================================

    def track(self, ball_center_x):
        """
        Feed one detection (or None) into the tracker.

        Args:
            ball_center_x:  ball centre **x**-coordinate in pixels,
                            or ``None`` when the detector found nothing.

        Returns:
            float | None:  filtered position in **mm** (left = positive),
            or ``None`` when the tracker has given up (lost > frame_add).
        """
        dt = time_diff()

        # ---------------------------------------------------------------
        # CASE 1 — no detection
        # ---------------------------------------------------------------
        if ball_center_x is None:
            self.raw_position_px = None
            self.raw_position_mm = None
            self.lost_frames += 1

            if (self.use_kf and self.kf.is_initialized
                    and self.lost_frames <= self.frame_add):
                # predict-only — smooth extrapolation
                pred_px = self.kf.predict(dt)
                self.position_px = pred_px
                self.position_mm = self._px_to_mm(pred_px)
                self.predict = True
                self.if_find = True
                return self.position_mm
            else:
                self._reset()
                return None

        # ---------------------------------------------------------------
        # CASE 2 — detection received
        # ---------------------------------------------------------------
        self.raw_position_px = ball_center_x
        self.lost_frames = 0
        self.predict = False
        self.if_find = True

        if self.use_kf:
            if not self.kf.is_initialized:
                self.kf.set_initial_state(ball_center_x)
                self.position_px = ball_center_x
            else:
                self.kf.predict(dt)
                self.position_px = self.kf.update(ball_center_x)
        else:
            self.position_px = ball_center_x

        self.raw_position_mm = self._px_to_mm(ball_center_x)
        self.position_mm = self._px_to_mm(self.position_px)

        return self.position_mm

    # ==================================================================
    # Public getters (motor / UART consumer reads these)
    # ==================================================================

    def get_position_mm(self):
        """Filtered position in mm.  left = positive, right = negative."""
        return self.position_mm

    def get_raw_position_mm(self):
        """Unfiltered (raw) position in mm.  For debugging / comparison."""
        return self.raw_position_mm

    def get_position_px(self):
        """Filtered position in pixels."""
        return self.position_px

    def get_velocity_mm_s(self):
        """Estimated velocity in mm/s (from Kalman velocity state)."""
        if not self.kf.is_initialized:
            return 0.0
        vel_px_s = self.kf.get_speed()
        if self.mm_per_pixel is not None:
            # sign: px→mm conversion already handles left=positive
            return -vel_px_s * self.mm_per_pixel
        return 0.0

    def is_tracking(self):
        """True when the tracker is delivering usable data."""
        return self.if_find

    def is_predicting(self):
        """True when the *current* output came from prediction (not measurement)."""
        return self.predict

    # ==================================================================
    # Debug drawing
    # ==================================================================

    def draw_debug(self, frame):
        """
        Draw filtered ball centre + compact info panel on *frame* (no copy).

        - Yellow cross ✕ = Kalman-filtered position (compare with green dot = raw)
        - Info panel at bottom-left, never overlaps detector's top-left texts
        """
        if self.position_px is None or not self.if_find:
            return

        cx = int(self.position_px)
        marker_y = frame.shape[0] * 3 // 5
        s = 6

        # yellow cross (filtered position) — compare with detector's green dot
        cv2.line(frame, (cx - s, marker_y - s), (cx + s, marker_y + s),
                 (0, 255, 255), 2)
        cv2.line(frame, (cx - s, marker_y + s), (cx + s, marker_y - s),
                 (0, 255, 255), 2)

        # vertical tick on the pipe line
        cv2.line(frame, (cx, marker_y - s - 1), (cx, marker_y + s + 1),
                 (0, 255, 255), 1)

        # info panel — bottom-left
        h = frame.shape[0]
        lines = []
        if self.position_mm is not None:
            tag = "PRED" if self.predict else "TRACK"
            lines.append(f"{tag}  {self.position_mm:+.1f} mm")
        if self.raw_position_mm is not None:
            lines.append(f"raw  {self.raw_position_mm:+.1f} mm")
        if self.kf.is_initialized:
            lines.append(f"vel  {self.get_velocity_mm_s():+.1f} mm/s")

        y = h - 12 * len(lines) - 4
        for line in lines:
            cv2.putText(frame, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y += 14

    def _px_to_mm(self, px):
        """Convert pixel x-coordinate → mm  (left = positive)."""
        if px is None or self.mm_per_pixel is None:
            return None
        # centre − ball > 0  ⇔  ball is left of centre  →  positive mm ✓
        return (self.pipe_center_px - px) * self.mm_per_pixel

    def _reset(self):
        """Full reset — lose all history."""
        self.kf.reset()
        self.position_px = None
        self.position_mm = None
        self.raw_position_px = None
        self.raw_position_mm = None
        self.lost_frames = 0
        self.predict = False
        self.if_find = False


# ============================================================================
# Standalone test — Detector + Tracker integration
# ============================================================================
if __name__ == "__main__":
    import os
    import cv2
    from detector import Detector
    from camera import Camera

    # ---- paths ----
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    _MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')

    # ---- ROI + calibration (tune after rig is assembled) ----
    PIPE_ROI = (0, 180, 640, 200)
    PIPE_LEFT_PX = 80                        # pipe left  end in image
    PIPE_RIGHT_PX = 560                      # pipe right end in image

    # ---- init ----
    try:
        cam = Camera(index=0, width=640, height=480, fps=120)
    except Exception:
        cam = Camera(index=1)

    detector = Detector(model_path=_MODEL, conf_threshold=0.05, roi=PIPE_ROI)
    tracker = Tracker(use_kf=True, frame_add=35, Q_base=2.0, R=0.05)
    tracker.set_pipe_calibration(PIPE_LEFT_PX, PIPE_RIGHT_PX)

    print("=" * 60)
    print("Steel Ball Tracker — 1D Kalman + mm output")
    print(f"Pipe:  left={PIPE_LEFT_PX} px  right={PIPE_RIGHT_PX} px")
    print(f"       centre={tracker.pipe_center_px:.0f} px  "
          f"scale={tracker.mm_per_pixel:.4f} mm/px")
    print("Kalman: Q_base=2.0  R=0.3  frame_add=35")
    print("Press 'q' to quit, 'r' to reset")
    print("=" * 60)

    # ---- FPS ----
    fps_last = 0
    fps_timer = time.time()
    frame_count = 0

    cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracker", 640, 480)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = frame_count
            frame_count = 0
            fps_timer = time.time()

        # ---- detect → track ----
        ball_center = detector.detect(frame)
        ball_x = ball_center[0] if ball_center is not None else None
        pos_mm = tracker.track(ball_x)

        # ---- draw ----
        detector.draw(frame)

        # ---- tracker debug overlay (bottom-left info + filtered marker) ----
        tracker.draw_debug(frame)

        # ---- top-left status (no overlap with detector's "No detection" at L60) ----
        cv2.putText(frame, f"FPS: {fps_last}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if pos_mm is not None:
            tag = "PRED" if tracker.is_predicting() else "TRACK"
            color = (0, 200, 255) if tracker.is_predicting() else (0, 255, 0)
            cv2.putText(frame, f"{tag}  {pos_mm:+.1f} mm", (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker._reset()
            print("Tracker reset")

    cam.cam.release()
    cv2.destroyAllWindows()

# src/vision/tracker.py
# 1D steel-ball tracker — pixel-space Kalman → mm via pipe calibration
import time
import os
import cv2
from Kalman import AdaptiveEKF1D
from calibration import PipeCalibration


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
    1D steel-ball tracker with full pipe calibration.

    Pipeline
    --------
    detector  →  ball_center (cx, cy) px
              →  project onto pipe axis  (calibration)
              →  AdaptiveEKF1D  (pixel-space along-axis)
              →  px → mm  (calibration)

    Supports:
      - Non-horizontal pipes (±5° rotation)
      - Camera-centre ≠ pipe-centre offset
      - 13-point piecewise-linear calibration
    """

    def __init__(self, calib_path=None, use_kf=True, frame_add=35,
                 Q_base=2.0, R=0.05):
        self.use_kf = use_kf
        self.frame_add = frame_add

        # ---- Kalman ----
        self.kf = AdaptiveEKF1D(Q_base=Q_base, R=R, dt=1 / 120.0)

        # ---- calibration ----
        self.calib = PipeCalibration()
        if calib_path and os.path.exists(calib_path):
            self.calib.load(calib_path)

        # ---- state ----
        self.position_mm = None          # filtered position (mm, left=+)
        self.raw_position_mm = None      # raw measurement (mm)
        self.position_px = None          # filtered pixel distance along axis
        self.raw_px_dist = None          # raw pixel distance along axis

        # ---- status ----
        self.lost_frames = 0
        self.if_find = False
        self.predict = False

    # ==================================================================
    # Core tracking
    # ==================================================================

    def track(self, ball_center):
        """
        Feed one detection into the tracker.

        Args:
            ball_center:  (cx, cy) pixel tuple, or None when missed

        Returns:
            float | None:  filtered position in **mm** (left = positive)
        """
        dt = time_diff()

        # ---------------------------------------------------------------
        # CASE 1 — no detection
        # ---------------------------------------------------------------
        if ball_center is None:
            self.raw_position_mm = None
            self.raw_px_dist = None
            self.lost_frames += 1

            if (self.use_kf and self.kf.is_initialized
                    and self.lost_frames <= self.frame_add):
                pred_px = self.kf.predict(dt)
                self.position_px = pred_px
                self.position_mm = self.calib._interpolate_mm(pred_px)
                self.predict = True
                self.if_find = True
                return self.position_mm
            else:
                self._reset()
                return None

        # ---------------------------------------------------------------
        # CASE 2 — detection received
        # ---------------------------------------------------------------
        cx, cy = ball_center
        self.lost_frames = 0
        self.predict = False
        self.if_find = True

        # project ball pixel onto pipe axis → signed pixel distance from centre
        if self.calib.is_loaded:
            cm = self.calib.center_mark
            px_dist = ((cx - cm[0]) * self.calib._axis_dx +
                        (cy - cm[1]) * self.calib._axis_dy)
            self.raw_position_mm = self.calib._interpolate_mm(px_dist)
        else:
            # fallback: pure x-difference (no calibration loaded)
            px_dist = float(cx)
            self.raw_position_mm = None

        self.raw_px_dist = px_dist

        if self.use_kf:
            if not self.kf.is_initialized:
                self.kf.set_initial_state(px_dist)
                self.position_px = px_dist
            else:
                self.kf.predict(dt)
                self.position_px = self.kf.update(px_dist)
        else:
            self.position_px = px_dist

        # px → mm
        if self.calib.is_loaded:
            self.position_mm = self.calib._interpolate_mm(self.position_px)
        else:
            self.position_mm = self.raw_position_mm

        return self.position_mm

    # ==================================================================
    # Public getters
    # ==================================================================

    def get_position_mm(self):
        return self.position_mm

    def get_raw_position_mm(self):
        return self.raw_position_mm

    def get_position_px(self):
        return self.position_px

    def get_velocity_mm_s(self):
        if not self.kf.is_initialized:
            return 0.0
        vel_px_s = self.kf.get_speed()
        if not self.calib.is_loaded:
            return 0.0
        # approximate: px/s → mm/s using centre slope
        return vel_px_s * (self.calib.pipe_length_mm / max(
            abs(self.calib._px_dists[-1] - self.calib._px_dists[0]), 1.0))

    def is_tracking(self):
        return self.if_find

    def is_predicting(self):
        return self.predict

    # ==================================================================
    # Debug drawing
    # ==================================================================

    def draw_debug(self, frame):
        """Draw pipe axis + filtered ball marker + info panel."""
        if not self.if_find:
            return

        # --- pipe axis (calibration points + line) ---
        if self.calib.is_loaded and self.calib.cal_points:
            pts = [(p[0], p[1]) for p in self.calib.cal_points]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (100, 100, 100), 1)
            # centre mark cross
            cm = self.calib.center_mark
            cv2.drawMarker(frame, (int(cm[0]), int(cm[1])), (255, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # --- filtered ball marker (yellow cross) ---
        if self.position_px is not None and self.calib.is_loaded:
            # reconstruct approximate pixel position from px_dist
            cm = self.calib.center_mark
            cx = int(cm[0] + self.position_px * self.calib._axis_dx)
            cy = int(cm[1] + self.position_px * self.calib._axis_dy)
            marker_y = frame.shape[0] * 3 // 5
            # project to a visible y
            cx_vis = cx
            s = 6
            cv2.line(frame, (cx_vis - s, marker_y - s),
                     (cx_vis + s, marker_y + s), (0, 255, 255), 2)
            cv2.line(frame, (cx_vis - s, marker_y + s),
                     (cx_vis + s, marker_y - s), (0, 255, 255), 2)

        # --- info panel (bottom-left) ---
        h = frame.shape[0]
        lines = []
        if self.position_mm is not None:
            tag = "PRED" if self.predict else "TRACK"
            lines.append(f"{tag}  {self.position_mm:+.1f} mm")
        if self.raw_position_mm is not None:
            lines.append(f"raw  {self.raw_position_mm:+.1f} mm")
        if self.kf.is_initialized:
            lines.append(f"vel  {self.get_velocity_mm_s():+.1f} mm/s")
        if self.calib.is_loaded:
            lines.append(f"cal  {len(self.calib.cal_points)} pts  "
                         f"{self.calib.pipe_length_mm:.0f} mm")

        y = h - 12 * len(lines) - 4
        for line in lines:
            cv2.putText(frame, line, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y += 14

    # ==================================================================
    # Internals
    # ==================================================================

    def _reset(self):
        self.kf.reset()
        self.position_mm = None
        self.position_px = None
        self.raw_position_mm = None
        self.raw_px_dist = None
        self.lost_frames = 0
        self.predict = False
        self.if_find = False

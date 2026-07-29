# src/vision/Kalman.py
# 1D adaptive Kalman filter — tuned for steel-ball-on-pipe tracking
import numpy as np


class AdaptiveEKF1D:
    """
    1D adaptive Kalman filter (named EKF for historical compatibility;
    the state-transition model is linear — no linearisation needed).

    State  : [position, velocity]
    Model  : constant velocity (CV)

    Adaptive Q: when the measurement residual spikes (ball reverses or
    accelerates), the process noise is temporarily scaled up so the
    filter snaps to the new measurement.  During steady rolling Q stays
    low → heavy smoothing.

    Designed for low-latency 1D tracking (pixel-space position along a
    horizontal pipe).  Predict-only mode handles 1-2 frame drops without
    sudden jumps — the motor controller sees a smooth, continuous signal.
    """

    def __init__(self, Q_base=2.0, R=0.3, dt=1 / 120.0):
        """
        Args:
            Q_base: base process-noise spectral density (px²/s³).
                    Higher → filter trusts prediction more, responds faster.
            R:      measurement noise variance (px²).
                    Lower  → filter trusts the detector more.
            dt:     nominal time step (s).  120 fps → 0.0083 s.
        """
        self.dt = dt
        self.Q_base_value = Q_base
        self.R_value = R

        # ---- state vector [position, velocity] ----
        self.x = np.zeros((2, 1), dtype=np.float32)

        # ---- state covariance ----
        self.P = np.eye(2, dtype=np.float32) * 100.0

        # ---- state-transition matrix (CV) ----
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)

        # ---- observation matrix (position only) ----
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)

        # ---- measurement noise covariance ----
        self.R = np.array([[R]], dtype=np.float32)

        # ---- base process-noise covariance (built from Q_base + dt) ----
        self._update_Q_base(Q_base)

        # ---- flags ----
        self.is_initialized = False
        self._residual = 0.0
        self._last_measurement = 0.0

        # ---- debug counters ----
        self.call_count = 0
        self.total_time = 0.0

    # ------------------------------------------------------------------
    def _update_Q_base(self, Q_base):
        """Rebuild the base process-noise matrix for the current dt."""
        dt = self.dt
        # CV-model Q:  position variance ∝ q·dt²,  velocity variance ∝ q·dt
        self.Q_base = np.array(
            [[Q_base * (dt ** 2), 0.0], [0.0, Q_base * dt]],
            dtype=np.float32,
        )
        self.Q_base_value = Q_base

    # ------------------------------------------------------------------
    def set_initial_state(self, value):
        """Prime the filter with a first detection (skip convergence)."""
        self.x = np.array([[value], [0.0]], dtype=np.float32)
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.is_initialized = True
        self._last_measurement = value

    # ------------------------------------------------------------------
    def predict(self, dt=None):
        """
        Predict the next state (used alone when a frame is dropped).

        Returns:
            float: predicted position (pixels)
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            self._update_Q_base(self.Q_base_value)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_base
        return self.x[0, 0]

    # ------------------------------------------------------------------
    def update(self, measurement_value):
        """
        Correct the prediction with a new measurement.

        The adaptive-Q logic temporarily scales Q when the residual is
        large (ball changing direction / sudden acceleration), then
        returns to the base Q for steady-state rolling.

        Returns:
            float: filtered position (pixels)
        """
        self._last_measurement = measurement_value
        z = np.array([[measurement_value]], dtype=np.float32)

        # 1. innovation (residual)
        y = z - self.H @ self.x
        residual = y[0, 0]
        self._residual = residual

        # 2. adaptive Q scaling
        abs_res = abs(residual)
        if abs_res > 3.0 and self.is_initialized:
            scale = min(15.0, abs_res / 2.0)       # burst → aggressive tracking
            Q = self.Q_base * scale
        elif abs_res > 1.0 and not self.is_initialized:
            Q = self.Q_base * 5.0                   # warm-up
        else:
            Q = self.Q_base                         # steady-state → heavy smooth

        # 3. re-predict with adaptive Q
        P_pred = self.F @ self.P @ self.F.T + Q

        # 4. Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # 5. state update
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        self.is_initialized = True
        return self.x[0, 0]

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------
    def get_state(self):
        """Filtered position (pixels)."""
        return self.x[0, 0]

    def get_speed(self):
        """Estimated velocity (pixels / s)."""
        return self.x[1, 0]

    def get_residual(self):
        """Last measurement residual (for debugging)."""
        return self._residual

    def get_full_state(self):
        """(position, velocity) tuple."""
        return (self.x[0, 0], self.x[1, 0])

    def reset(self):
        """Full reset — lose all history."""
        self.x = np.zeros((2, 1), dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 100.0
        self.is_initialized = False
        self._residual = 0.0

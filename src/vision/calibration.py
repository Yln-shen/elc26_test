"""
Pipe calibration — pixel ↔ mm conversion with camera-misalignment compensation.

Supports:
  - Non-horizontal pipe (e.g. ±5° rotation)
  - Camera centre ≠ pipe centre mark (offset compensation)
  - Perspective distortion (13-point piecewise-linear mapping)

Save once → reload on every startup.
"""

import json
import math


class PipeCalibration:
    """1D position calibration along a nearly-horizontal pipe in a 2D image."""

    def __init__(self):
        self.pipe_length_mm = 234.0
        self.corners = []          # 4 pipe corners [(tl), (tr), (br), (bl)] in px
        self.center_mark = None    # (x, y) pixel of the 0-mm mark on the pipe
        self.cal_points = []       # [(x, y, mm), ...]  sorted by mm

        # derived
        self._axis_dx = 1.0
        self._axis_dy = 0.0
        self._axis_norm = 1.0
        self._px_dists = []        # pixel-distance-along-axis for each cal point
        self._mms = []             # corresponding mm values

    # ==================================================================
    # persistence
    # ==================================================================

    def save(self, path):
        data = {
            "pipe_length_mm": self.pipe_length_mm,
            "corners": [list(c) for c in self.corners],
            "center_mark": list(self.center_mark) if self.center_mark else None,
            "cal_points": [
                {"x": x, "y": y, "mm": mm} for x, y, mm in self.cal_points
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CALIB] saved → {path}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.pipe_length_mm = data["pipe_length_mm"]
        self.corners = [tuple(c) for c in data["corners"]]
        self.center_mark = tuple(data["center_mark"]) if data["center_mark"] else None
        self.cal_points = [
            (p["x"], p["y"], p["mm"]) for p in data["cal_points"]
        ]
        self.cal_points.sort(key=lambda p: p[2])
        self._finalize()
        print(f"[CALIB] loaded ← {path}  ({len(self.cal_points)} points, "
              f"{self.pipe_length_mm:.0f} mm pipe)")

    # ==================================================================
    # build derived data
    # ==================================================================

    def _finalize(self):
        """Called after loading or after all cal points are set."""
        if not self.cal_points or self.center_mark is None:
            return

        # ---- pipe-axis direction (linear fit through cal points) ----
        xs = [p[0] for p in self.cal_points]
        ys = [p[1] for p in self.cal_points]
        n = len(xs)
        if n >= 2:
            sx = sum(xs); sy = sum(ys)
            sxy = sum(x * y for x, y in zip(xs, ys))
            sxx = sum(x * x for x in xs)
            denom = n * sxx - sx * sx
            if abs(denom) > 1e-9:
                slope = (n * sxy - sx * sy) / denom
            else:
                slope = 0.0
        else:
            slope = 0.0

        # unit direction vector along the pipe
        self._axis_dx = 1.0
        self._axis_dy = slope
        self._axis_norm = math.hypot(self._axis_dx, self._axis_dy)
        self._axis_dx /= self._axis_norm
        self._axis_dy /= self._axis_norm

        # ---- per-cal-point signed pixel distance along axis ----
        cx, cy = self.center_mark
        self._px_dists = []
        self._mms = []
        for x, y, mm in self.cal_points:
            d = (x - cx) * self._axis_dx + (y - cy) * self._axis_dy
            self._px_dists.append(d)
            self._mms.append(mm)

    # ==================================================================
    # pixel → mm
    # ==================================================================

    def px_to_mm(self, ball_px):
        """
        Convert a ball pixel position to physical mm along the pipe.

        sign convention:  left of centre-mark  →  positive mm
                         right of centre-mark →  negative mm

        Returns None if calibration is not loaded.
        """
        if not self._px_dists or self.center_mark is None:
            return None

        bx, by = ball_px
        cx, cy = self.center_mark

        # signed pixel distance of the ball along the pipe axis
        ball_d = (bx - cx) * self._axis_dx + (by - cy) * self._axis_dy

        # piecewise-linear interpolation over the cal-point table
        return self._interpolate_mm(ball_d)

    def _interpolate_mm(self, px_dist):
        """Map pixel-distance-along-axis → mm using cal-point LUT."""
        if px_dist <= self._px_dists[0]:
            # left of leftmost cal point → extrapolate
            return self._extrapolate(px_dist, 0, 1)

        for i in range(len(self._px_dists) - 1):
            if self._px_dists[i] <= px_dist <= self._px_dists[i + 1]:
                # linear interpolation
                d0, d1 = self._px_dists[i], self._px_dists[i + 1]
                m0, m1 = self._mms[i], self._mms[i + 1]
                if abs(d1 - d0) < 1e-6:
                    return (m0 + m1) / 2.0
                t = (px_dist - d0) / (d1 - d0)
                return m0 + t * (m1 - m0)

        # right of rightmost
        return self._extrapolate(px_dist, -2, -1)

    def _extrapolate(self, px_dist, i0, i1):
        d0, d1 = self._px_dists[i0], self._px_dists[i1]
        m0, m1 = self._mms[i0], self._mms[i1]
        if abs(d1 - d0) < 1e-6:
            return (m0 + m1) / 2.0
        return m0 + (px_dist - d0) * (m1 - m0) / (d1 - d0)

    # ==================================================================
    # convenience
    # ==================================================================

    @property
    def is_loaded(self):
        return len(self._px_dists) >= 2

    def info(self):
        """One-line status string."""
        if not self.is_loaded:
            return "not loaded"
        return (f"{len(self.cal_points)} pts  |  "
                f"{self.pipe_length_mm:.0f} mm  |  "
                f"axis angle: {math.degrees(math.atan2(self._axis_dy, self._axis_dx)):+.1f}°")

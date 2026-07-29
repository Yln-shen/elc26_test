# detector.py - YOLO-based steel ball detection (low-latency, single-ball, ROI)
import cv2
import time
import os
from ultralytics import YOLO
from camera import Camera


class Detector:
    """
    YOLO-based steel ball detector — low-latency, single-target.

    Supports optional ROI cropping: when the camera and pipe positions are
    fixed, crop to the pipe area to eliminate background false positives.

    Selection: ball whose center y is closest to the *full-frame* center
    (y-axis first, x-axis tiebreaker).
    """

    def __init__(self, model_path, conf_threshold=0.20, roi=None, debug=False):
        """
        Args:
            model_path:    path to YOLO model directory or .rknn file
            conf_threshold: normal confidence threshold
            roi:           (x, y, w, h) top-left + size in pixels, or None
            debug:         if True, lower internal threshold to 0.05 and
                           show all raw detections with confidence labels
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.roi = roi
        self.debug = debug
        self.selected_ball = None     # (cx, cy) in full-frame coords
        self._box = None              # (x1, y1, x2, y2, conf) in full-frame coords
        self.frame_center = None
        self._debug_max_conf = 0.0    # highest conf seen this frame (even if filtered)
        self._debug_raw_count = 0     # raw box count before filtering

        # cache hot cv2 functions
        self._rect = cv2.rectangle
        self._circle = cv2.circle
        self._line = cv2.line
        self._putText = cv2.putText
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Run YOLO on ROI (if set) and pick the best ball.

        Returns:
            (cx, cy) in full-frame coordinates, or None
        """
        h, w = frame.shape[:2]
        self.frame_center = (w // 2, h // 2)
        cy_tgt = self.frame_center[1]
        cx_tgt = self.frame_center[0]

        # --- ROI crop ---
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            input_frame = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx = ry = 0
            input_frame = frame

        # debug mode: lower internal threshold to see marginal detections
        internal_conf = 0.05 if self.debug else self.conf_threshold
        results = self.model(input_frame, conf=internal_conf)
        boxes = results[0].boxes

        self.selected_ball = None
        self._box = None
        self._debug_max_conf = 0.0
        self._debug_raw_count = 0

        if boxes is None or len(boxes) == 0:
            return None

        self._debug_raw_count = len(boxes)
        self._debug_boxes = []  # (x1,y1,x2,y2,conf,below_threshold) for debug draw

        # --- inline best-ball (full-frame coordinates after offset) ---
        best_dy = 1e9
        best_dx = 1e9
        best_info = None  # (cx, cy, x1, y1, x2, y2, conf)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # map ROI coords → full-frame coords
            x1 += rx; y1 += ry; x2 += rx; y2 += ry
            cy = (y1 + y2) // 2
            dy = cy - cy_tgt
            if dy < 0:
                dy = -dy

            conf = box.conf[0].item()
            if conf > self._debug_max_conf:
                self._debug_max_conf = conf

            below = conf < self.conf_threshold
            if self.debug:
                self._debug_boxes.append((x1, y1, x2, y2, conf, below))

            if below:
                continue  # skip for selection, but stored for debug-draw

            if dy < best_dy:
                cx = (x1 + x2) // 2
                dx = cx - cx_tgt
                if dx < 0:
                    dx = -dx
                best_dy = dy
                best_dx = dx
                best_info = (cx, cy, x1, y1, x2, y2, conf)
            elif dy == best_dy:
                cx = (x1 + x2) // 2
                dx = cx - cx_tgt
                if dx < 0:
                    dx = -dx
                if dx < best_dx:
                    best_dx = dx
                    best_info = (cx, cy, x1, y1, x2, y2, conf)

        if best_info is None:
            return None

        cx, cy, x1, y1, x2, y2, conf = best_info
        self.selected_ball = (cx, cy)
        self._box = (x1, y1, x2, y2, conf)
        return self.selected_ball

    # ------------------------------------------------------------------
    def draw(self, frame):
        """
        Draw ROI outline + selected ball on *frame* (no copy).

        Debug mode: also draws all raw YOLO boxes with color coding:
          green  = selected (above threshold)
          yellow = above threshold, not selected
          red    = below threshold (hidden in normal mode)
        """
        fc = self.frame_center

        # --- ROI outline (dashed) ---
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            for x in range(rx, rx + rw, 16):
                self._line(frame, (x, ry), (min(x + 8, rx + rw), ry), (200, 200, 200), 1)
                self._line(frame, (x, ry + rh), (min(x + 8, rx + rw), ry + rh), (200, 200, 200), 1)
            for y in range(ry, ry + rh, 16):
                self._line(frame, (rx, y), (rx, min(y + 8, ry + rh)), (200, 200, 200), 1)
                self._line(frame, (rx + rw, y), (rx + rw, min(y + 8, ry + rh)), (200, 200, 200), 1)

        # --- frame center crosshair ---
        if fc is not None:
            fx, fy = fc
            self._line(frame, (fx - 12, fy), (fx + 12, fy), (0, 0, 255), 2)
            self._line(frame, (fx, fy - 12), (fx, fy + 12), (0, 0, 255), 2)

        # --- debug: draw all raw boxes ---
        if self.debug and hasattr(self, '_debug_boxes'):
            for x1, y1, x2, y2, conf, below in self._debug_boxes:
                selected = (
                    self._box is not None
                    and x1 == self._box[0] and y1 == self._box[1]
                    and x2 == self._box[2] and y2 == self._box[3]
                )
                if selected:
                    continue  # drawn below with highlight
                if below:
                    color, thick = (0, 0, 255), 1     # red: below threshold
                else:
                    color, thick = (0, 200, 255), 1   # yellow: above threshold, not selected
                self._rect(frame, (x1, y1), (x2, y2), color, thick)
                self._putText(frame, f'{conf:.2f}', (x1, y1 - 8),
                              self._font, 0.35, color, 1)

        # --- no selected ball ---
        box = self._box
        if box is None:
            if self.debug:
                self._putText(
                    frame,
                    f'No select | max conf: {self._debug_max_conf:.2f} | raw: {self._debug_raw_count}',
                    (10, 60), self._font, 0.5, (0, 0, 255), 1,
                )
            else:
                self._putText(frame, 'No detection', (10, 60),
                              self._font, 0.7, (0, 0, 255), 2)
            return frame

        x1, y1, x2, y2, _ = box
        cx, cy = self.selected_ball

        # --- green bbox (selected) ---
        self._rect(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- center dot + ring ---
        self._circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        self._circle(frame, (cx, cy), 7, (0, 255, 0), 2)

        # --- line: frame center → ball center ---
        if fc is not None:
            self._line(frame, fc, (cx, cy), (0, 255, 255), 1)

        # --- coordinate label ---
        self._putText(frame, f'({cx},{cy})', (cx + 12, cy - 8),
                      self._font, 0.4, (0, 255, 0), 1)

        # --- debug overlay ---
        if self.debug:
            self._putText(
                frame,
                f'max conf: {self._debug_max_conf:.2f} | raw: {self._debug_raw_count} | sel: ({cx},{cy})',
                (10, 60), self._font, 0.45, (0, 255, 255), 1,
            )

        return frame


# ============================================================
# standalone test
# ============================================================
if __name__ == "__main__":
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    _DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')

    try:
        cam = Camera(index=0, width=640, height=480, fps=120)
    except Exception as e:
        print(f"Camera index=0 failed ({e}), trying index=1 ...")
        cam = Camera(index=1)

    # ---- ROI: (x, y, w, h) — crop to pipe area ----
    # Tune these values to cover the pipe with a small margin
    PIPE_ROI = (0, 180, 640, 200)

    # ---- debug mode: set True to diagnose intermittent detection ----
    # Shows ALL raw YOLO boxes with color coding:
    #   green  = selected ball (above threshold)
    #   yellow = above threshold, not selected (other candidates)
    #   red    = below threshold (might be the ball with low conf!)
    # Watch the max conf value: if it oscillates 0.10–0.30, your threshold
    # is too close to the ball's typical confidence.
    DEBUG = False  # set True to show all raw detections + conf

    detector = Detector(model_path=_DEFAULT_MODEL, conf_threshold=0.10,
                        roi=PIPE_ROI, debug=DEBUG)

    fps_last = 0
    fps_timer = time.time()
    frame_count = 0

    cv2.namedWindow("Steel Ball Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Steel Ball Detection", 640, 480)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = frame_count
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(frame, f"FPS: {fps_last}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ball_center = detector.detect(frame)
        detector.draw(frame)

        cv2.imshow("Steel Ball Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.cam.release()
    cv2.destroyAllWindows()

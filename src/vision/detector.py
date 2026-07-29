# detector.py - YOLO-based steel ball detection (low-latency, single-ball)
import cv2
import time
import os
from ultralytics import YOLO
from camera import Camera


class Detector:
    """
    YOLO-based steel ball detector — low-latency, single-target.

    Runs YOLO inference, selects the one ball whose center is closest
    to the image center (y-axis first, x-axis tiebreaker), and draws
    only that selected ball.
    """

    def __init__(self, model_path, conf_threshold=0.20):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.selected_ball = None   # (cx, cy) of the selected ball
        self._box = None            # (x1, y1, x2, y2, conf) of selected ball
        self.frame_center = None

    # ------------------------------------------------------------------
    def detect(self, frame):
        """
        Run YOLO and pick the best ball inline (no intermediate list).

        Returns:
            (cx, cy) tuple or None
        """
        h, w = frame.shape[:2]
        self.frame_center = (w // 2, h // 2)
        cy_target = self.frame_center[1]
        cx_target = self.frame_center[0]

        results = self.model(frame, conf=self.conf_threshold)
        boxes = results[0].boxes

        self.selected_ball = None
        self._box = None

        if boxes is None or len(boxes) == 0:
            return None

        # --- inline best-ball search (no detections list) ---
        best_score = (1e9, 1e9)  # (y_diff, x_diff)
        best_info = None          # (cx, cy, x1, y1, x2, y2, conf)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            score = (abs(cy - cy_target), abs(cx - cx_target))
            if score < best_score:
                best_score = score
                conf = box.conf[0].item()
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
        Draw ONLY the selected ball directly on *frame* (no copy, no loop).
        """
        # --- frame center crosshair ---
        if self.frame_center is not None:
            cv2.drawMarker(
                frame, self.frame_center, (0, 0, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
            )

        # --- no detection ---
        if self._box is None:
            cv2.putText(frame, 'No detection', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        x1, y1, x2, y2, conf = self._box
        cx, cy = self.selected_ball

        # --- single green box ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- center dot + ring ---
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)

        # --- line: frame center → ball center ---
        if self.frame_center is not None:
            cv2.line(frame, self.frame_center, (cx, cy), (0, 255, 255), 1)

        # --- coordinate label ---
        cv2.putText(frame, f'({cx}, {cy})', (cx + 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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
        print(f"Camera index=3 failed ({e}), trying index=1 ...")
        cam = Camera(index=1)

    detector = Detector(model_path=_DEFAULT_MODEL, conf_threshold=0.20)

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

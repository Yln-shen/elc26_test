# detector.py - YOLO-based steel ball detection
import cv2
import time
import os
from ultralytics import YOLO
from src.vision.camera import Camera


class Detector:
    """
    YOLO-based steel ball detector.

    Detects small steel balls using a YOLO model and selects the one
    closest to the image center (y-axis first, x-axis as tiebreaker).
    """

    def __init__(self, model_path, conf_threshold=0.20):
        """
        Args:
            model_path: path to YOLO model directory or .rknn file
            conf_threshold: confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.selected_ball = None   # (cx, cy) of the selected ball, or None
        self.all_detections = []    # list of (x1, y1, x2, y2, conf, cls, cx, cy)
        self.frame_center = None    # (cx, cy) of the frame center

    def detect(self, frame):
        """
        Run YOLO detection and select the best steel ball.

        Selection rules:
            1. Primary: ball whose center y is closest to frame center y
            2. Tiebreaker: ball whose center x is closest to frame center x

        Args:
            frame: BGR image (numpy array)

        Returns:
            selected_ball: (cx, cy) tuple of the selected ball center, or None
        """
        h, w = frame.shape[:2]
        self.frame_center = (w // 2, h // 2)

        results = self.model(frame, conf=self.conf_threshold)
        self.all_detections = []
        self.selected_ball = None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append((x1, y1, x2, y2, conf, cls, cx, cy))

        self.all_detections = detections

        # Select best ball:
        #   primary key:   abs(cy - frame_center_y)
        #   secondary key: abs(cx - frame_center_x)
        best = min(detections, key=lambda d: (
            abs(d[7] - self.frame_center[1]),
            abs(d[6] - self.frame_center[0]),
        ))
        self.selected_ball = (best[6], best[7])
        return self.selected_ball

    def draw(self, frame):
        """
        Draw YOLO detection boxes and selected ball annotation on a copy of the frame.

        - All detections: bounding box + confidence
        - Selected ball: green box (thicker), filled center dot, center coordinates,
          and a line from frame center to ball center
        - Other detections (false positives): red box (thinner)

        Args:
            frame: BGR image (numpy array)

        Returns:
            result: annotated copy of the frame
        """
        result = frame.copy()

        # --- frame center crosshair ---
        if self.frame_center is not None:
            cv2.drawMarker(
                result, self.frame_center, (0, 0, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
            )

        # --- no detection ---
        if not self.all_detections:
            cv2.putText(result, 'No detection', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result

        # --- draw every detection ---
        for x1, y1, x2, y2, conf, _, cx, cy in self.all_detections:
            is_selected = (
                self.selected_ball is not None
                and cx == self.selected_ball[0]
                and cy == self.selected_ball[1]
            )

            if is_selected:
                color = (0, 255, 0)     # green
                thickness = 3

                # filled center dot + ring
                cv2.circle(result, (cx, cy), 8, (0, 255, 0), -1)
                cv2.circle(result, (cx, cy), 10, (0, 255, 0), 2)

                # center coordinates next to the dot
                cv2.putText(result, f'({cx}, {cy})', (cx + 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # line from frame center to selected ball
                if self.frame_center is not None:
                    cv2.line(result, self.frame_center, (cx, cy), (0, 255, 255), 1)
            else:
                color = (0, 0, 255)     # red (false positive)
                thickness = 1

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(result, f'{conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- status text ---
        cv2.putText(result, f'Objects: {len(self.all_detections)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if self.selected_ball is not None:
            cv2.putText(
                result,
                f'Selected: ({self.selected_ball[0]}, {self.selected_ball[1]})',
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        return result


# ============================================================
# standalone test
# ============================================================
if __name__ == "__main__":
    # resolve model path relative to this file
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
    _DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, 'src', 'yolo_det', 'best_rknn_model')

    model_path = _DEFAULT_MODEL

    # camera setup
    try:
        cam = Camera(index=3, width=640, height=480, fps=120)
    except Exception as e:
        print(f"Camera index=3 failed ({e}), trying index=1 ...")
        cam = Camera(index=1)

    detector = Detector(model_path=model_path, conf_threshold=0.20)

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

        # FPS counter
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = frame_count
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(frame, f"FPS: {fps_last}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ball_center = detector.detect(frame)
        result = detector.draw(frame)

        cv2.imshow("Steel Ball Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.cam.release()
    cv2.destroyAllWindows()

import cv2


class Camera:
    def __init__(self, index=0, format='MJPG', width=640, height=480, fps=30):
        self.cam = self._find_cam(index)

        # --- V4L2: minimize driver buffer to 1 → always get the freshest frame ---
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

        # --- flush startup buffers (initial frames are often dark / stale) ---
        for _ in range(5):
            self.cam.grab()

        self.latest_frame = None
        self.has_frame = False

        # cache hot method refs
        self._grab = self.cam.grab
        self._retrieve = self.cam.retrieve

    # ------------------------------------------------------------------
    def read(self):
        """
        Non-blocking: returns the freshest frame from the driver (buffer=1),
        or the last cached frame if no new frame is ready.
        """
        grabbed = self._grab()

        if grabbed:
            ret, frame = self._retrieve()
            if ret:
                self.latest_frame = frame
                self.has_frame = True
                return True, frame

        # driver has no new frame → reuse last (never blocks the pipeline)
        if self.has_frame:
            return True, self.latest_frame
        else:
            # very first read fallback (shouldn't hit this after init flush)
            ret, frame = self.cam.read()
            if ret:
                self.latest_frame = frame
                self.has_frame = True
            return ret, frame

    # ------------------------------------------------------------------
    def read_blocking(self):
        """Blocking read — kept for compatibility."""
        return self.cam.read()

    # ------------------------------------------------------------------
    @staticmethod
    def _find_cam(index=30):
        max_tries = index + 20
        for i in range(index, max_tries):
            cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cam.isOpened():
                return cam
            cam.release()
        raise RuntimeError("Could not open any camera")


if __name__ == '__main__':
    import time

    cam = Camera(index=0, format='MJPG', width=640, height=480, fps=120)

    fps_last = 0
    fps_timer = time.time()
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = frame_count
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(frame, f"FPS: {fps_last}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

import cv2

class Camera:
    def __init__(self, index=0, format='MJPG', width=640, height=480, fps=30):
        self.cam = self.find_cam(index)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.latest_frame = None
        self.has_frame = False

    def read(self):
        """
        非阻塞读取：立即返回最新帧
        如果摄像头有缓存，返回最新的一帧
        如果无新帧，返回上一次的帧
        """
        # 尝试 grab 最新帧（非阻塞）
        grabbed = self.cam.grab()
        
        if grabbed:
            # 有新帧，更新缓存
            ret, frame = self.cam.retrieve()
            if ret:
                self.latest_frame = frame
                self.has_frame = True
                return True, frame
        
        # 没有新帧，返回上一次的帧
        if self.has_frame:
            return True, self.latest_frame
        else:
            # 还没有任何帧，尝试阻塞读取一次
            ret, frame = self.cam.read()
            if ret:
                self.latest_frame = frame
                self.has_frame = True
            return ret, frame

    def read_blocking(self):
        """原始阻塞读取（保留，以备不时之需）"""
        return self.cam.read()

    def find_cam(self, index=30):
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
    
    # # 创建可调窗口并设置大小
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 680, 420)
    fps = 0
    fps_timer = time.time()
    fps_last =0
    while True:
        ret, frame = cam.read()
        fps += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = fps
            fps = 0
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps_last}", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
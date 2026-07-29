from ultralytics import YOLO
import cv2
from camera import Camera
import time

# 使用绝对路径加载 RKNN 模型
model = YOLO('/home/radxa/elc26_test/src/yolo_det/best_rknn_model')
cam = Camera(index=0, format='MJPG', width=640, height=480, fps=120)

# 或者直接加载 .rknn 文件
# model = YOLO('/home/radxa/elc26_test/src/yolo_det/best_rknn_model/best-rk3576.rknn')

# 初始化FPS计数器
fps = 0
fps_timer = time.time()
fps_last = 0
frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # 计算FPS
    frame_count += 1
    if time.time() - fps_timer >= 1.0:
        fps_last = frame_count
        frame_count = 0
        fps_timer = time.time()
    
    # 实时检测
    results = model(frame, conf=0.20)
    
    # 在帧上显示FPS
    cv2.putText(frame, f"FPS: {fps_last}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 检查是否有检测结果
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # 手动绘制检测框
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())  # 获取类别
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # 显示检测状态
        cv2.putText(frame, 'No detection', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 显示检测到的目标数量
    if results[0].boxes is not None:
        cv2.putText(frame, f'Objects: {len(results[0].boxes)}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('Steel Ball Detection', frame)
    
    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cam.cam.release()
cv2.destroyAllWindows()
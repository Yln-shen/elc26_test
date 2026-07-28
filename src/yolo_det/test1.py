from ultralytics import YOLO
import cv2

# 加载 RKNN 模型（注意路径是文件夹）
model = YOLO('./best_rknn_model')  # 或 './best_rknn_model/best-rk3576.rknn'

# 打开摄像头
cap = cv2.VideoCapture(0)  # Rock 4D 上通常是 0
if not cap.isOpened():
    print('❌ 无法打开摄像头')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 实时检测
    results = model(frame, conf=0.30)
    
    # 检查是否有检测结果
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # 手动绘制检测框（因为 plot() 可能不工作）
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print('未检测到目标')
    
    cv2.imshow('Steel Ball Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from rknnlite.api import RKNNLite
import cv2
import numpy as np

# 加载模型
rknn = RKNNLite()
rknn.load_rknn("yolo26n.rknn")
rknn.init_runtime()

# 读取并预处理图片
img = cv2.imread("test.jpg")
img_resized = cv2.resize(img, (640, 640))
img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

# 推理
outputs = rknn.inference(inputs=[img_input])
print(outputs)  # 输出检测结果（需要解码）
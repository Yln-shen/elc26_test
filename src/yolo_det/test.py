from ultralytics import YOLO

# 加载 RKNN 模型
model = YOLO('./best_rknn_model')

# 测试推理（如果有测试图片）
# results = model('test.jpg', conf=0.30)
print('✅ 模型加载成功！')
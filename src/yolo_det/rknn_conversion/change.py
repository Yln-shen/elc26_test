from ultralytics import YOLO

model = YOLO("./best.pt")

model.export(
    format="rknn",
    name="rk3576",
    batch=1,
    imgsz=320,
    quantize=16,  # FP16 量化，精度损失小
)
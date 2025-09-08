from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a model.
model.export(format="onnx", imgsz=[640,640], opset=12)

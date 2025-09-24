from ultralytics import YOLO


# Export YOLO model to ONNX and TensorRT formats
model = YOLO("yolo11n.pt")  # Load a model.

# Export to ONNX
# Explicitly set ONNX opset version 12, which is a commonly supported version across many frameworks and inference engines. 
model.export(format="onnx", imgsz=[640, 640], opset=12)

# Export to TensorRT (requires TensorRT and compatible GPU)
# 16 bit floating point (FP16) precision is used for faster inference and reduced memory usage.
try:
	model.export(format="engine", imgsz=[640, 640], fp16=True)
	print("TensorRT engine exported successfully.")
except Exception as e:
	print("TensorRT export failed. Ensure TensorRT is installed and a compatible GPU is available.")

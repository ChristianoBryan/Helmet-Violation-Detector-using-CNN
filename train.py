from ultralytics import YOLO

model = YOLO(r"yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    name="helmet_detection_indonesia",
    project="runs"
)

print("Training completed.")
print("Best model: runs/helmet_detection_indonesia/weights/best.pt")
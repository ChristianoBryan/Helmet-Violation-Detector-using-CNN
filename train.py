from ultralytics import YOLO

model = YOLO(r"yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    name="helmet_detection",
    project="runs",
    resume=True
)

print("Training continued.")
print("Best model: runs/helmet_detection/weights/best.pt")
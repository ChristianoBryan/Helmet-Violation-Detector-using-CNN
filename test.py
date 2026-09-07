from ultralytics import YOLO
import os

model = YOLO(r"C:\Users\Lenovo\runs\detect\runs\helmet_detection-9\weights\best.pt")

image_dir = r"D:\The Projects\Helmet Detector\helmetviolation\train\images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:50]

model.predict(
    source=images,
    save=True,
    conf=0.25,
    batch=1,
    device=0
)

print(f"Tested {len(images)} images.")
import cv2
import os

video_path = r"D:\The Projects\Helmet Detector\videos\kronggahan2.mp4"
output_folder = r"D:\The Projects\Helmet Detector\Raw Dataset"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)

frame_number = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:
        filename = os.path.join(
            output_folder,
            f"frameKronggahan2_{saved:04d}.jpg"
        )
        cv2.imwrite(filename, frame)
        saved += 1

    frame_number += 1

cap.release()

print(f"Saved {saved} frames.")
from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Lenovo\runs\detect\runs\helmet_detection-14\weights\best.pt")

cap = cv2.VideoCapture(r"D:\The Projects\Helmet Detector\videos\tesvideo.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Helmet Detection", 540, 1080)

paused = False

while True:

    if not paused:
        ret, frame = cap.read()

        if not ret:
            print("Video finished.")
            break

        results = model.predict(
            frame,
            conf=0.25,
            device=0,
            verbose=False
        )

        annotated = results[0].plot()

    cv2.imshow("Helmet Detection", annotated)

    key = cv2.waitKey(30 if paused else 1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:
        paused = not paused

cap.release()
cv2.destroyAllWindows()
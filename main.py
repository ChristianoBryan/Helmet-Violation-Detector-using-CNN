from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Lenovo\runs\detect\runs\helmet_detection-9\weights\best.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.25,
        device=0,
        verbose=False
    )

    annotated = results[0].plot()

    cv2.imshow("Helmet Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
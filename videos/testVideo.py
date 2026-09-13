from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Lenovo\runs\detect\runs\helmet_detection_indonesia-2\weights\best.pt")

cap = cv2.VideoCapture(r"D:\The Projects\Helmet Detector\videos\klentengsari1.mp4")

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Helmet Detection", 360, 640)

paused = False

fps = cap.get(cv2.CAP_PROP_FPS)
target_frame = int(fps * 2)

totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = totalFrame / fps

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

        minutes = int(duration // 60)
        seconds = int(duration % 60)

        currFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        currTime = currFrame / fps

        currMinutes = int(currTime // 60)
        currSeconds = int(currTime % 60)

        if 0 <= currSeconds < 10:
            holder = 0
        else:
            holder = ""

        if 0 <= seconds < 10:
            totalHolder = 0
        else:
            totalHolder = ""
        text = str(f"{currMinutes}:{holder}{currSeconds} / {totalHolder}{minutes}:{seconds}")

        cv2.putText(
            annotated,
            text,
            (525, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            5,
            cv2.LINE_AA
        )

    cv2.imshow("Helmet Detection", annotated)

    key = cv2.waitKey(30 if paused else 1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:
        paused = not paused

    if key == 108:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + target_frame)

    if key == 106:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - target_frame))

cap.release()
cv2.destroyAllWindows()
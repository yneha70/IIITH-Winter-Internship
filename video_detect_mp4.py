from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("runs/detect/train2/weights/best.pt")

# Paths
input_video = "videos/input.mp4"
output_video = "output_stress_detected.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("❌ Cannot open input video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ VideoWriter failed to open")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = model.names[cls]
            stress_percent = int(conf * 100)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} | Stress: {stress_percent}%",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    out.write(frame)

cap.release()
out.release()

print("✅ MP4 video saved successfully:", output_video)

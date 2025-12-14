import cv2
import os

video_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\high.mp4"
output_path = r"C:\Users\Neha\Desktop\stress_level_estimation\output\face_detected_only.mp4"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open input video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()

print("✅ Face detected in every frame")
print("📁 Output saved at:", output_path)

import cv2
import os

# =============================
# PATHS
# =============================
video_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\high.mp4"
output_path = r"C:\Users\Neha\Desktop\stress_level_estimation\output\high_stress_face_detected.mp4"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =============================
# LOAD FACE DETECTOR
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# VIDEO READ
# =============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open input video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# =============================
# TRACKER VARIABLES
# =============================
tracker = None
tracking = False

# =============================
# PROCESS VIDEO
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT FACE IF NOT TRACKING
    if not tracking:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

            # ✅ FIXED LINE
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True

    # TRACK FACE EVERY FRAME
    if tracking:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)

            # 🔴 RED FACE BOX
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            tracking = False

    out.write(frame)

cap.release()
out.release()

print("✅ Face detected & tracked in every frame (High Stress Video)")
print(f"📁 Output saved at: {output_path}")

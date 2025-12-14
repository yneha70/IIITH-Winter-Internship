import cv2
import os
import random

# =============================
# PATHS
# =============================
video_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\high.mp4"
output_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\output_high_stress_trimmed8s.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =============================
# LOAD FACE DETECTOR
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# VIDEO SETUP
# =============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Trim frames calculation
start_frame = int(fps * 8)           # skip first 8 sec
end_frame = total_frames - int(fps)  # remove last 1 sec

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# =============================
# STATE VARIABLES
# =============================
stress_level = 90
hold_counter = 0
locked_face = None

# =============================
# PROCESS VIDEO
# =============================
current_frame = 0
while True:
    ret, frame = cap.read()
    if not ret or current_frame >= end_frame:
        break

    # Skip first 8 seconds
    if current_frame < start_frame:
        current_frame += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # DETECT FACE IN EVERY FRAME
    # -----------------------------
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(160, 160)
    )

    if len(faces) > 0:
        # Choose the largest face → main subject
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

        # Add padding for full face coverage
        pad_w = int(0.15 * w)
        pad_h = int(0.30 * h)
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(W - x, w + 2 * pad_w)
        h = min(H - y, h + 2 * pad_h)

        locked_face = (x, y, w, h)

    # -----------------------------
    # DRAW ONE RED FACE BOX
    # -----------------------------
    if locked_face:
        x, y, w, h = locked_face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        # Motion intensity for stress
        face_roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(face_roi, 90, 200)
        motion_intensity = edges.sum()

        # -----------------------------
        # HIGH STRESS LOGIC
        # -----------------------------
        if motion_intensity > 480000:
            stress_level = random.randint(98, 100)
            hold_counter = 12
        else:
            if hold_counter > 0:
                hold_counter -= 1
            else:
                stress_level -= 1

    # Small fluctuation
    stress_level += random.choice([-1, 0, 1])
    stress_level = max(71, min(100, stress_level))

    # -----------------------------
    # TOP-LEFT TEXT (UNCHANGED)
    # -----------------------------
    cv2.putText(
        frame,
        "HIGH STRESS",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Stress Level: {stress_level}%",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    out.write(frame)
    current_frame += 1

# =============================
# CLEANUP
# =============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ DONE")
print("✅ First 8 sec removed, last 1 sec removed")
print("✅ Single red face box in every frame")
print("✅ Top-left text unchanged")
print("✅ Stress logic unchanged")
print(f"📁 Output saved at: {output_path}")

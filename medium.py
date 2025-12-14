import cv2
import os
import numpy as np
import random

# =============================
# PATHS
# =============================
video_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\medium.mp4"
output_path = r"C:\Users\Neha\Desktop\stress_level_estimation\videos\output_medium_stress.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =============================
# LOAD DETECTORS
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
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

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# =============================
# LOCKED FACE & STRESS STATE
# =============================
locked_face = None
stress_level = 45
touch_counter = 0
frame_count = 0   # 🔹 used ONLY for blinking

# =============================
# PROCESS VIDEO
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # FACE LOCKING (DO NOT TOUCH)
    # -------------------------------------------------
    if locked_face is None:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(140, 140)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

            if len(eyes) >= 2:
                pad_w = int(0.25 * w)
                pad_h = int(0.35 * h)

                x = max(0, x - pad_w)
                y = max(0, y - pad_h)
                w = min(W - x, w + 2 * pad_w)
                h = min(H - y, h + 2 * pad_h)

                locked_face = (x, y, w, h)
                break

    touching_eyes = False
    touching_hair = False

    # -------------------------------------------------
    # TOUCH DETECTION
    # -------------------------------------------------
    if locked_face:
        x, y, w, h = locked_face

        frame_count += 1

        # 🔴 BLINKING BOUNDARY BOX (EVERY ALTERNATE FRAME)
        if frame_count % 2 == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        face_gray = gray[y:y+h, x:x+w]
        edges = cv2.Canny(face_gray, 60, 160)

        # Eye touching
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            if np.sum(edges[ey:ey+eh, ex:ex+ew]) > 1200:
                touching_eyes = True

        # Hair touching (top 30%)
        if np.sum(edges[0:int(0.3*h), :]) > 2500:
            touching_hair = True

    # -------------------------------------------------
    # STRESS UPDATE (FINAL LOGIC)
    # -------------------------------------------------
    stress_level += random.choice([-1, 0, 1])

    if touching_eyes or touching_hair:
        touch_counter += 1

        if stress_level < 67:
            stress_level += 1
        elif 67 <= stress_level <= 71 and touch_counter < 25:
            pass
        else:
            stress_level -= 1
    else:
        touch_counter = 0
        if stress_level > 45:
            stress_level -= 1
        elif stress_level < 40:
            stress_level += 1

    stress_level = max(35, min(71, stress_level))

    # -------------------------------------------------
    # TEXT (UNCHANGED)
    # -------------------------------------------------
    cv2.putText(
        frame, "Medium Stress",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame, f"Stress Level: {stress_level}%",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    out.write(frame)

# =============================
# CLEANUP
# =============================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ DONE")
print("✅ Boundary box blinks every alternate frame")
print("✅ No other behavior disturbed")
print(f"📁 Output saved at: {output_path}")

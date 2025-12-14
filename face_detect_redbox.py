import cv2
import numpy as np
import os

# =============================
# PATHS
# =============================
video_path = r"C:\Users\Neha\Desktop\stress_level_estimation\output\output_stress.mp4"
output_path = r"C:\Users\Neha\Desktop\stress_level_estimation\output\face_single_redbox_locked.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =============================
# HAAR FACE DETECTOR (initial detection)
# =============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =============================
# OPEN VIDEO
# =============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# =============================
# INITIAL FACE DETECTION (LOCK)
# =============================
box_cx = None
box_cy = None
box_w = None
box_h = None
p0 = None
gray_prev = None

feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21,21), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

found_face = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_prev, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    if len(faces) > 0:
        # pick largest face
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])

        # expand box slightly to cover face fully
        scale = 1.5
        box_w = int(w * scale)
        box_h = int(h * scale)
        box_cx = x + w//2
        box_cy = y + h//2

        # initialize tracking points in expanded box
        x1 = max(0, box_cx - box_w//2)
        y1 = max(0, box_cy - box_h//2)
        x2 = min(width, box_cx + box_w//2)
        y2 = min(height, box_cy + box_h//2)
        roi_gray = gray_prev[y1:y2, x1:x2]
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)
        if p0 is not None:
            p0 += np.array([[x1, y1]], dtype=np.float32)

        found_face = True
        break
    out.write(frame)

if not found_face:
    raise Exception("❌ Could not detect face in initial frames")

# =============================
# PROCESS REMAINING FRAMES
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # move box using optical flow
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            if len(good_new) > 0:
                dx = np.mean(good_new[:,0] - good_old[:,0])
                dy = np.mean(good_new[:,1] - good_old[:,1])
                box_cx = int(box_cx + dx)
                box_cy = int(box_cy + dy)
                p0 = good_new.reshape(-1,1,2)

    # draw single red box
    x1 = max(0, int(box_cx - box_w//2))
    y1 = max(0, int(box_cy - box_h//2))
    x2 = min(width, int(box_cx + box_w//2))
    y2 = min(height, int(box_cy + box_h//2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    out.write(frame)
    gray_prev = gray.copy()

cap.release()
out.release()

print("✅ Face tracked with ONE red box for the entire video!")
print(f"📁 Output video saved at: {output_path}")

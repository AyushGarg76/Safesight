"""
app_v3.py — Helmet detection inference
Uses the v3 model (MobileNetV3-Large, 224x224 input).
Detection pipeline: SSDLite person detector → MediaPipe face localizer → classifier.
"""
import cv2
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
)
import torchvision.transforms as T
from PIL import Image
import mediapipe as mp

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===== LOAD CLASSIFIER =====
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, 2)
)
checkpoint = torch.load("helmet_model_best_v3.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to(device)
class_names = checkpoint["class_names"]
print(f"Classes: {class_names} | Best val acc: {checkpoint.get('val_acc', '?'):.4f}")

# ===== TRANSFORM — must match 224x224 used in training =====
classify_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== PERSON DETECTOR =====
det_weights    = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
person_det     = ssdlite320_mobilenet_v3_large(weights=det_weights)
person_det.eval().to(device)
det_transform  = T.Compose([T.ToTensor()])

# ===== MEDIAPIPE =====
mp_face      = mp.solutions.face_detection
face_detect  = mp_face.FaceDetection(min_detection_confidence=0.4)

# ===== CONFIG =====
VIDEO_PATH          = "test_video5.mp4"
CONF_THRESHOLD      = 0.75    # classifier confidence to show a label
PERSON_SCORE        = 0.50    # SSD person confidence
DETECT_EVERY_N      = 3       # run SSD every N frames; track in-between
HISTORY_SIZE        = 7
ALERT_THRESHOLD     = 4       # violations in last HISTORY_SIZE frames = alert

# ===== HELPERS =====
def safe_crop(img, y1, y2, x1, x2):
    h, w = img.shape[:2]
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    crop = img[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

def classify_head(crop):
    if crop is None or crop.shape[0] < 25 or crop.shape[1] < 25:
        return None, None
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t   = classify_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item()

def detect_persons(pil_img):
    t = det_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = person_det(t)[0]
    boxes = []
    for box, lbl, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if lbl.item() == 1 and score.item() > PERSON_SCORE:
            boxes.append(tuple(box.int().tolist()))
    return boxes

# ===== VIDEO LOOP =====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

history     = []
frame_count = 0
last_boxes  = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended")
        break

    frame = cv2.resize(frame, (800, 600))
    fh, fw = frame.shape[:2]
    frame_count += 1

    # Run person detector every N frames
    if frame_count % DETECT_EVERY_N == 1:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last_boxes = detect_persons(pil)

    violations = 0

    for (px1, py1, px2, py2) in last_boxes:
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(fw, px2), min(fh, py2)
        pw, ph = px2 - px1, py2 - py1
        if pw < 40 or ph < 60:
            continue

        person_roi = frame[py1:py2, px1:px2]
        roi_h, roi_w = person_roi.shape[:2]

        # Try MediaPipe face first
        crop = None
        hx1, hy1, hx2, hy2 = px1, py1, px2, py1 + int(ph * 0.30)
        try:
            rgb_roi      = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            face_results = face_detect.process(rgb_roi)
            if face_results.detections:
                b   = face_results.detections[0].location_data.relative_bounding_box
                fx1 = int(b.xmin * roi_w)
                fy1 = int(b.ymin * roi_h)
                fx2 = fx1 + int(b.width  * roi_w)
                fy2 = fy1 + int(b.height * roi_h)
                # Expand upward to capture the helmet area above the face
                expand = int((fy2 - fy1) * 0.75)
                fy1    = max(0, fy1 - expand)
                candidate = safe_crop(person_roi, fy1, fy2, fx1, fx2)
                if candidate is not None:
                    crop = candidate
                    hx1, hy1 = px1 + fx1, py1 + fy1
                    hx2, hy2 = px1 + fx2, py1 + fy2
        except Exception:
            pass

        # Fallback: top 30% of person box
        if crop is None:
            crop = safe_crop(person_roi, 0, int(roi_h * 0.30), 0, roi_w)

        if crop is None:
            continue

        label, conf = classify_head(crop)
        if label is None:
            continue

        # Draw
        if label == "no_helmet" and conf > CONF_THRESHOLD:
            violations += 1
            color = (0, 0, 255)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
            cv2.putText(frame, f"NO HELMET {conf:.0%}",
                        (px1, max(14, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        elif label == "helmet" and conf > CONF_THRESHOLD:
            color = (0, 200, 0)
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, f"HELMET {conf:.0%}",
                        (px1, max(14, py1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Temporal alert
    history.append(1 if violations > 0 else 0)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    if sum(history) >= ALERT_THRESHOLD:
        cv2.rectangle(frame, (0, 0), (fw, 55), (0, 0, 160), -1)
        cv2.putText(frame, "ALERT: PERSON WITHOUT HELMET DETECTED",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    cv2.putText(frame, f"Persons: {len(last_boxes)}  Violations: {violations}",
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SafeSight v3 — Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

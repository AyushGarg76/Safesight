import cv2
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import mediapipe as mp

# ===== LOAD MODEL =====
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

checkpoint = torch.load("helmet_model_best.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class_names = checkpoint['class_names']

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===== MEDIAPIPE =====
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# ===== VIDEO PATH (CHANGE THIS) =====
VIDEO_PATH = r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\videoplayback.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error opening video")
    exit()

# ===== CLASSIFIER FUNCTION =====
def classify(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# ===== LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video finished")
        break

    frame = cv2.resize(frame, (800, 600))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # ===== EXPAND FOR HELMET =====
            y1 = max(0, y - int(bh * 0.7))
            y2 = y + bh
            x1 = max(0, x)
            x2 = min(w, x + bw)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            label, conf = classify(crop)

            color = (0,255,0) if label=="helmet" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    cv2.imshow("Helmet Detection - Video", frame)

    # press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
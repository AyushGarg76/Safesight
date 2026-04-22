import cv2
from color import helmet_color_mask, clean_mask
import os 

# ---------------- PERSON DETECTOR ----------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

import joblib
from hog_utils import extract_hog

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model", "svm_model.pkl")

model = joblib.load(model_path)

# model = joblib.load("model/svm_model.pkl")

def detect_persons(frame):
    boxes, _ = hog.detectMultiScale(frame,
                                    winStride=(8, 8),
                                    padding=(8, 8),
                                    scale=1.05)
    return boxes


# ---------------- HEAD EXTRACTION ----------------
def extract_head(frame, box):
    x, y, w, h = box
    head_y1 = y
    head_y2 = y + int(h * 0.25)

    head_x1 = x + int(w * 0.2)
    head_x2 = x + int(w * 0.8)

    head = frame[head_y1:head_y2, head_x1:head_x2]
    return head


# ---------------- HELMET DETECTION ----------------
import cv2

def detect_helmet(head_img):
    if head_img.size == 0:
        return False

    img_rgb = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)

    features, _ = extract_hog(img_rgb, patch_size=(64, 64))

    prediction = model.predict([features])[0]

    return prediction == "helmet"

# ---------------- MAIN PIPELINE ----------------
def process_frame(frame):
    persons = detect_persons(frame)

    annotated = frame.copy()
    violations = 0

    for (x, y, w, h) in persons:
        head = extract_head(frame, (x, y, w, h))

        has_helmet = detect_helmet(head)

        if h / w < 1.2:
            continue

        if not has_helmet:
            violations += 1

            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)

            cv2.putText(annotated, "NO HELMET",
                        (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2)

    return annotated, violations
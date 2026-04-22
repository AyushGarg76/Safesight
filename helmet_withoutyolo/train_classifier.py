import os
import cv2
import numpy as np
from hog_utils import extract_hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATASET_PATH = "dataset"

X = []
y = []

labels = ["helmet", "no_helmet"]

print("📥 Loading dataset...")

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features, _ = extract_hog(img_rgb, patch_size=(64, 64))

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Dataset size: {len(X)} samples")

# ---------------- TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("🧠 Training SVM...")

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")

print("✅ Model saved at model/svm_model.pkl")
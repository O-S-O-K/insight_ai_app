# ======================================================
# retrain_cnn.py - Fine-tune CNN with user feedback
# ======================================================

import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[0]
FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
MODEL_PATH = ROOT_DIR / "models/cnn_model.h5"
FINETUNED_MODEL_PATH = ROOT_DIR / "models/cnn_model_finetuned.h5"

# -----------------------------
# Load feedback CSV
# -----------------------------
if not FEEDBACK_CSV.exists():
    print(f"No feedback CSV found at {FEEDBACK_CSV}. Exiting.")
    exit(0)

df = pd.read_csv(FEEDBACK_CSV)
if df.empty:
    print("Feedback CSV is empty. Nothing to retrain.")
    exit(0)

# -----------------------------
# Load images and labels
# -----------------------------
X, y = [], []
for _, row in df.iterrows():
    img_path = FEEDBACK_IMG_DIR / row["uploaded_filename"]
    if img_path.exists():
        img = Image.open(img_path).resize((224, 224))
        X.append(img_to_array(img) / 255.0)
        y.append(row["user_label"])
    else:
        print(f"Warning: Image not found: {img_path}")

if len(X) == 0:
    print("No valid images found. Exiting.")
    exit(0)

X = np.array(X)
print(f"Loaded {len(X)} feedback images for retraining.")

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)
print(f"Number of unique classes: {num_classes}")

# -----------------------------
# Load model
# -----------------------------
if not MODEL_PATH.exists():
    print(f"Original CNN model not found at {MODEL_PATH}. Exiting.")
    exit(0)

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
print("Loaded original model and compiled.")

# -----------------------------
# Train / fine-tune
# -----------------------------
print("Starting fine-tuning...")
model.fit(
    X,
    y_categorical,
    epochs=5,
    batch_size=8,
    validation_split=0.2,
    shuffle=True,
)
print("Fine-tuning complete.")

# -----------------------------
# Save updated model
# -----------------------------
FINETUNED_MODEL_PATH.parent.mkdir(exist_ok=True)
model.save(FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved at {FINETUNED_MODEL_PATH}")

# -----------------------------
# Optionally: Clear feedback CSV if desired
# -----------------------------
# Uncomment to reset feedback log after retraining
# FEEDBACK_CSV.unlink()
# print("Feedback CSV cleared.")

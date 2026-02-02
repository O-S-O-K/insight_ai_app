# ======================================================
# retrain_cnn.py - Fine-tune CNN with user feedback
# Robust version: skips invalid/missing filenames
# ======================================================

import pandas as pd
from pathlib import Path
from PIL import Image, ExifTags
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
MODEL_PATH = ROOT_DIR / "models/cnn_baseline_functional.h5"
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
# EXIF-safe image loader
# -----------------------------
def load_image_exif_safe(path):
    """Load an image and apply EXIF rotation if present (mobile-safe)."""
    img = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img.resize((224, 224))

# -----------------------------
# Load images and labels
# -----------------------------
X, y = [], []
skipped = 0

for _, row in df.iterrows():
    filename = row.get("uploaded_filename")
    if not isinstance(filename, str) or not filename.strip():
        print(f"Skipping invalid filename: {filename}")
        skipped += 1
        continue

    img_path = FEEDBACK_IMG_DIR / filename
    if img_path.exists():
        img = load_image_exif_safe(img_path)
        X.append(img_to_array(img) / 255.0)
        y.append(row["user_label"])
    else:
        print(f"Warning: Image not found: {img_path}")
        skipped += 1

if len(X) == 0:
    print(f"No valid images found. Skipped {skipped} invalid entries. Exiting.")
    exit(0)

X = np.array(X)
print(f"Loaded {len(X)} feedback images for retraining. Skipped {skipped} entries.")

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
# Optional: Clear feedback CSV after retraining
# -----------------------------
# Uncomment if you want to reset feedback log
# FEEDBACK_CSV.unlink()
# print("Feedback CSV cleared.")

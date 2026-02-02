# ======================================================
# retrain_cnn.py - Fine-tune canonical CNN with user feedback
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[0]
FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
MODEL_PATH = ROOT_DIR / "models/cnn_baseline_functional.h5"
FINETUNED_MODEL_PATH = ROOT_DIR / "models/cnn_model_finetuned.h5"

IMG_SIZE = (224, 224)
EPOCHS = 5
BATCH_SIZE = 8

# -----------------------------
# Load feedback
# -----------------------------
if not FEEDBACK_CSV.exists():
    print("No feedback CSV found. Exiting.")
    exit(0)

df = pd.read_csv(FEEDBACK_CSV)
if df.empty:
    print("Feedback CSV is empty. Nothing to retrain.")
    exit(0)

# -----------------------------
# EXIF-safe image loader
# -----------------------------
def load_image_safe(path):
    img = Image.open(path)
    try:
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(274)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img.resize(IMG_SIZE)

# -----------------------------
# Load images and labels
# -----------------------------
X, y = [], []
skipped = 0

for _, row in df.iterrows():
    filename = row.get("uploaded_filename")
    if not isinstance(filename, str) or not filename.strip():
        skipped += 1
        continue

    img_path = FEEDBACK_IMG_DIR / filename
    if img_path.exists():
        img = load_image_safe(img_path)
        X.append(img_to_array(img)/255.0)
        y.append(row["user_label"])
    else:
        skipped += 1

if len(X) == 0:
    print(f"No valid images to retrain. Skipped {skipped} entries.")
    exit(0)

X = np.array(X)

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(le.classes_))

# -----------------------------
# Load canonical CNN
# -----------------------------
if not MODEL_PATH.exists():
    print(f"Original model not found at {MODEL_PATH}. Exiting.")
    exit(0)

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
print("Loaded original CNN model.")

# -----------------------------
# Fine-tune
# -----------------------------
print(f"Training on {len(X)} feedback images...")
model.fit(X, y_categorical, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, shuffle=True)
model.save(FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved at {FINETUNED_MODEL_PATH}")

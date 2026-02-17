"""
auto_retrain.py

Semi-automatic retraining of the CNN based on new feedback collected
from the InsightAI app.
"""

from pathlib import Path
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np

# -----------------------------
# Paths
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[0]
FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
MODEL_PATH = ROOT_DIR / "models/cnn_model.h5"
FINETUNED_MODEL_PATH = ROOT_DIR / "models/cnn_model_finetuned.h5"
LAST_RETRAIN_FILE = ROOT_DIR / "last_retrain.txt"

# -----------------------------
# Config
# -----------------------------
MIN_NEW_FEEDBACK = 50  # minimum new entries to trigger retraining
EPOCHS = 5
BATCH_SIZE = 8
IMG_SIZE = (224, 224)

# -----------------------------
# Load feedback
# -----------------------------
if not FEEDBACK_CSV.exists():
    print("No feedback CSV found. Nothing to do.")
    exit(0)

df = pd.read_csv(FEEDBACK_CSV)
total_feedback = len(df)

# Load last retrain counter
last_count = 0
if LAST_RETRAIN_FILE.exists():
    last_count = int(LAST_RETRAIN_FILE.read_text().strip())

new_feedback_count = total_feedback - last_count
if new_feedback_count < MIN_NEW_FEEDBACK:
    print(f"Not enough new feedback ({new_feedback_count}/{MIN_NEW_FEEDBACK}). Exiting.")
    exit(0)

print(f"Detected {new_feedback_count} new feedback entries. Retraining...")

# -----------------------------
# Prepare training data
# -----------------------------
X, y = [], []
for _, row in df.iterrows():
    img_path = FEEDBACK_IMG_DIR / row["uploaded_filename"]
    if img_path.exists():
        img = Image.open(img_path).resize(IMG_SIZE)
        X.append(img_to_array(img)/255.0)
        y.append(row["user_label"])

if len(X) == 0:
    print("No valid images found. Exiting.")
    exit(0)

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(le.classes_))

# -----------------------------
# Load and compile model
# -----------------------------
if not MODEL_PATH.exists():
    print(f"Original model not found at {MODEL_PATH}. Exiting.")
    exit(0)

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
print("Loaded original model.")

# -----------------------------
# Fine-tune
# -----------------------------
model.fit(
    X,
    y_categorical,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    shuffle=True
)
model.save(FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved at {FINETUNED_MODEL_PATH}")

# -----------------------------
# Update retrain counter
# -----------------------------
LAST_RETRAIN_FILE.write_text(str(total_feedback))
print("Updated last retrain counter.")

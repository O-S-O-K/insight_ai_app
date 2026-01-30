"""
auto_retrain.py

Checks if enough new feedback exists and retrains the CNN model if threshold is met.
"""

from pathlib import Path
import pandas as pd
from retrain_cnn import ROOT_DIR, FEEDBACK_CSV, FEEDBACK_IMG_DIR, MODEL_PATH, FINETUNED_MODEL_PATH
from retrain_cnn import load_model, Image, np, to_categorical, LabelEncoder, img_to_array, Adam

# -----------------------------
# CONFIG
# -----------------------------
MIN_NEW_FEEDBACK = 50  # minimum number of new feedback entries to trigger retraining
LAST_RETRAIN_FILE = ROOT_DIR / "last_retrain.txt"

# -----------------------------
# Check last retrain
# -----------------------------
last_retrain_count = 0
if LAST_RETRAIN_FILE.exists():
    last_retrain_count = int(LAST_RETRAIN_FILE.read_text().strip())

# -----------------------------
# Load feedback
# -----------------------------
if not FEEDBACK_CSV.exists():
    print("No feedback CSV found. Nothing to do.")
    exit(0)

df = pd.read_csv(FEEDBACK_CSV)
new_feedback_count = len(df) - last_retrain_count

if new_feedback_count < MIN_NEW_FEEDBACK:
    print(f"Not enough new feedback ({new_feedback_count}/{MIN_NEW_FEEDBACK}). Exiting.")
    exit(0)

print(f"New feedback detected: {new_feedback_count}. Retraining now...")

# -----------------------------
# Prepare data
# -----------------------------
X, y = [], []
for _, row in df.iterrows():
    img_path = FEEDBACK_IMG_DIR / row["uploaded_filename"]
    if img_path.exists():
        img = Image.open(img_path).resize((224, 224))
        X.append(img_to_array(img) / 255.0)
        y.append(row["user_label"])

if len(X) == 0:
    print("No valid images found. Exiting.")
    exit(0)

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

# -----------------------------
# Load model
# -----------------------------
if not MODEL_PATH.exists():
    print(f"Original CNN model not found at {MODEL_PATH}. Exiting.")
    exit(0)

model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# -----------------------------
# Train / fine-tune
# -----------------------------
model.fit(
    X,
    y_categorical,
    epochs=5,
    batch_size=8,
    validation_split=0.2,
    shuffle=True,
)
model.save(FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved at {FINETUNED_MODEL_PATH}")

# -----------------------------
# Update last retrain counter
# -----------------------------
LAST_RETRAIN_FILE.write_text(str(len(df)))
print("Last retrain count updated.")

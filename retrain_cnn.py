# retrain_cnn.py
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array

# Paths
FEEDBACK_CSV = Path("feedback_log.csv")
FEEDBACK_IMG_DIR = Path("feedback_images")
MODEL_PATH = Path("models/cnn_model.h5")
FINETUNED_MODEL_PATH = Path("models/cnn_model_finetuned.h5")

# Load feedback CSV
df = pd.read_csv(FEEDBACK_CSV)

# Load images and labels
X, y = [], []
for _, row in df.iterrows():
    img_path = FEEDBACK_IMG_DIR / row["uploaded_filename"]
    if img_path.exists():
        img = Image.open(img_path).resize((32,32))
        X.append(img_to_array(img)/255.0)
        y.append(row["user_selected_label"])

X = np.array(X)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=10)  # match CNN output

# Load model
model = load_model(MODEL_PATH)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train / fine-tune
model.fit(X, y_categorical, epochs=5, batch_size=8, validation_split=0.2, shuffle=True)

# Save updated model
model.save(FINETUNED_MODEL_PATH)
print(f"Fine-tuned model saved at {FINETUNED_MODEL_PATH}")

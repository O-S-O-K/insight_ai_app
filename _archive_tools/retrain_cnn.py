import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "feedback_images"
CSV = ROOT / "feedback_log.csv"
MODEL_OUT = ROOT / "models/cnn_model_finetuned.h5"

IMG_SIZE = (224, 224)

if not CSV.exists():
    print("No feedback yet.")
    exit()

df = pd.read_csv(CSV)
X, y = [], []

for _, row in df.iterrows():
    img_path = IMG_DIR / row["uploaded_filename"]
    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = preprocess_input(np.array(img))
    X.append(arr)
    y.append(row["user_label"])

X = np.array(X)

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base.trainable = False

x = GlobalAveragePooling2D()(base.output)
output = Dense(len(le.classes_), activation="softmax")(x)
model = Model(base.input, output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y_cat, epochs=5, batch_size=8, validation_split=0.2)

MODEL_OUT.parent.mkdir(exist_ok=True)
model.save(MODEL_OUT)

print("Fine-tuned MobileNetV2 saved.")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 🔑 TEMPORARY starter labels (can be anything)
initial_labels = ["dog", "cat", "car"]

label_map = {i: lbl for i, lbl in enumerate(initial_labels)}

# Save label map
with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump(label_map, f)

# Build model
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
output = Dense(len(initial_labels), activation="softmax")(x)

model = Model(base.input, output)
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Save bootstrap model
model.save(MODEL_DIR / "cnn_model.h5")

print("✅ Bootstrap model created.")

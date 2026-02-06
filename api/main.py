import sys
from pathlib import Path
import io
import base64
import traceback

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.cm as cm

# ----------------------------
# Path setup
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ----------------------------
# Config
# ----------------------------
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_model.h5"
IMG_SIZE = (224, 224)

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Insight AI API")

# ----------------------------
# Utilities
# ----------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

# ----------------------------
# Load model once
# ----------------------------
model = load_model(MODEL_PATH)
last_conv_layer_name = find_last_conv_layer(model)

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "gradcam_layer": last_conv_layer_name,
    }

# ----------------------------
# Prediction
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB").resize(IMG_SIZE)
        x = np.expand_dims(np.array(img), axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return {
            "class": class_idx,
            "confidence": confidence,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# Caption (mock for now)
# ----------------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    return {"caption": "This is a mock caption for testing."}

# ----------------------------
# Grad-CAM
# ----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        # Load image
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)

        x = np.expand_dims(np.array(img_resized), axis=0)
        x = preprocess_input(x)

        # Grad-CAM model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output],
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Apply colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)

        # Overlay on original image
        heatmap_img = Image.fromarray(heatmap_colored).resize(IMG_SIZE)
        overlay = Image.blend(img_resized, heatmap_img, alpha=0.4)

        # Encode as base64
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "gradcam_layer": last_conv_layer_name,
            "heatmap_base64":f"data:image/png;base64,{overlay_b64}",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

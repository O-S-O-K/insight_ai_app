# main.py
import os
import sys
from pathlib import Path
import io
import base64
import json
import traceback

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.cm as cm

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# ----------------------------
# Path setup
# ----------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FEEDBACK_DIR = ROOT / "feedback_images"
FEEDBACK_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "cnn_model.h5"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

IMG_SIZE = (224, 224)

# ----------------------------
# Load model metadata
# ----------------------------
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    LABEL_MAP = metadata.get("classes", {})
else:
    LABEL_MAP = {}

# ----------------------------
# Load CNN model
# ----------------------------
model = load_model(MODEL_PATH)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

last_conv_layer_name = find_last_conv_layer(model)

# ----------------------------
# Load BLIP model for captioning
# ----------------------------
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Insight AI API")

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
# Prediction endpoint
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
        class_name = LABEL_MAP.get(str(class_idx), f"Class {class_idx}")

        return {
            "class_idx": class_idx,
            "class_name": class_name,
            "confidence": confidence,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# Caption endpoint
# ----------------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        # BLIP expects PIL Image
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs)
        caption_text = blip_processor.decode(output_ids[0], skip_special_tokens=True)
        return {"caption": caption_text}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# Grad-CAM endpoint
# ----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), class_idx: int = None):
    try:
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        x = np.expand_dims(np.array(img_resized), axis=0)
        x = preprocess_input(x)

        # Grad-CAM model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        # Determine target class
        preds = model.predict(x)
        if class_idx is None:
            class_idx = int(np.argmax(preds[0]))
        class_name = LABEL_MAP.get(str(class_idx), f"Class {class_idx}")

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
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
        heatmap_img = Image.fromarray(heatmap_colored).resize(IMG_SIZE)

        # Overlay original image
        overlay = Image.blend(img_resized, heatmap_img, alpha=0.4)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "pred_class_idx": class_idx,
            "pred_class_name": class_name,
            "heatmap_base64": f"data:image/png;base64,{overlay_b64}",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# Human feedback endpoint
# ----------------------------
@app.post("/feedback")
async def feedback(file: UploadFile = File(...), entry: str = Form(...)):
    try:
        # Save image
        img_path = FEEDBACK_DIR / file.filename
        with open(img_path, "wb") as f:
            f.write(await file.read())

        # Save feedback JSON
        entry_data = json.loads(entry) if isinstance(entry, str) else entry
        feedback_log_path = FEEDBACK_DIR / "feedback_log.json"
        if feedback_log_path.exists():
            with open(feedback_log_path, "r") as f:
                log = json.load(f)
        else:
            log = []

        log.append({
            "filename": file.filename,
            "feedback": entry_data.get("feedback"),
            "rating": entry_data.get("rating"),
        })

        with open(feedback_log_path, "w") as f:
            json.dump(log, f, indent=2)

        return {"status": "success"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

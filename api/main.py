# main.py
import os
from pathlib import Path
import io
import base64
import json
import traceback
import hashlib

# Force CPU execution and quieter TF logs in CPU-only environments
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.cm as cm

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# ----------------------------
# Path setup
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FEEDBACK_DIR = ROOT / "feedback_images"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "cnn_model.h5"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

IMG_SIZE = (224, 224)
TOP_K = 3  # number of top predictions to return

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
model = keras.models.load_model(MODEL_PATH)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
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
# Prediction endpoint (top-K)
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB").resize(IMG_SIZE)
        x = np.expand_dims(np.array(img), axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)[0]
        top_indices = preds.argsort()[-TOP_K:][::-1]  # top K indices

        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                "class_idx": int(idx),
                "class_name": LABEL_MAP.get(str(idx), f"Class {idx}"),
                "confidence": float(preds[idx])
            })

        return {"predictions": top_predictions}

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
# Grad-CAM endpoint (multi-class)
# ----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), top_k: int = TOP_K):
    try:
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)
        x = np.expand_dims(np.array(img_resized), axis=0)
        x = preprocess_input(x)

        # Grad-CAM model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        preds = model.predict(x)[0]
        top_indices = preds.argsort()[-top_k:][::-1]

        gradcam_results = []

        for idx in top_indices:
            # Compute Grad-CAM
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x)
                loss = predictions[:, idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
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

            gradcam_results.append({
                "class_idx": int(idx),
                "class_name": LABEL_MAP.get(str(idx), f"Class {idx}"),
                "confidence": float(preds[idx]),
                "heatmap_base64": f"data:image/png;base64,{overlay_b64}"
            })

        return {"gradcams": gradcam_results}

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

import os
import io
import json
import base64
import sys
from typing import Optional
from pathlib import Path

# ----------------------------
# Ensure container can find utils/ and other sibling modules
ROOT = Path(__file__).resolve().parent  # `/app` inside container
sys.path.insert(0, str(ROOT))
# ----------------------------

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import requests

# reuse utilities from repo
from utils.gradcam import (
    find_last_conv_layer,
    get_gradcam_heatmap,
    overlay_heatmap,
)

# Model utilities (similar to frontend)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ----------------------------
# Paths for models and feedback
MODELS_DIR = ROOT / "models"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FINETUNED_MODEL_PATH = MODELS_DIR / "cnn_model_finetuned.h5"
FEEDBACK_CSV = ROOT / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

# ----------------------------
# FastAPI app setup
app = FastAPI(title="InsightAI Backend")

# Allow all origins for demo simplicity (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")  # optional, for BLIP via HF Inference API
HF_BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# ----------------------------
# Load model metadata
def load_model_metadata():
    if MODEL_METADATA_PATH.exists():
        return json.loads(MODEL_METADATA_PATH.read_text())
    return {
        "model_name": "mobilenetv2_imagenet",
        "version": "baseline",
        "architecture": "MobileNetV2 (ImageNet)",
        "trained_on": "ImageNet (pretrained)",
        "last_updated": "N/A",
    }

meta = load_model_metadata()

# Load or instantiate model
def load_cnn_model():
    if FINETUNED_MODEL_PATH.exists():
        model = load_model(FINETUNED_MODEL_PATH, compile=False)
        source = "Fine-tuned model"
    else:
        model = MobileNetV2(weights="imagenet", include_top=True)
        source = "ImageNet pretrained"
    return model, source

model, model_source = load_cnn_model()

# ----------------------------
# Helpers
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return ImageOps.exif_transpose(image) if hasattr(ImageOps, 'exif_transpose') else image

# ----------------------------
# Routes
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_resized = img.resize((224, 224))
        x = np.array(img_resized).astype(np.float32)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]
        predictions = [{"label": label, "score": float(score)} for (_id, label, score) in decoded]
        return {"predictions": predictions, "model_version": meta.get("version", "N/A")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), class_idx: Optional[int] = Form(None)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_resized = img.resize((224, 224))
        x = np.array(img_resized).astype(np.float32)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)

        last_conv = find_last_conv_layer(model)
        heatmap = get_gradcam_heatmap(model, last_conv, x, class_idx=class_idx)
        overlay = overlay_heatmap(heatmap, img, alpha=0.4)

        buffer = io.BytesIO()
        overlay.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"overlay_base64": encoded}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    if not HF_TOKEN:
        raise HTTPException(status_code=400, detail="HF_TOKEN not set on server")
    try:
        contents = await file.read()
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        api_url = f"https://api-inference.huggingface.co/models/{HF_BLIP_MODEL}"
        resp = requests.post(api_url, headers=headers, data=contents, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) and isinstance(data[0], dict):
            caption = data[0].get("generated_text") or data[0].get("caption")
        else:
            caption = str(data)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(entry: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        ed = json.loads(entry)
        if file is not None:
            contents = await file.read()
            img_path = FEEDBACK_IMG_DIR / f"{ed.get('image_hash', 'unknown')}.jpg"
            with open(img_path, "wb") as f:
                f.write(contents)
            ed["image_path"] = str(img_path)

        import pandas as pd
        df = pd.read_csv(FEEDBACK_CSV) if FEEDBACK_CSV.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([ed])], ignore_index=True)
        df.to_csv(FEEDBACK_CSV, index=False)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
def metadata():
    return meta

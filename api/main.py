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

# Use tf.keras from the bundled TensorFlow version
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
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
SAVEDMODEL_DIR = MODELS_DIR / "cnn_baseline_savedmodel"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Alternative model paths to try
ALT_MODEL_PATHS = [
    MODELS_DIR / "cnn_baseline_functional.h5",
    MODELS_DIR / "cnn_baseline.h5",
]

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
# Load CNN model with proper error handling
# ----------------------------
def load_model_safe():
    """Load model with fallback mechanisms and version compatibility handling"""
    model = None
    load_errors = []

    # Try SavedModel format first
    if SAVEDMODEL_DIR.exists():
        try:
            print(f"Attempting to load SavedModel from {SAVEDMODEL_DIR}")
            temp_model = tf.keras.models.load_model(str(SAVEDMODEL_DIR), compile=False)
            print(f"SavedModel loaded, type: {type(temp_model)}")

            # Validate it's a proper Keras model, not a _UserObject
            if not hasattr(temp_model, 'layers'):
                print(f"WARNING: SavedModel loaded as {type(temp_model).__name__} without 'layers' attribute")
                print("This SavedModel was likely created with tf.saved_model.save() instead of model.save()")
                print("Falling back to H5 format...")
                temp_model = None
            elif not hasattr(temp_model, 'predict'):
                print(f"WARNING: SavedModel missing 'predict' method")
                print("Falling back to H5 format...")
                temp_model = None
            else:
                print(f"SavedModel validation successful")
                model = temp_model
        except Exception as e:
            error_msg = f"SavedModel loading failed: {e}"
            print(error_msg)
            load_errors.append(error_msg)
            model = None

    # Fallback to H5 format if SavedModel didn't work
    if model is None and MODEL_PATH.exists():
        print(f"Attempting to load H5 model from {MODEL_PATH}")

        try:
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            print(f"  ✓ H5 model loaded successfully, type: {type(model)}")

            # Validate H5 model
            if not hasattr(model, 'layers'):
                raise AttributeError(f"H5 model (type: {type(model)}) does not have 'layers' attribute")
            if not hasattr(model, 'predict'):
                raise AttributeError(f"H5 model (type: {type(model)}) does not have 'predict' method")

            print(f"  ✓ H5 model validation successful")

        except Exception as e:
            error_msg = f"H5 model loading failed: {str(e)[:100]}"
            print(f"  ✗ {error_msg}")
            load_errors.append(error_msg)
            model = None

    # Try alternative H5 model files if primary failed
    if model is None:
        for alt_path in ALT_MODEL_PATHS:
            if not alt_path.exists():
                continue

            print(f"Attempting to load alternative H5 model from {alt_path.name}")
            try:
                model = tf.keras.models.load_model(str(alt_path), compile=False)
                print(f"  ✓ Alternative H5 model loaded successfully, type: {type(model)}")

                # Validate
                if not hasattr(model, 'layers'):
                    raise AttributeError("Model missing 'layers' attribute")
                if not hasattr(model, 'predict'):
                    raise AttributeError("Model missing 'predict' method")

                print(f"  ✓ Alternative model validation successful")
                break

            except Exception as e:
                error_msg = f"Alternative {alt_path.name} failed: {str(e)[:100]}"
                print(f"  ✗ {error_msg}")
                load_errors.append(error_msg)
                model = None
                continue

    # If all loading strategies failed
    if model is None:
        raise RuntimeError(
            f"Failed to load model from SavedModel and all H5 formats.\n"
            f"Errors:\n" + "\n".join(f"  - {err}" for err in load_errors) + "\n\n"
            f"The model files appear to be incompatible with TensorFlow 2.10.1.\n"
            f"Please regenerate the model using: python regenerate_savedmodel.py"
        )

    print(f"✓ Model loaded successfully with {len(model.layers)} layers")
    return model

def find_last_conv_layer(model):
    """Find the last Conv2D layer in the model for Grad-CAM"""
    if not hasattr(model, 'layers'):
        raise AttributeError(f"Model (type: {type(model)}) does not have 'layers' attribute")

    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            print(f"✓ Found last Conv2D layer: {layer.name}")
            return layer.name

    # If no Conv2D found, try to find it in nested models (like MobileNetV2)
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, layers.Conv2D):
                    print(f"✓ Found last Conv2D layer in nested model: {sublayer.name}")
                    return sublayer.name

    raise ValueError("No Conv2D layer found in model. Grad-CAM requires a convolutional layer.")

# Load and validate model
print("=" * 60)
print("INITIALIZING INSIGHT AI BACKEND")
print("=" * 60)
model = load_model_safe()
last_conv_layer_name = find_last_conv_layer(model)
print(f"✓ Grad-CAM configured for layer: {last_conv_layer_name}")
print("=" * 60)

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
        "model_loaded": model is not None,
        "model_type": str(type(model)),
        "has_layers": hasattr(model, 'layers'),
        "num_layers": len(model.layers) if hasattr(model, 'layers') else 0,
        "gradcam_layer": last_conv_layer_name,
        "tensorflow_version": tf.__version__,
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

        # Validate model has required attributes
        if not hasattr(model, 'get_layer'):
            raise AttributeError("Model does not have 'get_layer' method required for Grad-CAM")

        if not hasattr(model, 'inputs'):
            raise AttributeError("Model does not have 'inputs' attribute required for Grad-CAM")

        if not hasattr(model, 'output'):
            raise AttributeError("Model does not have 'output' attribute required for Grad-CAM")

        # Grad-CAM model
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError as e:
            raise ValueError(f"Could not find layer '{last_conv_layer_name}' in model: {e}")

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

            if grads is None:
                print(f"Warning: Gradient is None for class {idx}, skipping")
                continue

            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)  # Add epsilon to avoid division by zero
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

# main.py
import os
from pathlib import Path
import io
import base64
import json
import traceback
import hashlib

print("Step 1: Basic imports OK")

# Force CPU execution and quieter TF logs in CPU-only environments
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

print("Step 2: Environment variables set")

import numpy as np
from PIL import Image
import tensorflow as tf

print(f"Step 3: TensorFlow {tf.__version__} loaded")

# Handle both Keras 3 (separate package) and legacy tf.keras
try:
    # Try legacy tf.keras first (works with TF_USE_LEGACY_KERAS=1)
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    print("Step 4: Using tf.keras (legacy)")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    # Fall back to standalone Keras 3
    print(f"Step 4a: tf.keras failed ({e}), trying standalone keras")
    import keras
    from keras import layers
    from keras.models import Model
    from keras.applications.mobilenet_v2 import preprocess_input
    print("Step 4b: Using standalone keras")

import matplotlib.cm as cm

print("Step 5: Matplotlib loaded")

# Import torch for BLIP model
try:
    import torch
    print("Step 5a: PyTorch loaded")
except ImportError:
    torch = None
    print("Step 5a: PyTorch not available (BLIP captions will not work)")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

print("Step 6: FastAPI loaded")

# ----------------------------
# Path setup
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FEEDBACK_DIR = ROOT / "feedback_images"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "cnn_model_fixed.h5"
SAVEDMODEL_DIR = MODELS_DIR / "cnn_baseline_savedmodel"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Alternative model paths to try (fallback to old models if needed)
ALT_MODEL_PATHS = [
    MODELS_DIR / "cnn_model.h5",  # Old model as fallback
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

# ----------------------------
# Global variables for models (loaded at startup)
# ----------------------------
model = None
last_conv_layer_name = None
blip_processor = None
blip_model = None
device = None
models_loading = True
models_loaded = False

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Insight AI API")

# ----------------------------
# Background task to load models (non-blocking)
# ----------------------------
def load_models_background():
    """Load models in background thread"""
    global model, last_conv_layer_name, blip_processor, blip_model, device, models_loading, models_loaded

    try:
        print("=" * 60, flush=True)
        print("INITIALIZING INSIGHT AI BACKEND (background)", flush=True)
        print("=" * 60, flush=True)

        # Load CNN model
        print("Loading CNN model...", flush=True)
        model = load_model_safe()
        print(f"CNN model loaded: {type(model)}", flush=True)

        print("Finding last Conv2D layer...", flush=True)
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"✓ Grad-CAM configured for layer: {last_conv_layer_name}", flush=True)

        # Load BLIP model
        print("Loading BLIP captioning model from cache...", flush=True)
        from transformers import BlipProcessor, BlipForConditionalGeneration

        if torch is None:
            raise ImportError("PyTorch is not available. Cannot load BLIP model.")

        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✓ BLIP processor loaded", flush=True)

        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✓ BLIP model loaded", flush=True)

        blip_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)
        print(f"✓ BLIP model moved to device: {device}", flush=True)

        models_loaded = True
        models_loading = False

        print("=" * 60, flush=True)
        print("✓ ALL MODELS LOADED SUCCESSFULLY", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        models_loading = False
        models_loaded = False
        print("=" * 60, flush=True)
        print("❌ MODEL LOADING ERROR:", flush=True)
        print(str(e), flush=True)
        import traceback
        traceback.print_exc()
        print("=" * 60, flush=True)

# ----------------------------
# Startup event - launch background model loading
# ----------------------------
@app.on_event("startup")
async def startup_event():
    """Launch model loading in background (non-blocking)"""
    import threading
    print("Starting model loading in background thread...", flush=True)
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    print("✓ Background model loading initiated - server ready for requests", flush=True)

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "loading" if models_loading else ("ready" if models_loaded else "error"),
        "models_loading": models_loading,
        "models_loaded": models_loaded,
        "model_loaded": model is not None,
        "blip_loaded": blip_model is not None,
        "model_type": str(type(model)) if model else None,
        "gradcam_layer": last_conv_layer_name,
        "tensorflow_version": tf.__version__,
    }

# ----------------------------
# Prediction endpoint (top-K)
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if models are ready
    if models_loading:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are still loading. Please wait and try again in a few moments."}
        )
    if not models_loaded or model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Models failed to load. Please check server logs."}
        )

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
    # Check if models are ready
    if models_loading:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are still loading. Please wait and try again in a few moments."}
        )
    if not models_loaded or blip_model is None or blip_processor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "BLIP model failed to load. Please check server logs."}
        )

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
    # Check if models are ready
    if models_loading:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are still loading. Please wait and try again in a few moments."}
        )
    if not models_loaded or model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Models failed to load. Please check server logs."}
        )

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

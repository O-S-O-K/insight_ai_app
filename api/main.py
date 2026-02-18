# main.py
import os
from pathlib import Path
from typing import Optional
import io
import base64
import json
import traceback
import threading

print("Step 1: Basic imports OK")

# Force CPU execution and quieter TF logs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

# Feature flags
ENABLE_BLIP = os.environ.get("ENABLE_BLIP", "true").lower() == "true"
ENABLE_SHAP = os.environ.get("ENABLE_SHAP", "true").lower() == "true"
ENABLE_CLIP = os.environ.get("ENABLE_CLIP", "true").lower() == "true"
MODEL_TYPE = os.environ.get("MODEL_TYPE", "imagenet")  # "imagenet" | "medical"
ACTIVE_LEARNING_THRESHOLD = float(os.environ.get("ACTIVE_LEARNING_THRESHOLD", "0.5"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "insight-ai-inference")

print("Step 2: Feature flags set")

import numpy as np
from PIL import Image
import tensorflow as tf

print(f"Step 3: TensorFlow {tf.__version__} loaded")

# Handle both Keras 3 and legacy tf.keras
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    print("Step 4: Using tf.keras (legacy)")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    print(f"Step 4a: tf.keras failed ({e}), trying standalone keras")
    import keras
    from keras import layers
    from keras.models import Model
    from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    print("Step 4b: Using standalone keras")

import matplotlib.cm as cm

print("Step 5: Matplotlib loaded")

# PyTorch for BLIP and CLIP
try:
    import torch
    print("Step 5a: PyTorch loaded")
except ImportError:
    torch = None
    print("Step 5a: PyTorch not available (BLIP/CLIP will not work)")

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    MLFLOW_AVAILABLE = True
    print("Step 5b: MLflow loaded")
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False
    print("Step 5b: MLflow not available (install mlflow to enable tracking)")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
    print("Step 5c: SHAP loaded")
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    print("Step 5c: SHAP not available (install shap to enable explainability)")

from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

print("Step 6: FastAPI loaded")

# ----------------------------
# Path setup
# ----------------------------
# In HF Spaces the flat layout puts main.py at /app/main.py,
# so parents[0] == /app. In the original project layout main.py
# was at api/main.py and parents[1] was the project root.
# Use parents[0] to stay within /app regardless of layout.
ROOT = Path(__file__).resolve().parents[0]
MODELS_DIR = ROOT / "models"
FEEDBACK_DIR = ROOT / "feedback_images"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "cnn_baseline_functional.h5"
MEDICAL_MODEL_PATH = MODELS_DIR / "medical_model.h5"
SAVEDMODEL_DIR = MODELS_DIR / "cnn_baseline_savedmodel"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
MEDICAL_METADATA_PATH = MODELS_DIR / "medical_metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.npy"

# Fallback model paths
ALT_MODEL_PATHS = [
    MODELS_DIR / "cnn_model_fixed.h5",
    MODELS_DIR / "cnn_baseline.h5",
]

IMG_SIZE = (224, 224)
TOP_K = 3

# ----------------------------
# Load model metadata
# ----------------------------
if MODEL_TYPE == "medical" and MEDICAL_METADATA_PATH.exists():
    with open(MEDICAL_METADATA_PATH, "r") as f:
        metadata = json.load(f)
elif METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
else:
    metadata = {}

LABEL_MAP = metadata.get("classes", {})
TEMPERATURE = float(metadata.get("temperature", 1.0))

# ----------------------------
# Confidence calibration
# ----------------------------
def apply_temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature scaling for calibrated confidence scores.
    Divides logits by temperature before softmax to reduce overconfidence.
    """
    if temperature == 1.0:
        return logits
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / (exp_scaled.sum() + 1e-8)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(img: Image.Image, model_type: str = "imagenet") -> np.ndarray:
    """Preprocess image for model inference."""
    img_resized = img.resize(IMG_SIZE)
    x = np.expand_dims(np.array(img_resized), axis=0).astype(np.float32)
    # Both imagenet and medical models use MobileNetV2-style normalization [-1, 1]
    x = mobilenet_preprocess(x)
    return x

# ----------------------------
# Model loading
# ----------------------------
def load_model_safe():
    """Load CNN model with fallback mechanisms."""
    primary_path = MEDICAL_MODEL_PATH if (MODEL_TYPE == "medical" and MEDICAL_MODEL_PATH.exists()) else MODEL_PATH
    model = None
    load_errors = []

    if primary_path.exists():
        print(f"Attempting to load model from {primary_path}")
        try:
            model = tf.keras.models.load_model(str(primary_path), compile=False)
            if not hasattr(model, 'layers') or not hasattr(model, 'predict'):
                raise AttributeError("Model missing required attributes")
            print(f"  ✓ Model loaded: {len(model.layers)} layers")
        except Exception as e:
            load_errors.append(f"Primary load failed: {str(e)[:100]}")
            model = None

    if model is None:
        for alt_path in ALT_MODEL_PATHS:
            if not alt_path.exists():
                continue
            try:
                model = tf.keras.models.load_model(str(alt_path), compile=False)
                if not hasattr(model, 'layers') or not hasattr(model, 'predict'):
                    raise AttributeError("Model missing required attributes")
                print(f"  ✓ Fallback model loaded: {alt_path.name}")
                break
            except Exception as e:
                load_errors.append(f"Fallback {alt_path.name} failed: {str(e)[:100]}")
                model = None

    if model is None:
        raise RuntimeError(
            "Failed to load model from all paths.\nErrors:\n" +
            "\n".join(f"  - {err}" for err in load_errors)
        )

    return model

def find_last_conv_layer(model):
    """Find last Conv2D layer in model for Grad-CAM."""
    if not hasattr(model, 'layers'):
        raise AttributeError(f"Model (type: {type(model)}) does not have 'layers' attribute")

    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            print(f"✓ Found last Conv2D layer: {layer.name}")
            return layer.name

    # Search nested layers (e.g., MobileNetV2 inside custom model)
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, layers.Conv2D):
                    print(f"✓ Found Conv2D in nested model: {sublayer.name}")
                    return sublayer.name

    raise ValueError("No Conv2D layer found. Grad-CAM requires a convolutional layer.")

# ----------------------------
# Global model state
# ----------------------------
model = None
last_conv_layer_name = None
blip_processor = None
blip_model = None
clip_model = None
clip_processor = None
shap_background = None
shap_explainer = None
device = None
models_loading = True
models_loaded = False

# Per-type model registry (populated at startup)
_models: dict = {}        # {"imagenet": keras_model, "medical": keras_model}
_model_configs: dict = {} # {"imagenet": {label_map, temperature, last_conv}, ...}

# ----------------------------
# FastAPI app + versioned router
# ----------------------------
app = FastAPI(title="Insight AI API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# ----------------------------
# Background model loading
# ----------------------------
def load_models_background():
    """Load all models in a background thread (non-blocking startup)."""
    global model, last_conv_layer_name, blip_processor, blip_model
    global clip_model, clip_processor, shap_background, shap_explainer
    global device, models_loading, models_loaded
    global _models, _model_configs

    try:
        print("=" * 60, flush=True)
        print("INITIALIZING INSIGHT AI BACKEND (background)", flush=True)
        print(f"MODEL_TYPE: {MODEL_TYPE}", flush=True)
        print("=" * 60, flush=True)

        # Load CNN classifier
        print("Loading CNN model...", flush=True)
        model = load_model_safe()
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"✓ Grad-CAM layer: {last_conv_layer_name}", flush=True)

        # Register primary model in per-type dict
        _models[MODEL_TYPE] = model
        _model_configs[MODEL_TYPE] = {
            "label_map": LABEL_MAP,
            "temperature": TEMPERATURE,
            "last_conv": last_conv_layer_name,
        }

        # Opportunistically load the secondary model (non-fatal if missing/broken)
        secondary_type = "medical" if MODEL_TYPE == "imagenet" else "imagenet"
        secondary_path = MEDICAL_MODEL_PATH if secondary_type == "medical" else MODEL_PATH
        if secondary_path.exists() and secondary_type not in _models:
            try:
                print(f"Loading secondary ({secondary_type}) model...", flush=True)
                sec_model = tf.keras.models.load_model(str(secondary_path), compile=False)
                sec_meta_path = MEDICAL_METADATA_PATH if secondary_type == "medical" else METADATA_PATH
                sec_meta = json.load(open(str(sec_meta_path))) if sec_meta_path.exists() else {}
                sec_conv = find_last_conv_layer(sec_model)
                _models[secondary_type] = sec_model
                _model_configs[secondary_type] = {
                    "label_map": sec_meta.get("classes", {}),
                    "temperature": float(sec_meta.get("temperature", 1.0)),
                    "last_conv": sec_conv,
                }
                print(f"✓ Secondary model loaded: {secondary_type} (layer: {sec_conv})", flush=True)
            except Exception as sec_err:
                print(f"⚠ Secondary model load failed (non-fatal): {sec_err}", flush=True)

        # Load BLIP captioning model
        if ENABLE_BLIP:
            print("Loading BLIP captioning model...", flush=True)
            from transformers import BlipProcessor, BlipForConditionalGeneration
            if torch is None:
                raise ImportError("PyTorch not available. Cannot load BLIP.")
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            blip_model.to(device)
            print(f"✓ BLIP loaded on {device}", flush=True)
        else:
            print("⚠ BLIP disabled (ENABLE_BLIP=false)", flush=True)

        # Load CLIP zero-shot model
        if ENABLE_CLIP and torch is not None:
            print("Loading CLIP zero-shot model...", flush=True)
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model.to(clip_device)
            clip_model.eval()
            print(f"✓ CLIP ViT-B/32 loaded on {clip_device}", flush=True)
        elif ENABLE_CLIP:
            print("⚠ CLIP disabled: PyTorch not available", flush=True)
        else:
            print("⚠ CLIP disabled (ENABLE_CLIP=false)", flush=True)

        # Set up SHAP explainer
        if ENABLE_SHAP and SHAP_AVAILABLE and model is not None:
            print("Setting up SHAP GradientExplainer...", flush=True)
            if SHAP_BACKGROUND_PATH.exists():
                shap_background = np.load(str(SHAP_BACKGROUND_PATH))
                print(f"✓ SHAP background loaded: {shap_background.shape}", flush=True)
            else:
                # Generate random background samples for SHAP
                # 10 samples keeps CPU inference under 60s on HF Spaces free tier
                print("Generating SHAP background dataset (10 samples)...", flush=True)
                shap_background = np.random.uniform(-1, 1, (10, 224, 224, 3)).astype(np.float32)
                np.save(str(SHAP_BACKGROUND_PATH), shap_background)
                print("✓ SHAP background generated and cached", flush=True)
            shap_explainer = shap.GradientExplainer(model, shap_background)
            print("✓ SHAP GradientExplainer ready", flush=True)
        elif ENABLE_SHAP and not SHAP_AVAILABLE:
            print("⚠ SHAP disabled: shap package not installed", flush=True)
        else:
            print("⚠ SHAP disabled (ENABLE_SHAP=false)", flush=True)

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
        traceback.print_exc()
        print("=" * 60, flush=True)

@app.on_event("startup")
async def startup_event():
    """Launch model loading in background (non-blocking)."""
    print("Starting model loading in background thread...", flush=True)
    thread = threading.Thread(target=load_models_background, daemon=True)
    thread.start()
    print("✓ Background model loading initiated - server ready", flush=True)

# ----------------------------
# Helper: check models ready
# ----------------------------
def _check_models_ready():
    """Returns a JSONResponse error if models are not ready, else None."""
    if models_loading:
        return JSONResponse(
            status_code=503,
            content={"error": "Models are still loading. Please try again in a few moments."}
        )
    if not models_loaded or model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Models failed to load. Please check server logs."}
        )
    return None

# ----------------------------
# Root health check (unversioned)
# ----------------------------
@app.get("/")
def health():
    return {
        "status": "loading" if models_loading else ("ready" if models_loaded else "error"),
        "version": "1.0.0",
        "models_loading": models_loading,
        "models_loaded": models_loaded,
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "medical_model_available": MEDICAL_MODEL_PATH.exists(),
        "blip_enabled": ENABLE_BLIP,
        "blip_loaded": blip_model is not None,
        "clip_enabled": ENABLE_CLIP,
        "clip_loaded": clip_model is not None,
        "shap_enabled": ENABLE_SHAP,
        "shap_ready": shap_explainer is not None,
        "mlflow_enabled": MLFLOW_AVAILABLE,
        "gradcam_layer": last_conv_layer_name,
        "tensorflow_version": tf.__version__,
    }

# ----------------------------
# v1: Prediction endpoint
# ----------------------------
@router.post("/predict")
async def predict(file: UploadFile = File(...), model_type: Optional[str] = Form(None)):
    err = _check_models_ready()
    if err:
        return err

    try:
        img = Image.open(file.file).convert("RGB")

        # Resolve which model / config to use for this request
        requested = model_type or MODEL_TYPE
        m = _models.get(requested) or model
        config = _model_configs.get(requested) or _model_configs.get(MODEL_TYPE, {})
        label_map = config.get("label_map", LABEL_MAP)
        temperature = config.get("temperature", TEMPERATURE)
        effective_type = requested if requested in _models else MODEL_TYPE

        x = preprocess_image(img, effective_type)

        # Use model() instead of model.predict() to get raw logits for calibration
        raw_preds = m(x, training=False).numpy()[0]
        calibrated_preds = apply_temperature_scaling(raw_preds, temperature)

        top_indices = calibrated_preds.argsort()[-TOP_K:][::-1]
        top_predictions = [
            {
                "class_idx": int(idx),
                "class_name": label_map.get(str(idx), f"Class {idx}"),
                "confidence": float(calibrated_preds[idx]),
                "raw_confidence": float(raw_preds[idx]),
            }
            for idx in top_indices
        ]

        top1_conf = top_predictions[0]["confidence"]
        active_learning_flag = top1_conf < ACTIVE_LEARNING_THRESHOLD

        # MLflow: log inference metadata
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name="predict"):
                    mlflow.log_param("model_type", effective_type)
                    mlflow.log_param("top_k", TOP_K)
                    mlflow.log_param("temperature", temperature)
                    mlflow.log_metric("top1_confidence", top1_conf)
                    mlflow.log_metric("active_learning_flag", int(active_learning_flag))
                    mlflow.set_tag("endpoint", "predict")
                    mlflow.set_tag("top1_class", top_predictions[0]["class_name"])
            except Exception:
                pass  # MLflow errors never break inference

        return {
            "predictions": top_predictions,
            "calibrated": temperature != 1.0,
            "temperature": temperature,
            "active_learning_flag": active_learning_flag,
            "low_confidence": active_learning_flag,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# v1: Caption endpoint
# ----------------------------
@router.post("/caption")
async def caption(file: UploadFile = File(...)):
    if not ENABLE_BLIP:
        return JSONResponse(
            status_code=503,
            content={"error": "Image captioning is disabled (ENABLE_BLIP=false)."}
        )
    err = _check_models_ready()
    if err:
        return err
    if blip_model is None or blip_processor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "BLIP model failed to load. Check server logs."}
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
# v1: Grad-CAM endpoint
# ----------------------------
@router.post("/gradcam")
async def gradcam(file: UploadFile = File(...), top_k: int = TOP_K, model_type: Optional[str] = Form(None)):
    err = _check_models_ready()
    if err:
        return err

    try:
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)

        # Resolve which model / config to use for this request
        requested = model_type or MODEL_TYPE
        m = _models.get(requested) or model
        config = _model_configs.get(requested) or _model_configs.get(MODEL_TYPE, {})
        label_map = config.get("label_map", LABEL_MAP)
        effective_type = requested if requested in _models else MODEL_TYPE
        conv_layer_name = config.get("last_conv", last_conv_layer_name)

        x = preprocess_image(img_resized, effective_type)

        if not all(hasattr(m, a) for a in ['get_layer', 'inputs', 'output']):
            raise AttributeError("Model missing attributes required for Grad-CAM")

        # Build grad model, handling nested layers (MobileNetV2 inside custom model)
        try:
            last_conv_layer = m.get_layer(conv_layer_name)
            grad_model = Model(inputs=m.inputs, outputs=[last_conv_layer.output, m.output])
        except ValueError:
            parent_model = None
            parent_idx = None
            for i, layer in enumerate(m.layers):
                if hasattr(layer, 'layers'):
                    try:
                        _ = layer.get_layer(conv_layer_name)
                        parent_model = layer
                        parent_idx = i
                        break
                    except ValueError:
                        continue

            if parent_model is None:
                raise ValueError(f"Could not find layer '{conv_layer_name}'")

            nested_conv_layer = parent_model.get_layer(conv_layer_name)
            parent_with_conv = Model(
                inputs=parent_model.input,
                outputs=[nested_conv_layer.output, parent_model.output]
            )
            x_in = m.inputs[0]
            conv_output, parent_output = parent_with_conv(x_in)
            final_output = parent_output
            for layer in m.layers[parent_idx + 1:]:
                final_output = layer(final_output)
            grad_model = Model(inputs=m.inputs, outputs=[conv_output, final_output])

        preds = m.predict(x)[0]
        top_indices = preds.argsort()[-top_k:][::-1]
        gradcam_results = []

        for idx in top_indices:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x)
                loss = predictions[:, idx]

            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                continue

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_out = conv_outputs[0]
            heatmap = conv_out @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            heatmap = heatmap.numpy()

            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            heatmap_colored = np.uint8(255 * heatmap_colored)
            heatmap_img = Image.fromarray(heatmap_colored).resize(IMG_SIZE)
            overlay = Image.blend(img_resized, heatmap_img, alpha=0.4)
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            gradcam_results.append({
                "class_idx": int(idx),
                "class_name": label_map.get(str(idx), f"Class {idx}"),
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
# v1: SHAP endpoint
# ----------------------------
@router.post("/shap")
async def shap_explain(file: UploadFile = File(...), model_type: Optional[str] = Form(None)):
    if not ENABLE_SHAP:
        return JSONResponse(
            status_code=503,
            content={"error": "SHAP is disabled (ENABLE_SHAP=false)."}
        )
    if not SHAP_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "SHAP package not installed. Add 'shap' to requirements.txt."}
        )
    err = _check_models_ready()
    if err:
        return err
    if shap_explainer is None:
        return JSONResponse(
            status_code=503,
            content={"error": "SHAP explainer not initialized. Check server logs."}
        )

    try:
        img = Image.open(file.file).convert("RGB")
        x = preprocess_image(img, MODEL_TYPE)

        # Compute SHAP values for top predicted class only.
        # ranked_outputs=1 limits computation to the top-1 class (vs all 1000),
        # which is ~1000x faster on CPU.
        shap_values, indices = shap_explainer.shap_values(x, ranked_outputs=1)

        raw_preds = model(x, training=False).numpy()[0]
        top_class_idx = int(indices[0][0])

        # shap_values is a list with one entry (ranked_outputs=1)
        shap_img = shap_values[0][0]  # [H, W, C]

        # Aggregate across channels and normalize
        shap_agg = np.abs(shap_img).mean(axis=-1)  # [H, W]
        shap_norm = (shap_agg - shap_agg.min()) / (shap_agg.max() - shap_agg.min() + 1e-8)

        # RdBu_r colormap: red = important regions, blue = suppressing regions
        heatmap_colored = cm.RdBu_r(1 - shap_norm)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        heatmap_img = Image.fromarray(heatmap_colored).resize(IMG_SIZE)
        original = img.resize(IMG_SIZE)
        overlay = Image.blend(original, heatmap_img, alpha=0.5)

        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        shap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        top_class_name = LABEL_MAP.get(str(top_class_idx), f"Class {top_class_idx}")
        top_confidence = float(apply_temperature_scaling(raw_preds, TEMPERATURE)[top_class_idx])

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name="shap"):
                    mlflow.log_param("model_type", MODEL_TYPE)
                    mlflow.log_metric("top1_confidence", top_confidence)
                    mlflow.set_tag("endpoint", "shap")
                    mlflow.set_tag("top1_class", top_class_name)
            except Exception:
                pass

        return {
            "shap_plot_base64": f"data:image/png;base64,{shap_b64}",
            "top_class": top_class_name,
            "top_class_idx": top_class_idx,
            "top_confidence": top_confidence,
            "explanation": "Red regions contribute most to this prediction. Blue regions suppress it.",
            "method": "SHAP GradientExplainer",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# v1: CLIP zero-shot endpoint
# ----------------------------
@router.post("/clip")
async def clip_classify(
    file: UploadFile = File(...),
    labels: str = Form(...)  # JSON array: '["cat", "dog", "car"]'
):
    if not ENABLE_CLIP:
        return JSONResponse(
            status_code=503,
            content={"error": "CLIP is disabled (ENABLE_CLIP=false)."}
        )
    if clip_model is None or clip_processor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "CLIP model not loaded. Check ENABLE_CLIP and server logs."}
        )

    try:
        label_list = json.loads(labels)
        if not isinstance(label_list, list) or not label_list:
            return JSONResponse(
                status_code=400,
                content={"error": "labels must be a non-empty JSON array of strings."}
            )
        if len(label_list) > 20:
            return JSONResponse(
                status_code=400,
                content={"error": "Maximum 20 labels allowed per request."}
            )

        img = Image.open(file.file).convert("RGB")

        clip_device = next(clip_model.parameters()).device
        inputs = clip_processor(
            text=label_list,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(clip_device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # [1, num_labels]
            probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()

        results = sorted(
            [{"label": label, "score": float(score)} for label, score in zip(label_list, probs)],
            key=lambda x: x["score"],
            reverse=True
        )

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name="clip"):
                    mlflow.log_param("num_labels", len(label_list))
                    mlflow.log_metric("top1_score", results[0]["score"])
                    mlflow.set_tag("endpoint", "clip")
                    mlflow.set_tag("top1_label", results[0]["label"])
            except Exception:
                pass

        return {
            "results": results,
            "model": "CLIP ViT-B/32",
            "zero_shot": True,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# v1: Feedback endpoint (with active learning fields)
# ----------------------------
@router.post("/feedback")
async def feedback(file: UploadFile = File(...), entry: str = Form(...)):
    try:
        img_path = FEEDBACK_DIR / file.filename
        with open(img_path, "wb") as f:
            f.write(await file.read())

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
            "active_learning_flag": entry_data.get("active_learning_flag", False),
            "flag_reason": entry_data.get("flag_reason", "user_submitted"),
            "top1_confidence": entry_data.get("top1_confidence"),
            "predicted_class": entry_data.get("predicted_class"),
        })

        with open(feedback_log_path, "w") as f:
            json.dump(log, f, indent=2)

        return {"status": "success"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# v1: Active learning summary
# ----------------------------
@router.get("/active-learning/summary")
async def active_learning_summary():
    try:
        feedback_log_path = FEEDBACK_DIR / "feedback_log.json"
        if not feedback_log_path.exists():
            return {"total": 0, "flagged": 0, "flag_rate": 0.0, "flagged_samples": []}

        with open(feedback_log_path, "r") as f:
            log = json.load(f)

        flagged = [e for e in log if e.get("active_learning_flag")]
        return {
            "total": len(log),
            "flagged": len(flagged),
            "flag_rate": round(len(flagged) / max(len(log), 1), 3),
            "flagged_samples": flagged[-10:],  # Last 10 flagged entries
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# v1: MLflow recent runs
# ----------------------------
@router.get("/mlflow/recent-runs")
async def get_recent_runs(limit: int = 10):
    if not MLFLOW_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "MLflow not installed. Add 'mlflow' to requirements.txt."}
        )
    try:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            order_by=["start_time DESC"],
            max_results=limit,
            output_format="list"
        )
        formatted = []
        for run in runs:
            formatted.append({
                "run_id": run.info.run_id[:8],
                "endpoint": run.data.tags.get("endpoint", "unknown"),
                "top1_class": (
                    run.data.tags.get("top1_class") or
                    run.data.tags.get("top1_label", "")
                ),
                "top1_confidence": (
                    run.data.metrics.get("top1_confidence") or
                    run.data.metrics.get("top1_score")
                ),
                "model_type": run.data.params.get("model_type", "imagenet"),
                "started_ms": run.info.start_time,
            })
        return {"runs": formatted, "total": len(formatted)}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )

# ----------------------------
# Register versioned router
# ----------------------------
app.include_router(router)

# ----------------------------
# Legacy unversioned routes (backward compatibility)
# ----------------------------
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    return await predict(file)

@app.post("/caption")
async def caption_legacy(file: UploadFile = File(...)):
    return await caption(file)

@app.post("/gradcam")
async def gradcam_legacy(file: UploadFile = File(...), top_k: int = TOP_K):
    return await gradcam(file, top_k)

@app.post("/feedback")
async def feedback_legacy(file: UploadFile = File(...), entry: str = Form(...)):
    return await feedback(file, entry)

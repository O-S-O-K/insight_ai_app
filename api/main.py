import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.gradcam import find_last_conv_layer  # assumes you have this function

# Paths
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_model.h5"

# FastAPI app
app = FastAPI(title="Insight AI API")

# Load model once
model = load_model(MODEL_PATH)
last_conv_layer_name = find_last_conv_layer(model)

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        img_array = np.expand_dims(np.array(image.resize((224, 224))), axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds[0]))

        return {"class": pred_class}

    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"error": str(e), "trace": traceback_str}

# ----------------------------
# Caption endpoint (placeholder)
# ----------------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        # Placeholder caption
        return JSONResponse(content={"caption": "This is a mock caption for testing."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# Grad-CAM endpoint
# ----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        # Load image
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize((224, 224))
        x = np.expand_dims(np.array(img_resized), axis=0)
        x = preprocess_input(x)

        # Grad-CAM
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = np.uint8(255 * heatmap.numpy())

        # Encode heatmap as base64 so frontend can display
        import io, base64
        heatmap_img = Image.fromarray(heatmap).resize(img_resized.size)
        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"gradcam_layer": last_conv_layer_name, "heatmap_base64": heatmap_b64}

    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"error": str(e), "trace": traceback_str}

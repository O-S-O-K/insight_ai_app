import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import base64

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.gradcam import find_last_conv_layer, get_gradcam_heatmap, overlay_heatmap
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        x = np.array(image)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)
        decoded = decode_predictions(preds, top=3)[0]

        results = [{"label": label, "score": float(score)} for (_, label, score) in decoded]

        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# Caption endpoint (dummy BLIP / placeholder)
# ----------------------------
@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        # For now, return a placeholder caption
        return JSONResponse(content={"caption": "This is a mock caption for testing."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------------------
# Grad-CAM endpoint
# ----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...), class_idx: int = None):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_resized = image.resize((224, 224))

        x = np.array(image_resized)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        if class_idx is None:
            class_idx = int(np.argmax(model.predict(x)[0]))

        heatmap = get_gradcam_heatmap(model, x, last_conv_layer_name, class_idx)
        overlay = overlay_heatmap(heatmap, image, alpha=0.4)

        # Return overlay as base64 string
        buffered = io.BytesIO()
        overlay.save(buffered, format="PNG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={"overlay_base64": overlay_b64})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
~
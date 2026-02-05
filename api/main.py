import sys
from pathlib import Path

# -------------------------------------------------
# Ensure repo root is on PYTHONPATH (Docker + Render)
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # /app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# Standard imports
# -------------------------------------------------
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io

# -------------------------------------------------
# Project imports (now work correctly)
# -------------------------------------------------
from utils.gradcam import (
    find_last_conv_layer,
    get_gradcam_heatmap,
    overlay_heatmap,
)

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_model.h5"

# -------------------------------------------------
# App init
# -------------------------------------------------
app = FastAPI(title="Insight AI API")

# -------------------------------------------------
# Load model ONCE at startup
# -------------------------------------------------
model = load_model(MODEL_PATH)
last_conv_layer_name = find_last_conv_layer(model)

# -------------------------------------------------
# Health check (Render uses this implicitly)
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # Grad-CAM
        heatmap = get_gradcam_heatmap(
            model,
            img_array,
            last_conv_layer_name,
            predicted_class,
        )

        return JSONResponse(
            content={
                "class": predicted_class,
                "confidence": confidence,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

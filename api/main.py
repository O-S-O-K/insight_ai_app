import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import base64
import traceback
import tensorflow as tf

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
        from PIL import Image
        import numpy as np
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        img_array = preprocess_input(np.expand_dims(image.resize((224,224)), axis=0))

        # Predict
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds[0]))

        return {"class": pred_class}

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        # Return the error info instead of 500
        return {"error": str(e), "trace": traceback_str}

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
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import tensorflow as tf
import traceback

app = FastAPI()

# Assume 'model' is already loaded somewhere globally:
# model = load_model("models/cnn_model.h5")

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    try:
        # Load image
        img = Image.open(file.file).convert("RGB")
        img_resized = img.resize((224, 224))
        x = np.expand_dims(np.array(img_resized), axis=0)
        x = preprocess_input(x)

        # Automatically pick the last conv layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in the model for Grad-CAM.")

        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(
            [model.inputs], [last_conv_layer.output, model.output]
        )

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

        return {"gradcam_layer": last_conv_layer_name, "heatmap_shape": heatmap.shape}

    except Exception as e:
        # Return detailed error info instead of 500
        traceback_str = traceback.format_exc()
        return {"error": str(e), "trace": traceback_str}


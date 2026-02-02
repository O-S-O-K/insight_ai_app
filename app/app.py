# app/app.py
import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import hashlib
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.gradcam import (
    predict_image,
    find_last_conv_layer,
    get_gradcam_heatmap,
    overlay_heatmap,
)

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = ROOT / "models/cnn_model_finetuned.h5"
BASE_MODEL_PATH = ROOT / "models/cnn_model.h5"
LABELS_PATH = ROOT / "models/label_map.json"

FEEDBACK_CSV = ROOT / "feedback_log.csv"
IMG_DIR = ROOT / "feedback_images"
IMG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="InsightAI", layout="wide")

# -----------------------------
# Session safety
# -----------------------------
for k in ["last_hash", "submitted", "feedback"]:
    st.session_state.setdefault(k, None)

# -----------------------------
# Load model + labels
# -----------------------------
@st.cache_resource
def load_assets():
    model_path = MODEL_PATH if MODEL_PATH.exists() else BASE_MODEL_PATH
    model = load_model(model_path)

    if not LABELS_PATH.exists():
        st.error("label_map.json missing — retrain at least once.")
        st.stop()

    label_map = json.loads(LABELS_PATH.read_text())
    inv_label_map = {int(k): v for k, v in label_map.items()}

    return model, inv_label_map

model, label_map = load_assets()

# -----------------------------
# UI
# -----------------------------
st.title("🧠 InsightAI")
st.subheader("Explainable Image Classification with Human Feedback")

uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

def file_hash(f):
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h

if uploaded:
    h = file_hash(uploaded)

    if st.session_state.last_hash != h:
        st.session_state.last_hash = h
        st.session_state.submitted = False
        st.session_state.feedback = None

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", width="content")

    # -----------------------------
    # Prediction
    # -----------------------------
    preds = predict_image(model, img, top=3, label_map=label_map)

    st.subheader("🔍 Predictions")
    for i, (lbl, score) in enumerate(preds, 1):
        st.write(f"{i}. **{lbl}** — {score*100:.2f}%")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    img_resized = img.resize(IMG_SIZE)
    tensor = preprocess_input(np.expand_dims(np.array(img_resized), 0))

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(
        model,
        last_conv,
        tensor,
        class_idx=list(label_map.keys())[0],
    )

    cam = overlay_heatmap(heatmap, img)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", width="content")
    c2.image(cam, caption="Grad-CAM", width="content")

    # -----------------------------
    # Feedback
    # -----------------------------
    st.subheader("🧠 Feedback")

    if not st.session_state.submitted:
        correct = st.radio("Was the top prediction correct?", ["Yes", "No"])

        if correct == "Yes":
            label = preds[0][0]
        else:
            label = st.text_input("Enter correct label")

        if st.button("Submit feedback"):
            if not label:
                st.warning("Label required.")
            else:
                path = IMG_DIR / uploaded.name
                img.save(path)

                entry = {
                    "uploaded_filename": path.name,
                    "model_prediction": preds[0][0],
                    "user_label": label,
                    "was_correct": correct,
                }

                df = pd.read_csv(FEEDBACK_CSV) if FEEDBACK_CSV.exists() else pd.DataFrame()
                df = pd.concat([df, pd.DataFrame([entry])])
                df.to_csv(FEEDBACK_CSV, index=False)

                st.session_state.submitted = True
                st.success("Feedback saved.")

    else:
        st.info("Feedback already submitted.")

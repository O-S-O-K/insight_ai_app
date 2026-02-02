# ======================================================
# InsightAI — Explainable Image Classification App
# ======================================================

import os
import sys
import json
import hashlib
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FINETUNED_MODEL_PATH = MODELS_DIR / "cnn_model_finetuned.h5"

FEEDBACK_CSV = ROOT / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

# ======================================================
# IMPORT GRADCAM UTILS
# ======================================================
from utils.gradcam import (
    find_last_conv_layer,
    get_gradcam_heatmap,
    overlay_heatmap,
)

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="InsightAI", layout="wide")

# ======================================================
# SESSION STATE SAFETY
# ======================================================
for k in ["last_image_hash", "feedback_submitted", "feedback"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ======================================================
# HELPERS
# ======================================================
def file_hash(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


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

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_cnn_model():
    """
    Priority:
    1. Fine-tuned model (if exists)
    2. Pretrained MobileNetV2 (ImageNet)
    """
    if FINETUNED_MODEL_PATH.exists():
        model = load_model(FINETUNED_MODEL_PATH, compile=False)
        source = "Fine-tuned model"
    else:
        model = MobileNetV2(weights="imagenet", include_top=True)
        source = "ImageNet pretrained"
    return model, source


model, model_source = load_cnn_model()
meta = load_model_metadata()

# ======================================================
# HEADER
# ======================================================
st.title("🧠 InsightAI")
st.subheader("Explainable Image Classification with Human Feedback")

with st.expander("ℹ️ Model Information", expanded=True):
    st.markdown(
        f"""
**Model source:** {model_source}  
**Model name:** `{meta['model_name']}`  
**Version:** `{meta['version']}`  
**Architecture:** {meta['architecture']}  
**Training data:** {meta['trained_on']}  
**Last updated:** {meta['last_updated']}
"""
    )

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_hash = file_hash(uploaded_file)

    # Reset state on new image
    if st.session_state.last_image_hash != img_hash:
        st.session_state.last_image_hash = img_hash
        st.session_state.feedback_submitted = False
        st.session_state.feedback = None

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width="content")

    # ======================================================
    # PREDICTIONS
    # ======================================================
    img_resized = img.resize((224, 224))
    x = preprocess_input(np.expand_dims(np.array(img_resized), 0))

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    st.subheader("🔍 Top Predictions")
    for i, (_, label, score) in enumerate(decoded, 1):
        st.write(f"{i}. **{label}** — {score * 100:.2f}%")

    # ======================================================
    # GRAD-CAM
    # ======================================================
    st.subheader("🔥 Grad-CAM Explanation")
    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(
        model,
        last_conv,
        x,
        pred_index=np.argmax(preds[0]),  # optional; can remove to default top prediction
    )

    alpha = st.slider("Heatmap intensity", 0.2, 0.7, 0.4, 0.05)
    cam_img = overlay_heatmap(heatmap, img, alpha)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", width="content")
    c2.image(cam_img, caption="Grad-CAM", width="content")

    # ======================================================
    # FEEDBACK
    # ======================================================
    st.subheader("🧠 Human Feedback")

    if not st.session_state.feedback_submitted:
        correct = st.radio(
            "Was the model’s top prediction correct?",
            ["Yes", "No"],
            horizontal=True,
        )

        if correct == "Yes":
            user_label = decoded[0][1]
        else:
            user_label = st.text_input("What should the correct label be?")

        if st.button("Submit Feedback"):
            if not user_label:
                st.warning("Please provide a label.")
            else:
                img_save_path = FEEDBACK_IMG_DIR / f"{img_hash}.jpg"
                img.save(img_save_path)

                entry = {
                    "image_hash": img_hash,
                    "image_path": str(img_save_path),
                    "model_prediction": decoded[0][1],
                    "user_label": user_label,
                    "was_correct": correct,
                }

                df = pd.read_csv(FEEDBACK_CSV) if FEEDBACK_CSV.exists() else pd.DataFrame()
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
                df.to_csv(FEEDBACK_CSV, index=False)

                st.session_state.feedback = entry
                st.session_state.feedback_submitted = True
                st.success("Feedback recorded. Thank you!")

    else:
        st.info("Feedback already submitted for this image.")

    if st.session_state.feedback:
        with st.expander("View recorded feedback"):
            st.json(st.session_state.feedback)

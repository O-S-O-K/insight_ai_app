# ======================================================
# InsightAI - Explainable Image Classification App
# Fully compatible with custom CNN
# ======================================================

import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import hashlib

# -----------------------------
# Path safety for utils package
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.gradcam import (
    load_cnn_model,
    predict_image,
    find_last_conv_layer,
    get_gradcam_heatmap,
    overlay_heatmap,
)
from utils.blip_caption import generate_blip_caption

# ======================================================
# CONFIG / FEATURE FLAGS
# ======================================================
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"
ENABLE_BLIP = not IS_CLOUD

FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="InsightAI", layout="wide")

# ======================================================
# SESSION STATE
# ======================================================
SESSION_DEFAULTS = {
    "last_image_hash": None,
    "feedback_submitted": False,
    "feedback": None,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ======================================================
# HELPERS
# ======================================================
def get_file_hash(uploaded_file):
    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(data).hexdigest()

def load_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return img

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model_cached():
    return load_cnn_model()

model = load_model_cached()

# ======================================================
# HEADER
# ======================================================
st.title("🧠 InsightAI")
st.subheader("Explainable Image Classification with Human Feedback")
st.markdown(
    """
Upload an image to see:
- **Top-3 CNN predictions**
- **Grad-CAM explanations**
- **Optional human feedback**
"""
)

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_hash = get_file_hash(uploaded_file)

    # Reset state for new image
    if st.session_state.last_image_hash != img_hash:
        st.session_state.last_image_hash = img_hash
        st.session_state.feedback_submitted = False
        st.session_state.feedback = None

        # Clear previous feedback widgets
        for k in list(st.session_state.keys()):
            if k.startswith("feedback_"):
                del st.session_state[k]

    img = load_image(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Predictions
    # -----------------------------
    @st.cache_resource
    def predict_cached(model, img_hash, img):
        return predict_image(model, img, top=3)

    preds = predict_cached(model, img_hash, img)

    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, 1):
        st.write(f"{i}. **{label}** — {score*100:.2f}%")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    st.subheader("🔥 Grad-CAM Explanation")
    img_resized = img.resize((224, 224))
    img_tensor = np.expand_dims(np.array(img_resized)/255.0, 0)

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(model, last_conv, img_tensor)

    alpha = st.slider("Heatmap intensity", 0.2, 0.7, 0.4, 0.05)
    cam_img = overlay_heatmap(heatmap, img, alpha)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", use_column_width=True)
    c2.image(cam_img, caption="Grad-CAM", use_column_width=True)

    # -----------------------------
    # Feedback
    # -----------------------------
    st.subheader("🧠 Feedback")

    if not st.session_state.feedback_submitted:
        base = f"feedback_{img_hash}"

        correct = st.radio(
            "Was the model’s top prediction correct?",
            ["Yes", "No"],
            horizontal=True,
            key=f"{base}_correct",
        )

        user_label = None
        if correct == "Yes":
            user_label = preds[0][0]
        else:
            options = [label for label, _ in preds[1:]] + ["Other"]
            choice = st.radio("Select correct label", options, key=f"{base}_choice")
            if choice == "Other":
                user_label = st.text_input("Enter label", key=f"{base}_text")
            else:
                user_label = choice

        if st.button("Submit Feedback", key=f"{base}_submit"):
            if not user_label or not user_label.strip():
                st.warning("Please provide a valid label.")
            else:
                entry = {
                    "uploaded_filename": uploaded_file.name,
                    "model_prediction": preds[0][0],
                    "user_label": user_label,
                    "was_correct": correct,
                }
                if FEEDBACK_CSV.exists():
                    df = pd.concat([pd.read_csv(FEEDBACK_CSV), pd.DataFrame([entry])], ignore_index=True)
                else:
                    df = pd.DataFrame([entry])
                df.to_csv(FEEDBACK_CSV, index=False)

                st.session_state.feedback = entry
                st.session_state.feedback_submitted = True
                st.success("Thanks! Feedback recorded.")
    else:
        st.info("Feedback already submitted for this image.")

    # -----------------------------
    # Show feedback
    # -----------------------------
    if st.session_state.feedback_submitted and st.session_state.feedback is not None:
        with st.expander("View recorded feedback"):
            st.json(st.session_state.feedback)

    # -----------------------------
    # BLIP caption
    # -----------------------------
    st.subheader("📝 Image Caption")
    if ENABLE_BLIP:
        @st.cache_resource
        def blip_cached(img_hash, img):
            return generate_blip_caption(img)
        caption = blip_cached(img_hash, img)
        st.write(caption if caption else "Caption generation disabled in cloud demo.")
    else:
        st.info("Image captioning disabled in cloud demo.")

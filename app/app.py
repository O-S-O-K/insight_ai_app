# ======================================================
# InsightAI - Explainable Image Classification App
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ======================================================
# CONFIG / FEATURE FLAGS
# ======================================================
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"
ENABLE_BLIP = not IS_CLOUD  # BLIP disabled in cloud deployments

# Feedback paths
FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="InsightAI", layout="wide")

# ======================================================
# ABSOLUTE SESSION STATE SAFETY (CLOUD-SAFE)
# ======================================================
# These MUST be initialized unconditionally on every run
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

if "feedback" not in st.session_state:
    st.session_state.feedback = None

# ======================================================
# 🔐 SESSION STATE INITIALIZATION (GLOBAL & CRASH‑PROOF)
# ======================================================
DEFAULT_SESSION_STATE = {
    "last_image_hash": None,
    "feedback_submitted": False,
    "feedback": None,
}

for key, default in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ======================================================
# HELPERS
# ======================================================
def get_file_hash(uploaded_file):
    """Generate MD5 hash for uploaded file bytes"""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_bytes).hexdigest()

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    return load_cnn_model()

model = load_model()

# ======================================================
# HEADER
# ======================================================
st.title("🧠 InsightAI")
st.subheader("Explainable Image Classification with Human Feedback")
st.markdown(
    """
Upload an image to see:
- **Top-3 CNN predictions**
- **Grad-CAM visual explanations**
- **Optional human feedback**
- **BLIP-generated captions (local only)**
"""
)

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # -----------------------------
    # Compute image hash
    # -----------------------------
    img_hash = get_file_hash(uploaded_file)

    # -----------------------------
    # Reset state when image changes
    # -----------------------------
    if st.session_state.last_image_hash != img_hash:
        st.session_state.last_image_hash = img_hash
        st.session_state.feedback_submitted = False
        st.session_state.feedback = None

        # Remove ALL widget keys tied to previous image
        for k in list(st.session_state.keys()):
            if k.startswith("feedback_"):
                del st.session_state[k]

    # -----------------------------
    # Load and display image
    # -----------------------------
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Top-3 predictions (hash‑scoped cache)
    # -----------------------------
    @st.cache_resource
    def predict_image_cached(model, img_hash, img):
        return predict_image(model, img, top=3)

    preds = predict_image_cached(model, img_hash, img)

    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, start=1):
        st.write(f"{i}. **{label}** — {score * 100:.2f}%")

    # -----------------------------
    # Grad-CAM overlay
    # -----------------------------
    st.subheader("🔥 Grad-CAM Explanation")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = preprocess_input(img_tensor)

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(model, last_conv, img_tensor)

    alpha = st.slider(
        "Heatmap intensity", min_value=0.2, max_value=0.7, value=0.4, step=0.05
    )
    cam_img = overlay_heatmap(heatmap, img, alpha=alpha)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(cam_img, caption="Grad-CAM Overlay", use_column_width=True)

    st.markdown(
        """
**What this shows:**  
Highlighted regions indicate which parts of the image most influenced the model’s prediction.
"""
    )

    # -----------------------------
    # Feedback (state‑safe & image‑scoped)
    # -----------------------------
    st.subheader("🧠 Feedback")

    if not st.session_state.get("feedback_submitted", False):
        base_key = f"feedback_{img_hash}"

        correct = st.radio(
            "Was the model’s top prediction correct?",
            ["Yes", "No"],
            horizontal=True,
            key=f"{base_key}_correct",
        )

        user_label = None

        if correct == "Yes":
            user_label = preds[0][0]
        else:
            st.markdown("### What is the correct label?")
            top_labels = [label for label, _ in preds]
            options = top_labels[1:] + ["Other"]

            selection = st.radio(
                "Select one of the alternatives or choose *Other*",
                options,
                key=f"{base_key}_selection",
            )

            if selection == "Other":
                user_label = st.text_input(
                    "Enter the correct label",
                    key=f"{base_key}_text",
                )
            else:
                user_label = selection

        if st.button("Submit Feedback", key=f"{base_key}_submit"):
            if not user_label or not user_label.strip():
                st.warning("Please provide a valid label.")
            else:
                # Save image
                img_filename = uploaded_file.name
                try:
                    img.save(FEEDBACK_IMG_DIR / img_filename)
                except Exception as e:
                    st.warning(f"Could not save image: {e}")

                feedback_entry = {
                    "uploaded_filename": img_filename,
                    "model_prediction": preds[0][0],
                    "user_label": user_label,
                    "was_correct": correct,
                }

                if FEEDBACK_CSV.exists():
                    df_existing = pd.read_csv(FEEDBACK_CSV)
                    df = pd.concat(
                        [df_existing, pd.DataFrame([feedback_entry])],
                        ignore_index=True,
                    )
                else:
                    df = pd.DataFrame([feedback_entry])

                df.to_csv(FEEDBACK_CSV, index=False)

                st.session_state.feedback = feedback_entry
                st.session_state.feedback_submitted = True
                st.success("Thanks! Your feedback has been recorded.")

    else:
        st.info("Feedback already submitted for this image.")

    if st.session_state.feedback_submitted and st.session_state.feedback:
        with st.expander("View recorded feedback"):
            st.json(st.session_state.feedback)

    # -----------------------------
    # BLIP Caption (hash‑scoped cache)
    # -----------------------------
    st.subheader("📝 Image Caption")

    if ENABLE_BLIP:
        @st.cache_resource
        def generate_blip_caption_cached(img_hash, img):
            return generate_blip_caption(img)

        caption = generate_blip_caption_cached(img_hash, img)

        if caption:
            st.write(caption)
        else:
            st.warning("Caption generation failed.")
    else:
        st.info(
            "Image captioning is disabled in the public demo due to cloud resource limits. "
            "Available in local deployment."
        )

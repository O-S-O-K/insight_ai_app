# ======================================================
# InsightAI - Explainable Image Classification App
# (Streamlit Cloud–safe session state)
# ======================================================

import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image, ExifTags
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
ENABLE_BLIP = not IS_CLOUD

FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="InsightAI", layout="wide")

# ======================================================
# 🔒 ABSOLUTE SESSION STATE INITIALIZATION
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

def load_image_with_exif(uploaded_file):
    """Load image and apply EXIF rotation (mobile-safe)."""
    img = Image.open(uploaded_file)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img.convert("RGB")

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

    # Reset state on new image
    if st.session_state.last_image_hash != img_hash:
        st.session_state.last_image_hash = img_hash
        st.session_state.feedback_submitted = False
        st.session_state.feedback = None

        for k in list(st.session_state.keys()):
            if k.startswith("feedback_"):
                del st.session_state[k]

    # Load mobile-safe image
    img = load_image_with_exif(uploaded_file)
    st.image(img, caption="Uploaded Image", width="content")

    # -----------------------------
    # Predictions
    # -----------------------------
    @st.cache_resource
    def predict_cached(model, img_hash, img):
        return predict_image(model, img, top=3)

    preds = predict_cached(model, img_hash, img)

    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, 1):
        st.write(f"{i}. **{label}** — {score * 100:.2f}%")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    st.subheader("🔥 Grad-CAM Explanation")
    img_resized = img.resize((224, 224))
    img_tensor = preprocess_input(np.expand_dims(np.array(img_resized), 0))

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(model, last_conv, img_tensor)

    alpha = st.slider("Heatmap intensity", 0.2, 0.7, 0.4, 0.05)
    cam_img = overlay_heatmap(heatmap, img, alpha)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original", width="content")
    c2.image(cam_img, caption="Grad-CAM", width="content")

    # -----------------------------
    # Feedback
    # -----------------------------
    st.subheader("🧠 Feedback")

    if not st.session_state.get("feedback_submitted", False):
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
                # -----------------------------
                # Save uploaded image for retraining
                # -----------------------------
                FEEDBACK_IMG_DIR.mkdir(exist_ok=True)
                uploaded_file.seek(0)
                feedback_img = load_image_with_exif(uploaded_file)

                feedback_img_path = FEEDBACK_IMG_DIR / uploaded_file.name
                if feedback_img_path.exists():
                    name, ext = uploaded_file.name.rsplit(".", 1)
                    feedback_img_path = FEEDBACK_IMG_DIR / f"{name}_{img_hash[:6]}.{ext}"

                feedback_img.save(feedback_img_path)

                # -----------------------------
                # Save feedback CSV
                # -----------------------------
                entry = {
                    "uploaded_filename": feedback_img_path.name,
                    "model_prediction": preds[0][0],
                    "user_label": user_label,
                    "was_correct": correct,
                }

                if FEEDBACK_CSV.exists():
                    df = pd.concat([
                        pd.read_csv(FEEDBACK_CSV),
                        pd.DataFrame([entry])
                    ], ignore_index=True)
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
    if st.session_state.get("feedback_submitted", False) and st.session_state.get("feedback"):
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

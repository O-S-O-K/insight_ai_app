# app/app.py
import os
import json
import hashlib
from pathlib import Path
import io
import base64

import streamlit as st
from PIL import Image

# ----------------------------
# Backend configuration
# ----------------------------
os.environ["INSIGHT_BACKEND_URL"] = "http://localhost:8000"
USE_MOCK = os.environ.get("USE_MOCK_API", "false").lower() == "true"

if USE_MOCK:
    from utils.mock_api_client import *
else:
    from utils.api_client import (
        predict_image as call_backend_predict,
        caption_image as call_backend_caption,
        gradcam_image as call_backend_gradcam,
        submit_feedback as post_feedback_to_backend,
    )

st.write(f"Using {'mock' if USE_MOCK else 'live'} API client")

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Insight AI", layout="centered")
ROOT = Path(__file__).resolve().parent

# ----------------------------
# Load metadata for labels
# ----------------------------
METADATA_PATH = ROOT.parent / "models" / "model_metadata.json"
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    LABEL_MAP = metadata.get("classes", {})
else:
    LABEL_MAP = {}

# ----------------------------
# Helpers
# ----------------------------
def image_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h

def reset_state_on_new_image(new_hash: str):
    if st.session_state.get("image_hash") != new_hash:
        st.session_state.image_hash = new_hash
        st.session_state.feedback_submitted = False
        st.session_state.prediction = None
        st.session_state.caption = None
        st.session_state.gradcam = None

def display_gradcam_image(overlay_img: Image.Image, alpha: float, original_img: Image.Image):
    """Blend overlay with original using alpha and show in Streamlit"""
    blended = Image.blend(original_img, overlay_img, alpha)
    st.image(blended, caption="Grad-CAM Overlay", width="stretch")

# ----------------------------
# UI
# ----------------------------
st.title("Insight AI")
st.caption("Explainable image classification with BLIP captions and Grad-CAM visualizations")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", width="stretch")

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    col1, col2, col3 = st.columns(3)

    # ------------------------
    # Predict
    # ------------------------
    with col1:
        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                try:
                    result = call_backend_predict(uploaded_file)
                    # Map class_idx to human-readable name if metadata exists
                    class_idx = result.get("class_idx")
                    result["class_name"] = LABEL_MAP.get(str(class_idx), f"Class {class_idx}")
                    st.session_state.prediction = result
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("prediction"):
        pred = st.session_state.prediction
        st.subheader("Prediction")
        st.write(f"Class: **{pred['class_name']}** (Index: {pred['class_idx']})")
        st.write(f"Confidence: **{pred['confidence']*100:.2f}%**")

    # ------------------------
    # Caption
    # ------------------------
    with col2:
        if st.button("Caption"):
            with st.spinner("Generating caption..."):
                try:
                    result = call_backend_caption(uploaded_file)
                    st.session_state.caption = result
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("caption"):
        st.subheader("Image Caption (BLIP)")
        st.write(st.session_state.caption.get("caption"))

    # ------------------------
    # Grad-CAM
    # ------------------------
    with col3:
        alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.4, 0.05)
        if st.button("Grad-CAM"):
            with st.spinner("Computing Grad-CAM..."):
                try:
                    class_idx = st.session_state.get("prediction", {}).get("class_idx")
                    result = call_backend_gradcam(uploaded_file, class_idx=class_idx)
                    # Map Grad-CAM class to human-readable label
                    pred_idx = result.get("pred_class_idx")
                    result["pred_class_name"] = LABEL_MAP.get(str(pred_idx), f"Class {pred_idx}")
                    st.session_state.gradcam = result
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("gradcam"):
        st.subheader("Grad-CAM Output")
        gradcam_data = st.session_state.gradcam
        st.write(f"Class: **{gradcam_data['pred_class_name']}** (Index: {gradcam_data['pred_class_idx']})")
        # Decode base64 overlay
        b64_data = gradcam_data["heatmap_base64"].split(",")[1]
        overlay_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        display_gradcam_image(overlay_img, alpha, img)

    # ------------------------
    # Human Feedback
    # ------------------------
    st.divider()
    st.subheader("Human Feedback")

    if not st.session_state.get("feedback_submitted", False):
        feedback_text = st.text_area("Your feedback")
        rating = st.slider("Confidence score", 1, 5, 3)

        if st.button("Submit feedback"):
            entry = {"feedback": feedback_text, "rating": rating}
            try:
                post_feedback_to_backend(uploaded_file, entry)
                st.session_state.feedback_submitted = True
                st.success("Feedback submitted — thank you!")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Feedback already submitted for this image.")

else:
    st.info("Upload an image to begin.")

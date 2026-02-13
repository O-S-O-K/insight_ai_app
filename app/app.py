# app/app.py
import os
import json
import hashlib
from pathlib import Path
import io
import base64

import streamlit as st
from PIL import Image, ImageOps

# ----------------------------
# Backend configuration
# ----------------------------
USE_MOCK = os.environ.get("USE_MOCK_API", "false").lower() == "true"


def _resolve_backend_url() -> str | None:
    """Prefer Streamlit secrets, then env; do not override if already set."""
    secrets_url = None
    try:
        secrets_url = st.secrets.get("INSIGHT_BACKEND_URL")
    except Exception:
        secrets_url = None

    env_url = os.environ.get("INSIGHT_BACKEND_URL")
    chosen = secrets_url or env_url

    if chosen:
        os.environ["INSIGHT_BACKEND_URL"] = chosen
    return chosen


if USE_MOCK:
    from utils.mock_api_client import *
    st.write("Using mock API client")
else:
    backend_url = _resolve_backend_url()
    if not backend_url:
        st.error("Backend URL is not configured. Set INSIGHT_BACKEND_URL in Streamlit secrets.")
        st.stop()

    from utils.api_client import (
        predict_image as call_backend_predict,
        caption_image as call_backend_caption,
        gradcam_image as call_backend_gradcam,
        submit_feedback as post_feedback_to_backend,
    )

    st.write(f"Using live API client @ {backend_url}")

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Insight AI", layout="wide")
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

TOP_K = 3  # number of top predictions to display

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
        st.session_state.predictions = None
        st.session_state.caption = None
        st.session_state.gradcams = None

def display_gradcam_image(overlay_img: Image.Image, alpha: float, original_img: Image.Image):
    """Blend overlay with original using alpha and show in Streamlit"""
    original_img = original_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB").resize(original_img.size)
    blended = Image.blend(original_img, overlay_img, alpha)
    st.image(blended, caption="Grad-CAM Overlay", width="stretch")

# ----------------------------
# UI
# ----------------------------
st.title("Insight AI")
st.caption("Explainable image classification with BLIP captions and Grad-CAM visualizations")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image and handle EXIF orientation (important for mobile uploads)
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Auto-rotate based on EXIF data
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    col1, col2, col3 = st.columns([1, 1, 1])

    # ------------------------
    # Predict
    # ------------------------
    with col1:
        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                try:
                    result = call_backend_predict(uploaded_file)
                    st.session_state.predictions = result.get("predictions", [])
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("predictions"):
        st.subheader("Top Predictions")
        for i, pred in enumerate(st.session_state.predictions, start=1):
            st.write(f"{i}. **{pred['class_name']}** - Confidence: {pred['confidence']*100:.2f}%")

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
        if st.button("Grad-CAM"):
            with st.spinner("Computing Grad-CAM for top predictions..."):
                try:
                    result = call_backend_gradcam(uploaded_file, top_k=TOP_K)
                    st.session_state.gradcams = result.get("gradcams", [])
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("gradcams"):
        st.subheader("Grad-CAM Outputs")
        alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.4, 0.05)
        tabs = st.tabs([f"{g['class_name']} ({g['confidence']*100:.1f}%)" for g in st.session_state.gradcams])
        for tab, gradcam_data in zip(tabs, st.session_state.gradcams):
            with tab:
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

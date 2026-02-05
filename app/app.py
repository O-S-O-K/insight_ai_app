# app/app.py
import os
import json
import hashlib
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# point to local backend for development
os.environ["INSIGHT_BACKEND_URL"] = "http://localhost:8000"
# Toggle between mock and real API client
USE_MOCK = os.environ.get("USE_MOCK_API", "true").lower() == "true"

if USE_MOCK:
    from utils.mock_api_client import *
else:
    from utils.api_client import (
        predict_image as call_backend_predict,
        caption_image as call_backend_caption,
        gradcam_image as call_backend_gradcam,
        submit_feedback as post_feedback_to_backend,
    )

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Insight AI",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def image_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def reset_state_on_new_image(new_hash: str):
    if st.session_state.get("image_hash") != new_hash:
        st.session_state.image_hash = new_hash
        st.session_state.feedback_submitted = False


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("Insight AI")
st.caption("Explainable image classification with human-in-the-loop feedback")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    with col1:
        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                try:
                    result = call_backend_predict(uploaded_file)
                    st.session_state.prediction = result
                except Exception as e:
                    st.error(str(e))

    if "prediction" in st.session_state:
        pred = st.session_state.prediction
        st.subheader("Prediction")
        st.json(pred)

    # ------------------------------------------------------------------
    # Caption
    # ------------------------------------------------------------------
    with col2:
        if st.button("Caption"):
            with st.spinner("Generating caption..."):
                try:
                    result = call_backend_caption(uploaded_file)
                    st.session_state.caption = result
                except Exception as e:
                    st.error(str(e))

    if "caption" in st.session_state:
        st.subheader("Image caption")
        st.write(st.session_state.caption)

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------
    with col3:
        if st.button("Grad-CAM"):
            with st.spinner("Computing Grad-CAM..."):
                try:
                    result = call_backend_gradcam(uploaded_file)
                    st.session_state.gradcam = result
                except Exception as e:
                    st.error(str(e))

    if "gradcam" in st.session_state:
        st.subheader("Grad-CAM Output")
        st.json(st.session_state.gradcam)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Human Feedback")

    if not st.session_state.get("feedback_submitted", False):
        feedback_text = st.text_area("Your feedback")
        rating = st.slider("Confidence score", 1, 5, 3)

        if st.button("Submit feedback"):
            entry = {
                "feedback": feedback_text,
                "rating": rating,
            }
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

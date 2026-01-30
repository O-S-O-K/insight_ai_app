# ======================================================
# InsightAI - Explainable Image Classification App
# ======================================================

import os
import streamlit as st
import numpy as np
from PIL import Image

# ---- Path safety (Cloud + Local)
import sys
from pathlib import Path
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
ENABLE_BLIP = not IS_CLOUD  # BLIP disabled on Streamlit Cloud

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="InsightAI",
    layout="wide",
)

# ======================================================
# LOAD MODEL (CACHED)
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
"""
)

# ======================================================
# IMAGE UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    # ----------------------------------------------
    # Load image
    # ----------------------------------------------
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------
    preds = predict_image(model, img, top=3)

    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, start=1):
        st.write(f"{i}. **{label}** — {score * 100:.2f}%")

    # ----------------------------------------------
    # Grad-CAM
    # ----------------------------------------------
    st.subheader("🔥 Grad-CAM Explanation")

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor = preprocess_input(img_tensor)

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(model, last_conv, img_tensor)

    alpha = st.slider(
        "Heatmap intensity",
        min_value=0.2,
        max_value=0.7,
        value=0.4,
        step=0.05,
    )

    cam_img = overlay_heatmap(heatmap, img, alpha=alpha)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image", use_column_width=True)

    with col2:
        st.image(
            cam_img,
            caption="Grad-CAM Overlay (Model Attention)",
            use_column_width=True,
        )

    st.markdown(
        """
**What this shows:**  
Highlighted regions indicate which parts of the image most influenced the model’s prediction.
"""
    )

    # ----------------------------------------------
    # Feedback (Human-in-the-Loop)
    # ----------------------------------------------
    st.subheader("🧠 Feedback")

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    if not st.session_state.feedback_submitted:
        correct = st.radio(
            "Was the model’s top prediction correct?",
            ["Yes", "No"],
            horizontal=True,
        )

        if correct == "No":
            user_label = st.text_input(
                "What should the correct label be?"
            )
        else:
            user_label = preds[0][0]

        if st.button("Submit Feedback"):
            st.session_state.feedback = {
                "model_prediction": preds[0][0],
                "user_label": user_label,
            }
            st.session_state.feedback_submitted = True
            st.success("Thanks! Your feedback has been recorded.")
    else:
        st.info("Feedback already submitted for this image.")

    # ----------------------------------------------
    # Optional: Feedback summary (demo transparency)
    # ----------------------------------------------
    if "feedback" in st.session_state:
        with st.expander("View recorded feedback"):
            st.json(st.session_state.feedback)

    # ----------------------------------------------
    # BLIP Caption (Optional)
    # ----------------------------------------------
    st.subheader("📝 Image Caption")

    if ENABLE_BLIP:
        with st.spinner("Generating image caption..."):
            caption = generate_blip_caption(img)

        if caption:
            st.write(caption)
        else:
            st.info("No caption generated.")
    else:
        st.info("Image captioning is disabled in cloud deployments.")

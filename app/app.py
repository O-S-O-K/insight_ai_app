import os
import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parents[1]
sys.path.append(str(ROOT_DIR))

from utils.gradcam import (
    load_cnn_model,
    predict_image,
    get_gradcam_heatmap,
    overlay_heatmap,
    find_last_conv_layer,
)

from utils.blip_caption import generate_blip_caption

# ======================================================
# CONFIG / FEATURE FLAGS
# ======================================================
IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"
ENABLE_BLIP = not IS_CLOUD  # Disable BLIP by default on cloud

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
# UI HEADER
# ======================================================
st.title("🧠 InsightAI")
st.subheader("Explainable Image Classification with Human Feedback")
st.markdown(
    """
Upload an image to see:
- CNN predictions
- Grad-CAM explanations
- Optional BLIP image captioning
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
    # ------------------------------
    # Image loading
    # ------------------------------
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ------------------------------
    # Prediction
    # ------------------------------
    preds = predict_image(model, img, top=3)

    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, start=1):
        st.write(f"{i}. **{label}** — {score*100:.2f}%")

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
            "top_prediction": preds[0][0],
            "user_label": user_label,
        }
        st.session_state.feedback_submitted = True
        st.success("Thanks! Your feedback has been recorded.")
else:
    st.info("Feedback already submitted for this image.")

    # ------------------------------
    # Grad-CAM
    # ------------------------------
    top_label = preds[0][0]

    # Preprocess for Grad-CAM
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_tensor = preprocess_input(img_tensor)

    last_conv = find_last_conv_layer(model)
    heatmap = get_gradcam_heatmap(model, last_conv, img_tensor)
    cam_img = overlay_heatmap(heatmap, img)

    st.subheader("🔥 Grad-CAM Explanation")
    st.image(cam_img, use_column_width=True)

    # ------------------------------
    # BLIP Caption (Optional)
    # ------------------------------
    st.subheader("📝 Image Caption")

    if ENABLE_BLIP:
        with st.spinner("Generating image caption..."):
            try:
                caption = generate_blip_caption(img)
            except Exception:
                caption = None
                st.warning("BLIP captioning failed. Running without captions.")

        if caption:
            st.write(caption)
        else:
            st.info("No caption generated.")
    else:
        st.info("BLIP captioning is disabled in cloud deployments.")

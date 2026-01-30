# ======================================================
# InsightAI - Explainable Image Classification App
# ======================================================

import os
import sys
from pathlib import Path
import shutil
import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

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
ENABLE_BLIP = not IS_CLOUD  # BLIP disabled in cloud for stability

# Feedback paths
FEEDBACK_CSV = ROOT_DIR / "feedback_log.csv"
FEEDBACK_IMG_DIR = ROOT_DIR / "feedback_images"
FEEDBACK_IMG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="InsightAI", layout="wide")

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
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # -----------------------------
    # Load and display image
    # -----------------------------
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -----------------------------
    # Top-3 predictions
    # -----------------------------
    preds = predict_image(model, img, top=3)
    st.subheader("🔍 Top Predictions")
    for i, (label, score) in enumerate(preds, start=1):
        st.write(f"{i}. **{label}** — {score * 100:.2f}%")

    # -----------------------------
    # Grad-CAM
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
    # Feedback (Human-in-the-loop)
    # -----------------------------
    st.subheader("🧠 Feedback")

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    if not st.session_state.feedback_submitted:

        correct = st.radio(
            "Was the model’s top prediction correct?",
            ["Yes", "No"],
            horizontal=True,
        )

        user_label = None

        if correct == "Yes":
            user_label = preds[0][0]
        else:
            st.markdown("### What is the correct label?")
            top_labels = [label for label, _ in preds]
            options = top_labels[1:] + ["Other"]

            selection = st.radio(
                "Select one of the alternatives or choose *Other*", options
            )

            if selection == "Other":
                user_label = st.text_input("Enter the correct label")
            else:
                user_label = selection

        if st.button("Submit Feedback"):
            if user_label is None or user_label.strip() == "":
                st.warning("Please provide a valid label.")
            else:
                # -----------------------------
                # Save image locally for retraining
                # -----------------------------
                img_filename = uploaded_file.name
                img_save_path = FEEDBACK_IMG_DIR / img_filename
                try:
                    img.save(img_save_path)
                except Exception as e:
                    st.warning(f"Could not save image: {e}")

                # -----------------------------
                # Append feedback to CSV
                # -----------------------------
                feedback_entry = {
                    "uploaded_filename": img_filename,
                    "model_prediction": preds[0][0],
                    "user_label": user_label,
                    "was_correct": correct,
                }

                if FEEDBACK_CSV.exists():
                    df_existing = pd.read_csv(FEEDBACK_CSV)
                    df = pd.concat([df_existing, pd.DataFrame([feedback_entry])], ignore_index=True)
                else:
                    df = pd.DataFrame([feedback_entry])

                df.to_csv(FEEDBACK_CSV, index=False)

                st.session_state.feedback = feedback_entry
                st.session_state.feedback_submitted = True
                st.success("Thanks! Your feedback has been recorded and saved for retraining.")

    else:
        st.info("Feedback already submitted for this image.")

    # Optional: show feedback
    if "feedback" in st.session_state:
        with st.expander("View recorded feedback"):
            st.json(st.session_state.feedback)

    # -----------------------------
    # BLIP Caption
    # -----------------------------
    st.subheader("📝 Image Caption")

    if ENABLE_BLIP:
        with st.spinner("Generating image caption..."):
            caption = generate_blip_caption(img)

        if caption:
            st.write(caption)
        else:
            st.warning("Caption generation failed.")
    else:
        st.info(
            "Image captioning is disabled in the public demo due to cloud resource limits. "
            "Available in local deployment."
        )

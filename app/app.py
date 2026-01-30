# app/app.py
import os
import sys
import json
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocessing import preprocess_image
from utils.gradcam import (
    load_cnn_model_pretrained,
    get_gradcam_heatmap,
    overlay_heatmap,
    find_last_conv_layer
)
from utils.keyword_mapping import KEYWORD_CLASS_MAP
from utils.blip_caption import generate_blip_caption

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="InsightAI", layout="centered")
st.title("InsightAI: Interactive Image Classification with Feedback")
st.write("TensorFlow version:", __import__("tensorflow").__version__)

# ------------------------------
# Load CNN model
# ------------------------------
@st.cache_resource
def load_model():
    return load_cnn_model_pretrained()

cnn_model = load_model()
st.write("Model loaded:", cnn_model is not None)

# ------------------------------
# Load dynamic mapping
# ------------------------------
mapping_path = Path("blip_dynamic_mapping.json")
if mapping_path.exists():
    with open(mapping_path, "r") as f:
        dynamic_mapping = json.load(f)
else:
    dynamic_mapping = {}

feedback_log_path = Path("feedback_log.csv")

# ------------------------------
# Upload image
# ------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img_tensor = preprocess_image(img)

    # CNN predictions
    preds = cnn_model.predict(img_tensor)
    decoded = decode_predictions(preds, top=3)[0]

    st.write("Top 3 Predictions:")
    for _, label, score in decoded:
        st.write(f"{label}: {score*100:.2f}%")

    # BLIP caption
    caption = generate_blip_caption(img)
    st.write("BLIP Caption:")
    st.write(caption)

    # Map BLIP caption to classes
    mapped_classes = []
    caption_lower = caption.lower()
    for keyword, classes in KEYWORD_CLASS_MAP.items():
        if keyword in caption_lower:
            mapped_classes.extend(classes)
    for key, classes in dynamic_mapping.items():
        if key in caption_lower:
            mapped_classes.extend(classes)
    mapped_classes = list(set(mapped_classes))
    st.write("Mapped classes based on BLIP caption and past feedback:", mapped_classes)

    # Session-only boosting
    boost_factor = 1.5
    adjusted_preds = preds.copy()
    for i, (_, label, score) in enumerate(decoded):
        if label in mapped_classes:
            adjusted_preds[0][i] *= boost_factor
    adjusted_preds[0] /= np.sum(adjusted_preds[0])

    decoded_boosted = decode_predictions(adjusted_preds, top=3)[0]
    st.write("Top 3 Predictions (after session-only boost):")
    for _, label, score in decoded_boosted:
        st.write(f"{label}: {score*100:.2f}%")

    # ------------------------------
    # Feedback with dynamic Grad-CAM
    # ------------------------------
    st.write("---")
    st.write("Provide feedback to improve the model:")

    pred_correct = st.radio(
        f"Was the top prediction '{decoded_boosted[0][1]}' correct?",
        options=["Yes", "No"]
    )

    user_selected_label = decoded_boosted[0][1]  # default top prediction

    if pred_correct == "No":
        top3_options = [f"{label} ({score*100:.2f}%)" for _, label, score in decoded_boosted] + ["Other"]
        selected = st.radio("Select the correct class from top-3 or Other:", options=top3_options, index=1)

        if selected == "Other":
            user_selected_label = st.text_input("Enter the correct label:", "")
        else:
            user_selected_label = selected.split(" ")[0]

    caption_correct = st.radio("Was the BLIP caption correct?", options=["Yes", "No"])
    user_text = st.text_input("Optional feedback (comments, typo fixes, etc.):")

    # ------------------------------
    # Grad-CAM overlay for user-selected label
    # ------------------------------
    last_conv_layer = find_last_conv_layer(cnn_model)
    if last_conv_layer:
        # Determine class index
        class_index = None
        for i, (_, label, _) in enumerate(decoded_boosted):
            if label == user_selected_label:
                class_index = i
                break
        if class_index is None:
            class_index = np.argmax(preds)

        heatmap = get_gradcam_heatmap(cnn_model, last_conv_layer, img_tensor, pred_index=class_index)
        overlay = overlay_heatmap(heatmap, img, alpha=0.4)
        st.image(overlay, caption=f"Grad-CAM Overlay for '{user_selected_label}'", width=300)
    else:
        st.error("No Conv2D layer found for Grad-CAM.")

    # ------------------------------
    # Submit feedback
    # ------------------------------
    if st.button("Submit Feedback"):
        if not user_selected_label:
            st.error("Please provide a label for feedback.")
        else:
            # Update dynamic mapping
            if caption_lower in dynamic_mapping:
                if user_selected_label not in dynamic_mapping[caption_lower]:
                    dynamic_mapping[caption_lower].append(user_selected_label)
            else:
                dynamic_mapping[caption_lower] = [user_selected_label]

            with open(mapping_path, "w") as f:
                json.dump(dynamic_mapping, f, indent=2)

            feedback_row = {
                "timestamp": datetime.utcnow().isoformat(),
                "top_prediction": decoded_boosted[0][1],
                "top_prediction_conf": decoded_boosted[0][2],
                "pred_correct": pred_correct,
                "user_selected_label": user_selected_label,
                "blip_caption": caption,
                "caption_correct": caption_correct,
                "user_feedback": user_text
            }

            df = pd.DataFrame([feedback_row])
            df.to_csv(
                feedback_log_path,
                mode="a",
                header=not feedback_log_path.exists(),
                index=False
            )

            st.success("Feedback saved! ✅")

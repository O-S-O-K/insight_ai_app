import os
import torch
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-small"
LOCAL_MODEL_PATH = "models/blip"

# ======================================================
# LOAD MODEL (CACHED, SAFE)
# ======================================================

@st.cache_resource
def load_blip_model():
    """
    Loads BLIP model either from local path (preferred)
    or Hugging Face Hub.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        processor = BlipProcessor.from_pretrained(LOCAL_MODEL_PATH)
        model = BlipForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)
    else:
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

    model.eval()
    return processor, model

# ======================================================
# GENERATE CAPTION
# ======================================================

def generate_blip_caption(image):
    """
    Generates a natural-language caption for a PIL image.
    Returns None on failure.
    """
    try:
        processor, model = load_blip_model()

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, max_length=30)

        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

    except Exception:
        return None

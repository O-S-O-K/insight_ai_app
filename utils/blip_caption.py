# utils/blip_caption.py

from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# ------------------------------
# Constants
# ------------------------------
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-small"  # smaller for Cloud
LOCAL_MODEL_PATH = "models/blip-small"  # optional local path if you pre-download

# ------------------------------
# Lazy-load BLIP processor & model
# ------------------------------
@st.cache_resource
def load_blip_model():
    """
    Loads the BLIP processor and model. Uses local path if available, otherwise downloads.
    Cached across Streamlit sessions.
    """
    # If you pre-downloaded the model to LOCAL_MODEL_PATH
    if os.path.exists(LOCAL_MODEL_PATH):
        processor = BlipProcessor.from_pretrained(LOCAL_MODEL_PATH)
        model = BlipForConditionalGeneration.from_pretrained(LOCAL_MODEL_PATH)
    else:
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)
    return processor, model

# ------------------------------
# Caption generation function
# ------------------------------
def generate_blip_caption(image: Image.Image) -> str:
    """
    Generates a natural-language caption for a PIL image using BLIP.
    """
    processor, model = load_blip_model()

    # Prepare image for BLIP
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption
    out = model.generate(**inputs)
    
    # Decode output to string
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

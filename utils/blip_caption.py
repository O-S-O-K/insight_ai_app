# utils/blip_caption.py

import os
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

@st.cache_resource(show_spinner="Loading vision-language model (BLIP)...")
def load_blip_model():
    """
    Loads BLIP processor and model once per app container.
    Safe for cloud deployment.
    """
    token = os.getenv("HF_TOKEN", None)

    processor = BlipProcessor.from_pretrained(
        BLIP_MODEL_ID,
        token=token
    )
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_ID,
        token=token
    )

    return processor, model


def generate_blip_caption(image: Image.Image) -> str | None:
    """
    Generates a caption for the input PIL image.
    Returns None if BLIP is unavailable.
    """
    try:
        processor, model = load_blip_model()
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        return processor.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        # Never crash the app for captioning
        print(f"[BLIP disabled] {e}")
        return None

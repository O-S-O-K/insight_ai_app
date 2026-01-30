# utils/blip_caption.py

from PIL import Image
import torch
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache_resource(show_spinner="Loading BLIP captioning model...")
def load_blip_model():
    """
    Lazily load and cache the BLIP processor and model.
    This runs once per app instance.
    """
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.eval()
    return processor, model


def generate_blip_caption(image: Image.Image) -> str:
    """
    Generate a caption for the input PIL image using BLIP.
    """
    processor, model = load_blip_model()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs)

    caption = processor.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return caption

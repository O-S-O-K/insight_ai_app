from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

processor = None
model = None

def load_caption_model():
    global processor, model
    if model is None:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    return processor, model

def generate_caption(image):
    processor, model = load_caption_model()
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

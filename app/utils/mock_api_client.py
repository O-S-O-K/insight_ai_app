def predict_image(_):
    return {
        "label": "mock_cat",
        "confidence": 0.93
    }

def caption_image(_):
    return "A mock image of a cat sitting on a chair."

def gradcam_image(_):
    return {
        "heatmap": "mock_base64_string"
    }

def submit_feedback(_, __):
    return {"status": "ok"}

# utils/mock_api_client.py
import base64

def predict_image(uploaded_file):
    return {
        "predictions": [
            {"label": "cat", "score": 0.92},
            {"label": "dog", "score": 0.05},
            {"label": "rabbit", "score": 0.03},
        ]
    }

def caption_image(uploaded_file):
    return "A cute animal in a photo."

def gradcam_image(uploaded_file):
    # Return a dummy base64 overlay
    return {"overlay_base64": ""}

def submit_feedback(uploaded_file, entry):
    # Just print to console locally
    print("Feedback received:", entry)
    return {"status": "ok"}

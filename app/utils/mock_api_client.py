# utils/mock_api_client.py
import base64

# -----------------------
# Mock API functions
# -----------------------

def predict_image(uploaded_file, top_k: int = 3):
    return {
        "predictions": [
            {"class_idx": 0, "class_name": "cat", "confidence": 0.92},
            {"class_idx": 1, "class_name": "dog", "confidence": 0.05},
            {"class_idx": 2, "class_name": "rabbit", "confidence": 0.03},
        ]
    }

def caption_image(uploaded_file):
    return {"caption": "A cute animal in a photo."}

def gradcam_image(uploaded_file, top_k: int = 3):
    # Return a dummy base64 overlay matching real API format
    return {
        "gradcams": [
            {"class_idx": 0, "class_name": "cat", "confidence": 0.92, "heatmap_base64": "data:image/png;base64,"},
            {"class_idx": 1, "class_name": "dog", "confidence": 0.05, "heatmap_base64": "data:image/png;base64,"},
            {"class_idx": 2, "class_name": "rabbit", "confidence": 0.03, "heatmap_base64": "data:image/png;base64,"},
        ]
    }

def submit_feedback(uploaded_file, entry):
    # Just print to console locally
    print("Feedback received:", entry)
    return {"status": "ok"}

# -----------------------
# Aliases matching app.py
# -----------------------
call_backend_predict = predict_image
call_backend_caption = caption_image
call_backend_gradcam = gradcam_image
post_feedback_to_backend = submit_feedback

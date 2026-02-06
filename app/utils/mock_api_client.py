# utils/mock_api_client.py
import base64

# -----------------------
# Mock API functions
# -----------------------

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

# -----------------------
# Aliases matching app.py
# -----------------------
call_backend_predict = predict_image
call_backend_caption = caption_image
call_backend_gradcam = gradcam_image
post_feedback_to_backend = submit_feedback

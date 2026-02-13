import os
import requests


def _get_base_url() -> str:
    base_url = os.environ.get("INSIGHT_BACKEND_URL")
    if not base_url:
        raise RuntimeError(
            "INSIGHT_BACKEND_URL is not set. Configure it via Streamlit secrets or environment."
        )
    return base_url


def _file_tuple(file_obj):
    """Prepare file for multipart upload"""
    return (file_obj.name, file_obj, "image/jpeg")


def _handle_response(response):
    """Handle API response and raise user-friendly errors"""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Try to extract error message from JSON response
        try:
            error_data = response.json()
            error_msg = error_data.get("error", str(e))
        except:
            error_msg = str(e)
        raise RuntimeError(error_msg) from e


# ----------------------------
# Prediction
# ----------------------------
def predict_image(file_obj, top_k: int = 3):
    """Return top-K predictions from backend"""
    files = {"file": _file_tuple(file_obj)}
    data = {"top_k": str(top_k)}
    r = requests.post(f"{_get_base_url()}/predict", files=files, data=data, timeout=30)
    return _handle_response(r)

# ----------------------------
# Grad-CAM
# ----------------------------
def gradcam_image(file_obj, top_k: int = 3):
    """Return Grad-CAM overlays for top-K predictions"""
    files = {"file": _file_tuple(file_obj)}
    data = {"top_k": str(top_k)}
    r = requests.post(f"{_get_base_url()}/gradcam", files=files, data=data, timeout=60)
    return _handle_response(r)

# ----------------------------
# Caption
# ----------------------------
def caption_image(file_obj):
    """Return BLIP caption for the image"""
    files = {"file": _file_tuple(file_obj)}
    r = requests.post(f"{_get_base_url()}/caption", files=files, timeout=30)
    return _handle_response(r)

# ----------------------------
# Human Feedback
# ----------------------------
def submit_feedback(file_obj, entry: dict):
    """Submit human feedback"""
    import json
    files = {"file": _file_tuple(file_obj)}
    data = {"entry": json.dumps(entry)}
    r = requests.post(f"{_get_base_url()}/feedback", files=files, data=data, timeout=30)
    return _handle_response(r)

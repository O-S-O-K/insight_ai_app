import os
import json
import requests


def _get_base_url() -> str:
    base_url = os.environ.get("INSIGHT_BACKEND_URL")
    if not base_url:
        raise RuntimeError(
            "INSIGHT_BACKEND_URL is not set. Configure it via Streamlit secrets or environment."
        )
    return base_url.rstrip("/")


def _v1_url(path: str) -> str:
    """Build a versioned API endpoint URL."""
    return f"{_get_base_url()}/api/v1{path}"


def _file_tuple(file_obj):
    """Prepare file for multipart upload."""
    return (file_obj.name, file_obj, "image/jpeg")


def _handle_response(response):
    """Handle API response and raise user-friendly errors."""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_data = response.json()
            error_msg = error_data.get("error", str(e))
        except Exception:
            error_msg = str(e)
        raise RuntimeError(error_msg) from e


# ----------------------------
# Prediction (with calibration)
# ----------------------------
def predict_image(file_obj, top_k: int = 3):
    """Return top-K calibrated predictions from backend."""
    files = {"file": _file_tuple(file_obj)}
    data = {"top_k": str(top_k)}
    r = requests.post(_v1_url("/predict"), files=files, data=data, timeout=60)
    return _handle_response(r)

# ----------------------------
# Grad-CAM
# ----------------------------
def gradcam_image(file_obj, top_k: int = 3):
    """Return Grad-CAM overlays for top-K predictions."""
    files = {"file": _file_tuple(file_obj)}
    data = {"top_k": str(top_k)}
    r = requests.post(_v1_url("/gradcam"), files=files, data=data, timeout=60)
    return _handle_response(r)

# ----------------------------
# Caption (BLIP)
# ----------------------------
def caption_image(file_obj):
    """Return BLIP caption for the image."""
    files = {"file": _file_tuple(file_obj)}
    r = requests.post(_v1_url("/caption"), files=files, timeout=60)
    return _handle_response(r)

# ----------------------------
# SHAP explainability
# ----------------------------
def shap_explain(file_obj):
    """Return SHAP GradientExplainer attribution map."""
    files = {"file": _file_tuple(file_obj)}
    r = requests.post(_v1_url("/shap"), files=files, timeout=180)
    return _handle_response(r)

# ----------------------------
# CLIP zero-shot classification
# ----------------------------
def clip_classify(file_obj, labels: list):
    """Return CLIP similarity scores for custom labels."""
    files = {"file": _file_tuple(file_obj)}
    data = {"labels": json.dumps(labels)}
    r = requests.post(_v1_url("/clip"), files=files, data=data, timeout=60)
    return _handle_response(r)

# ----------------------------
# Human Feedback (with active learning fields)
# ----------------------------
def submit_feedback(file_obj, entry: dict):
    """Submit human feedback with optional active learning flag."""
    files = {"file": _file_tuple(file_obj)}
    data = {"entry": json.dumps(entry)}
    r = requests.post(_v1_url("/feedback"), files=files, data=data, timeout=30)
    return _handle_response(r)

# ----------------------------
# Active Learning Summary
# ----------------------------
def get_active_learning_summary():
    """Return active learning flagging statistics."""
    r = requests.get(_v1_url("/active-learning/summary"), timeout=15)
    return _handle_response(r)

# ----------------------------
# MLflow Recent Runs
# ----------------------------
def get_mlflow_runs(limit: int = 10):
    """Return recent MLflow inference runs."""
    r = requests.get(_v1_url(f"/mlflow/recent-runs?limit={limit}"), timeout=15)
    return _handle_response(r)

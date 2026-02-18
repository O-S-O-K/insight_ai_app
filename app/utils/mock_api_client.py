# utils/mock_api_client.py
import base64
import random

# -----------------------
# Mock API functions
# -----------------------

def predict_image(uploaded_file, top_k: int = 3, model_type: str = None):
    return {
        "predictions": [
            {"class_idx": 0, "class_name": "cat", "confidence": 0.92},
            {"class_idx": 1, "class_name": "dog", "confidence": 0.05},
            {"class_idx": 2, "class_name": "rabbit", "confidence": 0.03},
        ],
        "calibrated": True,
        "temperature": 1.5,
    }

def caption_image(uploaded_file):
    return {"caption": "A cute animal in a photo."}

def gradcam_image(uploaded_file, top_k: int = 3, model_type: str = None):
    return {
        "gradcams": [
            {"class_idx": 0, "class_name": "cat", "confidence": 0.92, "heatmap_base64": "data:image/png;base64,"},
            {"class_idx": 1, "class_name": "dog", "confidence": 0.05, "heatmap_base64": "data:image/png;base64,"},
            {"class_idx": 2, "class_name": "rabbit", "confidence": 0.03, "heatmap_base64": "data:image/png;base64,"},
        ]
    }

def submit_feedback(uploaded_file, entry):
    print("Feedback received:", entry)
    return {"status": "ok"}

# ----------------------------
# SHAP Explainability (mock)
# ----------------------------
def shap_explain(file_obj, model_type: str = None):
    """Mock SHAP GradientExplainer attribution map."""
    return {
        "shap_plot_base64": "data:image/png;base64,",
        "top_class": "cat",
        "top_class_idx": 0,
        "top_confidence": 0.92,
        "explanation": "Highlighted regions contributed most to the 'cat' prediction.",
        "method": "SHAP GradientExplainer",
    }

# ----------------------------
# CLIP zero-shot (mock)
# ----------------------------
def clip_classify(file_obj, labels: list):
    """Mock CLIP similarity scores for custom labels."""
    if not labels:
        return {"results": [], "model": "CLIP ViT-B/32", "zero_shot": True}
    # Assign random-ish scores that sum to ~1
    raw = [random.uniform(0.1, 1.0) for _ in labels]
    total = sum(raw)
    scores = [round(r / total, 4) for r in raw]
    results = sorted(
        [{"label": lbl, "score": sc} for lbl, sc in zip(labels, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {"results": results, "model": "CLIP ViT-B/32", "zero_shot": True}

# ----------------------------
# Active Learning (mock)
# ----------------------------
def get_active_learning_summary():
    """Mock active learning flagging statistics."""
    return {
        "total_feedback": 0,
        "flagged": 0,
        "flag_rate": 0.0,
        "flagged_samples": [],
    }

# ----------------------------
# MLflow Recent Runs (mock)
# ----------------------------
def get_mlflow_runs(limit: int = 10):
    """Mock MLflow inference runs."""
    return {"runs": [], "total": 0}

# -----------------------
# Aliases matching app.py
# -----------------------
call_backend_predict = predict_image
call_backend_caption = caption_image
call_backend_gradcam = gradcam_image
post_feedback_to_backend = submit_feedback

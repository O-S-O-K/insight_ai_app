# app/app.py
import os
import json
import hashlib
from pathlib import Path
import io
import base64
import requests

import streamlit as st
from PIL import Image, ImageOps

# ----------------------------
# Streamlit config (must be first st call)
# ----------------------------
st.set_page_config(
    page_title="Insight AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Backend configuration
# ----------------------------
USE_MOCK = os.environ.get("USE_MOCK_API", "false").lower() == "true"
ACTIVE_LEARNING_THRESHOLD = float(os.environ.get("ACTIVE_LEARNING_THRESHOLD", "0.5"))


def _resolve_backend_url() -> str | None:
    """Prefer Streamlit secrets, then env; do not override if already set."""
    secrets_url = None
    try:
        secrets_url = st.secrets.get("INSIGHT_BACKEND_URL")
    except Exception:
        secrets_url = None

    env_url = os.environ.get("INSIGHT_BACKEND_URL")
    chosen = secrets_url or env_url

    if chosen:
        os.environ["INSIGHT_BACKEND_URL"] = chosen
    return chosen


if USE_MOCK:
    from utils.mock_api_client import (
        predict_image as call_backend_predict,
        caption_image as call_backend_caption,
        gradcam_image as call_backend_gradcam,
        submit_feedback as post_feedback_to_backend,
        shap_explain,
        clip_classify,
        get_active_learning_summary,
        get_mlflow_runs,
    )
else:
    backend_url = _resolve_backend_url()
    if not backend_url:
        st.error("Backend URL is not configured. Set INSIGHT_BACKEND_URL in Streamlit secrets.")
        st.stop()

    from utils.api_client import (
        predict_image as call_backend_predict,
        caption_image as call_backend_caption,
        gradcam_image as call_backend_gradcam,
        submit_feedback as post_feedback_to_backend,
        shap_explain,
        clip_classify,
        get_active_learning_summary,
        get_mlflow_runs,
    )

# ----------------------------
# Backend health check
# ----------------------------
def check_backend_health():
    """Check if backend is available and responsive."""
    if USE_MOCK:
        return {"status": "ready", "healthy": True}

    backend_url = _resolve_backend_url()
    if not backend_url:
        return {"status": "error", "healthy": False, "message": "No backend URL configured"}

    try:
        response = requests.get(f"{backend_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            healthy = status in ["ready", "loading"]
            return {"status": status, "healthy": healthy, "data": data, "message": None}
        else:
            return {"status": "error", "healthy": False, "message": f"Backend returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "healthy": False, "message": "Cannot connect to backend server"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "healthy": False, "message": "Backend server is not responding (timeout)"}
    except Exception as e:
        return {"status": "error", "healthy": False, "message": f"Error checking backend: {str(e)}"}


def show_maintenance_page(health_status):
    """Display a friendly maintenance page when backend is unavailable."""
    st.markdown(
        """
        <style>
        .maintenance-container { text-align: center; padding: 3rem 2rem; }
        .maintenance-icon { font-size: 5rem; margin-bottom: 1rem; }
        .maintenance-title { font-size: 2rem; font-weight: 600; margin-bottom: 1rem; color: #FF6B6B; }
        .maintenance-message { font-size: 1.1rem; color: #666; margin-bottom: 2rem; line-height: 1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    status = health_status.get("status", "unknown")
    message = health_status.get("message", "")

    if status == "offline":
        icon, title = "🔌", "Backend Currently Offline"
        description = "The AI backend service is currently unavailable. This may be due to maintenance or resource limits."
    elif status == "timeout":
        icon, title = "⏱️", "Backend Not Responding"
        description = "The backend server is taking too long to respond. It may be starting up or experiencing high load."
    elif status == "loading":
        icon, title = "⏳", "AI Models Loading"
        description = "The backend is currently loading AI models. This usually takes 2-5 minutes. Please check back shortly!"
    elif status == "error":
        icon, title = "⚠️", "Backend Service Error"
        description = "The backend encountered an error. This may be a temporary issue."
    else:
        icon, title = "🛠️", "Service Under Maintenance"
        description = "The Insight AI backend is currently unavailable."

    st.markdown(
        f"""
        <div class="maintenance-container">
            <div class="maintenance-icon">{icon}</div>
            <div class="maintenance-title">{title}</div>
            <div class="maintenance-message">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical Details"):
        st.json(health_status)
        if message:
            st.info(f"**Details:** {message}")
        st.markdown("""
        **Common Reasons:**
        - Backend instance is spinning up (HF Spaces cold start)
        - AI models are still loading (takes 2-5 minutes after deploy)
        - Backend is deploying new updates

        **What to do:**
        - Wait a few minutes and refresh the page
        - Check back later if the issue persists
        """)

    if st.button("🔄 Check Again", type="primary"):
        st.rerun()

    st.divider()
    st.markdown("""
    ### About Insight AI

    Insight AI is an explainable image classification tool that provides:
    - 🎯 **Predictions** - Identify ImageNet object classes with calibrated confidence
    - 🔥 **Grad-CAM** - Visual explanations of model decisions
    - 🧠 **SHAP** - Feature attribution maps (GradientExplainer)
    - 🔍 **CLIP** - Zero-shot classification with custom labels
    - 💬 **Captions** - Natural language image descriptions (BLIP)

    **GitHub**: [O-S-O-K/insight_ai_app](https://github.com/O-S-O-K/insight_ai_app)
    """)

    st.stop()


# ----------------------------
# Load metadata
# ----------------------------
ROOT = Path(__file__).resolve().parent
METADATA_PATH = ROOT.parent / "models" / "model_metadata.json"
MEDICAL_METADATA_PATH = ROOT.parent / "models" / "medical_metadata.json"

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    LABEL_MAP = metadata.get("classes", {})
    TEMPERATURE = metadata.get("temperature", 1.0)
else:
    LABEL_MAP = {}
    TEMPERATURE = 1.0

TOP_K = 3

# ----------------------------
# Helpers
# ----------------------------
def image_hash(uploaded_file) -> str:
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def reset_state_on_new_image(new_hash: str):
    if st.session_state.get("image_hash") != new_hash:
        st.session_state.image_hash = new_hash
        st.session_state.feedback_submitted = False
        st.session_state.predictions = None
        st.session_state.caption = None
        st.session_state.gradcams = None
        st.session_state.shap_result = None
        st.session_state.clip_results = None
        st.session_state.active_learning_flag = False
        st.session_state.flag_reason = None


def display_gradcam_image(overlay_img: Image.Image, alpha: float, original_img: Image.Image):
    """Blend overlay with original using alpha and display."""
    original_img = original_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB").resize(original_img.size)
    blended = Image.blend(original_img, overlay_img, alpha)
    st.image(blended, caption="Grad-CAM Overlay", use_container_width=True)


# ----------------------------
# Health check
# ----------------------------
if not USE_MOCK:
    with st.spinner("Checking backend status..."):
        health_status = check_backend_health()

    if not health_status["healthy"]:
        show_maintenance_page(health_status)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Insight AI")
    st.caption("Explainable Image Intelligence")

    st.divider()

    # Model info
    st.subheader("Model")
    model_display = st.selectbox(
        "Active model",
        ["ImageNet (General)", "Medical Imaging (ISIC)"],
        help="Switch requires backend MODEL_TYPE env var update.",
    )
    st.caption(
        "MobileNetV2 · 1000 classes" if "ImageNet" in model_display
        else "EfficientNetB0 · Melanoma / Benign"
    )
    if TEMPERATURE != 1.0:
        st.caption(f"Temperature scaling: T={TEMPERATURE}")

    st.divider()

    # MLflow recent runs
    st.subheader("MLflow Runs")
    if st.button("Refresh runs", key="refresh_mlflow"):
        st.session_state.mlflow_runs = None

    if st.session_state.get("mlflow_runs") is None:
        try:
            mlflow_data = get_mlflow_runs(limit=5)
            st.session_state.mlflow_runs = mlflow_data
        except Exception:
            st.session_state.mlflow_runs = {"runs": [], "total": 0}

    mlflow_data = st.session_state.get("mlflow_runs", {"runs": [], "total": 0})
    runs = mlflow_data.get("runs", [])

    if runs:
        for run in runs[:5]:
            endpoint = run.get("tags", {}).get("endpoint", run.get("endpoint", "predict"))
            conf = run.get("metrics", {}).get("top1_confidence", run.get("top1_confidence"))
            conf_str = f" · {conf*100:.1f}%" if conf is not None else ""
            st.caption(f"`{endpoint}`{conf_str}")
    else:
        st.caption("No runs recorded yet.")

    st.divider()

    # Active learning summary
    st.subheader("Active Learning")
    try:
        al_data = get_active_learning_summary()
        total = al_data.get("total_feedback", 0)
        flagged = al_data.get("flagged", 0)
        rate = al_data.get("flag_rate", 0.0)
        st.metric("Flagged / Total", f"{flagged} / {total}")
        if total > 0:
            st.progress(min(rate, 1.0), text=f"Flag rate: {rate*100:.1f}%")
    except Exception:
        st.caption("Active learning data unavailable.")

# ----------------------------
# Main UI
# ----------------------------
st.title("Insight AI")
st.caption("Explainable image classification · BLIP captions · Grad-CAM · SHAP · CLIP zero-shot")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img)
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    # ----------------------------------------
    # Action buttons: Predict | Caption | Grad-CAM | SHAP
    # ----------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    # --- Predict ---
    with col1:
        if st.button("Predict", use_container_width=True):
            with st.spinner("Running prediction..."):
                try:
                    result = call_backend_predict(uploaded_file, top_k=TOP_K)
                    preds = result.get("predictions", [])
                    st.session_state.predictions = preds
                    st.session_state.predict_meta = {
                        "calibrated": result.get("calibrated", False),
                        "temperature": result.get("temperature", 1.0),
                    }
                    # Auto-flag low-confidence predictions
                    if preds and preds[0]["confidence"] < ACTIVE_LEARNING_THRESHOLD:
                        st.session_state.active_learning_flag = True
                        st.session_state.flag_reason = "low_confidence"
                except Exception as e:
                    st.error(str(e))

    # --- Caption ---
    with col2:
        if st.button("Caption", use_container_width=True):
            with st.spinner("Generating caption..."):
                try:
                    result = call_backend_caption(uploaded_file)
                    st.session_state.caption = result
                except Exception as e:
                    st.error(str(e))

    # --- Grad-CAM ---
    with col3:
        if st.button("Grad-CAM", use_container_width=True):
            with st.spinner("Computing Grad-CAM..."):
                try:
                    result = call_backend_gradcam(uploaded_file, top_k=TOP_K)
                    st.session_state.gradcams = result.get("gradcams", [])
                except Exception as e:
                    st.error(str(e))

    # --- SHAP ---
    with col4:
        if st.button("SHAP", use_container_width=True):
            with st.spinner("Computing SHAP attributions..."):
                try:
                    result = shap_explain(uploaded_file)
                    st.session_state.shap_result = result
                except Exception as e:
                    st.error(str(e))

    st.divider()

    # ----------------------------------------
    # Results display
    # ----------------------------------------

    # Predictions
    if st.session_state.get("predictions"):
        preds = st.session_state.predictions
        meta = st.session_state.get("predict_meta", {})
        calibrated = meta.get("calibrated", False)
        temp = meta.get("temperature", 1.0)

        badge = " *(calibrated)*" if calibrated else ""
        st.subheader(f"Top Predictions{badge}")

        # Active learning warning
        if st.session_state.get("active_learning_flag") and st.session_state.get("flag_reason") == "low_confidence":
            st.warning(
                f"Low-confidence prediction (top confidence < {ACTIVE_LEARNING_THRESHOLD*100:.0f}%) — "
                "this image has been auto-flagged for human review."
            )

        for i, pred in enumerate(preds, start=1):
            conf = pred["confidence"] * 100
            bar_color = "normal" if conf >= 50 else "inverse"
            st.write(f"{i}. **{pred['class_name']}** — {conf:.2f}%")
            st.progress(min(pred["confidence"], 1.0))

        if calibrated:
            st.caption(f"Confidence scores adjusted via temperature scaling (T={temp})")

    # Caption
    if st.session_state.get("caption"):
        st.subheader("Image Caption (BLIP)")
        st.write(st.session_state.caption.get("caption"))

    # Grad-CAM and SHAP together in tabs
    show_gradcam = bool(st.session_state.get("gradcams"))
    show_shap = bool(st.session_state.get("shap_result"))

    if show_gradcam or show_shap:
        viz_tabs = []
        if show_gradcam:
            viz_tabs.append("Grad-CAM")
        if show_shap:
            viz_tabs.append("SHAP")

        tabs = st.tabs(viz_tabs)
        tab_idx = 0

        if show_gradcam:
            with tabs[tab_idx]:
                st.subheader("Grad-CAM Outputs")
                alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.4, 0.05, key="gradcam_alpha")
                gcam_tabs = st.tabs(
                    [f"{g['class_name']} ({g['confidence']*100:.1f}%)" for g in st.session_state.gradcams]
                )
                for gcam_tab, gradcam_data in zip(gcam_tabs, st.session_state.gradcams):
                    with gcam_tab:
                        b64_data = gradcam_data["heatmap_base64"].split(",")[1]
                        overlay_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                        display_gradcam_image(overlay_img, alpha, img)
            tab_idx += 1

        if show_shap:
            with tabs[tab_idx]:
                shap = st.session_state.shap_result
                st.subheader("SHAP Feature Attribution")
                st.caption(f"Method: {shap.get('method', 'SHAP GradientExplainer')}")
                b64 = shap.get("shap_plot_base64", "")
                if b64 and len(b64) > 30:
                    # Strip data URI prefix if present
                    raw = b64.split(",")[1] if "," in b64 else b64
                    shap_img = Image.open(io.BytesIO(base64.b64decode(raw)))
                    st.image(shap_img, caption="SHAP Attribution Map", use_container_width=True)
                st.info(shap.get("explanation", "Red regions increase the prediction; blue regions decrease it."))
                if shap.get("top_class"):
                    st.caption(
                        f"Top predicted class: **{shap['top_class']}** "
                        f"({shap.get('top_confidence', 0)*100:.1f}%)"
                    )

    # ----------------------------------------
    # CLIP Zero-Shot Classification
    # ----------------------------------------
    st.divider()
    st.subheader("CLIP Zero-Shot Classification")
    st.caption("Use OpenAI CLIP to classify against any custom labels — no training required.")

    clip_input = st.text_input(
        "Custom labels (comma-separated)",
        placeholder="e.g. cat, dog, car, airplane",
        key="clip_label_input",
    )

    if st.button("Classify with CLIP", use_container_width=False):
        labels = [lbl.strip() for lbl in clip_input.split(",") if lbl.strip()]
        if not labels:
            st.warning("Enter at least one label.")
        elif len(labels) > 20:
            st.warning("Please enter 20 or fewer labels.")
        else:
            with st.spinner("Running CLIP zero-shot classification..."):
                try:
                    result = clip_classify(uploaded_file, labels)
                    st.session_state.clip_results = result
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("clip_results"):
        clip_data = st.session_state.clip_results
        results = clip_data.get("results", [])
        model_name = clip_data.get("model", "CLIP ViT-B/32")
        st.caption(f"Model: {model_name}")
        if results:
            for r in results:
                st.write(f"**{r['label']}** — {r['score']*100:.1f}%")
                st.progress(min(r["score"], 1.0))

    # ----------------------------------------
    # Human Feedback
    # ----------------------------------------
    st.divider()
    st.subheader("Human Feedback")

    if not st.session_state.get("feedback_submitted", False):
        feedback_text = st.text_area("Your feedback")
        rating = st.slider("Rating", 1, 5, 3)

        # Active learning manual flag
        auto_flagged = st.session_state.get("active_learning_flag", False)
        manual_flag = st.checkbox(
            "Flag this image for human review",
            value=auto_flagged,
            help="Flag uncertain or incorrect predictions for the active learning pipeline.",
        )

        if st.button("Submit Feedback"):
            preds = st.session_state.get("predictions", [])
            top_conf = preds[0]["confidence"] if preds else None
            top_class = preds[0]["class_name"] if preds else None

            flag_reason = st.session_state.get("flag_reason") if auto_flagged else ("user_flagged" if manual_flag else None)

            entry = {
                "feedback": feedback_text,
                "rating": rating,
                "active_learning_flag": manual_flag,
                "flag_reason": flag_reason,
                "top1_confidence": top_conf,
                "predicted_class": top_class,
            }
            try:
                post_feedback_to_backend(uploaded_file, entry)
                # Update MLflow cache to refresh on next sidebar render
                st.session_state.mlflow_runs = None
                st.session_state.feedback_submitted = True
                st.success("Feedback submitted — thank you!")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Feedback already submitted for this image.")

else:
    st.info("Upload an image to begin.")

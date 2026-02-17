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
# Backend configuration
# ----------------------------
USE_MOCK = os.environ.get("USE_MOCK_API", "false").lower() == "true"


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
    from utils.mock_api_client import *
    st.write("Using mock API client")
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
    )

    st.write(f"Using live API client @ {backend_url}")

# ----------------------------
# Backend health check
# ----------------------------
def check_backend_health():
    """Check if backend is available and responsive"""
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

            # Backend is healthy if status is "ready" or "loading"
            # "error" means backend failed to load models
            healthy = status in ["ready", "loading"]

            return {
                "status": status,
                "healthy": healthy,
                "data": data,
                "message": None
            }
        else:
            return {
                "status": "error",
                "healthy": False,
                "message": f"Backend returned status code {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "status": "offline",
            "healthy": False,
            "message": "Cannot connect to backend server"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "healthy": False,
            "message": "Backend server is not responding (timeout)"
        }
    except Exception as e:
        return {
            "status": "error",
            "healthy": False,
            "message": f"Error checking backend: {str(e)}"
        }


def show_maintenance_page(health_status):
    """Display a friendly maintenance page when backend is unavailable"""
    st.markdown(
        """
        <style>
        .maintenance-container {
            text-align: center;
            padding: 3rem 2rem;
        }
        .maintenance-icon {
            font-size: 5rem;
            margin-bottom: 1rem;
        }
        .maintenance-title {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #FF6B6B;
        }
        .maintenance-message {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        .status-info {
            background-color: #f8f9fa;
            border-left: 4px solid #FF6B6B;
            padding: 1rem;
            margin: 2rem auto;
            max-width: 600px;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    status = health_status.get("status", "unknown")
    message = health_status.get("message", "")

    # Determine icon and message based on status
    if status == "offline":
        icon = "🔌"
        title = "Backend Currently Offline"
        description = "The AI backend service is currently unavailable. This may be due to maintenance or resource limits on the free tier."
    elif status == "timeout":
        icon = "⏱️"
        title = "Backend Not Responding"
        description = "The backend server is taking too long to respond. It may be starting up or experiencing high load."
    elif status == "loading":
        icon = "⏳"
        title = "AI Models Loading"
        description = "The backend is currently loading AI models. This usually takes 5-10 minutes. Please check back shortly!"
    elif status == "error":
        icon = "⚠️"
        title = "Backend Service Error"
        description = "The backend encountered an error while starting up. This may be a temporary issue."
    else:
        icon = "🛠️"
        title = "Service Under Maintenance"
        description = "The Insight AI backend is currently unavailable. We're working to get it back online."

    st.markdown(
        f"""
        <div class="maintenance-container">
            <div class="maintenance-icon">{icon}</div>
            <div class="maintenance-title">{title}</div>
            <div class="maintenance-message">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show status details in expander
    with st.expander("Technical Details"):
        st.json(health_status)

        if message:
            st.info(f"**Details:** {message}")

        st.markdown("""
        **Common Reasons:**
        - Render free tier instances spin down after inactivity
        - Monthly free tier hours exhausted
        - Backend is deploying new updates
        - AI models are loading (takes 5-10 minutes)

        **What to do:**
        - Wait a few minutes and refresh the page
        - Check back later if free tier hours are exhausted
        - Contact support if the issue persists
        """)

    # Refresh button
    if st.button("🔄 Check Again", type="primary"):
        st.rerun()

    st.divider()

    st.markdown("""
    ### About Insight AI

    Insight AI is an explainable image classification tool that provides:
    - 🎯 **Predictions** - Identify 1000 ImageNet object classes
    - 🔥 **Grad-CAM** - Visual explanations of model decisions
    - 💬 **Captions** - Natural language image descriptions

    **GitHub**: [O-S-O-K/insight_ai_app](https://github.com/O-S-O-K/insight_ai_app)

    **Author**: Sheron Schley
    """)

    st.stop()

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Insight AI", layout="wide")
ROOT = Path(__file__).resolve().parent

# ----------------------------
# Load metadata for labels
# ----------------------------
METADATA_PATH = ROOT.parent / "models" / "model_metadata.json"
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    LABEL_MAP = metadata.get("classes", {})
else:
    LABEL_MAP = {}

TOP_K = 3  # number of top predictions to display

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

def display_gradcam_image(overlay_img: Image.Image, alpha: float, original_img: Image.Image):
    """Blend overlay with original using alpha and show in Streamlit"""
    original_img = original_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB").resize(original_img.size)
    blended = Image.blend(original_img, overlay_img, alpha)
    st.image(blended, caption="Grad-CAM Overlay", width="stretch")

# ----------------------------
# UI
# ----------------------------
# Check backend health before showing the main app
if not USE_MOCK:
    with st.spinner("Checking backend status..."):
        health_status = check_backend_health()

    if not health_status["healthy"]:
        show_maintenance_page(health_status)

st.title("Insight AI")
st.caption("Explainable image classification with BLIP captions and Grad-CAM visualizations")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image and handle EXIF orientation (important for mobile uploads)
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Auto-rotate based on EXIF data
    st.image(img, caption="Uploaded image", use_container_width=True)

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    col1, col2, col3 = st.columns([1, 1, 1])

    # ------------------------
    # Predict
    # ------------------------
    with col1:
        if st.button("Predict"):
            with st.spinner("Running prediction..."):
                try:
                    result = call_backend_predict(uploaded_file)
                    st.session_state.predictions = result.get("predictions", [])
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("predictions"):
        st.subheader("Top Predictions")
        for i, pred in enumerate(st.session_state.predictions, start=1):
            st.write(f"{i}. **{pred['class_name']}** - Confidence: {pred['confidence']*100:.2f}%")

    # ------------------------
    # Caption
    # ------------------------
    with col2:
        if st.button("Caption"):
            with st.spinner("Generating caption..."):
                try:
                    result = call_backend_caption(uploaded_file)
                    st.session_state.caption = result
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("caption"):
        st.subheader("Image Caption (BLIP)")
        st.write(st.session_state.caption.get("caption"))

    # ------------------------
    # Grad-CAM
    # ------------------------
    with col3:
        if st.button("Grad-CAM"):
            with st.spinner("Computing Grad-CAM for top predictions..."):
                try:
                    result = call_backend_gradcam(uploaded_file, top_k=TOP_K)
                    st.session_state.gradcams = result.get("gradcams", [])
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("gradcams"):
        st.subheader("Grad-CAM Outputs")
        alpha = st.slider("Heatmap intensity", 0.0, 1.0, 0.4, 0.05)
        tabs = st.tabs([f"{g['class_name']} ({g['confidence']*100:.1f}%)" for g in st.session_state.gradcams])
        for tab, gradcam_data in zip(tabs, st.session_state.gradcams):
            with tab:
                b64_data = gradcam_data["heatmap_base64"].split(",")[1]
                overlay_img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                display_gradcam_image(overlay_img, alpha, img)

    # ------------------------
    # Human Feedback
    # ------------------------
    st.divider()
    st.subheader("Human Feedback")

    if not st.session_state.get("feedback_submitted", False):
        feedback_text = st.text_area("Your feedback")
        rating = st.slider("Confidence score", 1, 5, 3)

        if st.button("Submit feedback"):
            entry = {"feedback": feedback_text, "rating": rating}
            try:
                post_feedback_to_backend(uploaded_file, entry)
                st.session_state.feedback_submitted = True
                st.success("Feedback submitted — thank you!")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Feedback already submitted for this image.")

else:
    st.info("Upload an image to begin.")

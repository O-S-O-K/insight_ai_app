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

    if st.button("Check Again", type="primary"):
        st.rerun()

    st.divider()
    st.markdown("""
    ### About Insight AI

    Insight AI is an explainable image classification tool that provides:
    - **Predictions** — Identify object classes with calibrated confidence
    - **Grad-CAM** — Visual explanations of model decisions
    - **SHAP** — Feature attribution maps (GradientExplainer)
    - **CLIP** — Zero-shot classification with custom labels
    - **Captions** — Natural language image descriptions (BLIP)

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
        st.session_state.predictions = None
        st.session_state.predict_meta = {}
        st.session_state.caption = None
        st.session_state.gradcams = None
        st.session_state.shap_result = None
        st.session_state.clip_results = None
        st.session_state.active_learning_flag = False
        st.session_state.flag_reason = None
        st.session_state.feedback_submitted = False
        st.session_state.feedback_count = 0


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) to a PIL Image."""
    raw = b64_str.split(",")[1] if "," in b64_str else b64_str
    return Image.open(io.BytesIO(base64.b64decode(raw)))


# ----------------------------
# Health check
# ----------------------------
if not USE_MOCK:
    with st.spinner("Checking backend status..."):
        health_status = check_backend_health()

    if not health_status["healthy"]:
        show_maintenance_page(health_status)
else:
    health_status = {"status": "ready", "healthy": True, "data": {}}

# Grab live backend info for sidebar
_health_data = health_status.get("data", {})
_backend_model_type = _health_data.get("model_type", "imagenet")
_medical_available = _health_data.get("medical_model_available", False)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.caption("Explainable Image Intelligence")
    st.divider()

    # --- Model status (read-only badge) ---
    st.subheader("Model")
    if _backend_model_type == "medical":
        st.success("Medical Imaging (EfficientNetB0)")
        st.caption("Classes: Melanoma / Benign")
    else:
        st.info("ImageNet General (MobileNetV2)")
        st.caption("1,000 ImageNet classes")

    if TEMPERATURE != 1.0:
        st.caption(f"Temperature scaling: T={TEMPERATURE}")

    # --- Functional model selector ---
    _model_options = {
        "imagenet": "ImageNet General (MobileNetV2)",
        "medical": "Medical Imaging (EfficientNetB0)",
    }
    _default_idx = 1 if _backend_model_type == "medical" else 0
    st.selectbox(
        "Active model",
        options=list(_model_options.keys()),
        format_func=lambda k: _model_options[k],
        index=_default_idx,
        key="selected_model",
        help="Selects which model the backend uses for Predict and Grad-CAM.",
    )
    if st.session_state.get("selected_model") == "medical" and not _medical_available and not USE_MOCK:
        st.warning("Medical model not deployed on backend — will fall back to ImageNet.")

    st.divider()

    # --- Recent Predictions (MLflow) ---
    col_r, col_b = st.columns([3, 1])
    with col_r:
        st.subheader("Recent Predictions")
    with col_b:
        if st.button("Refresh", key="refresh_mlflow", use_container_width=True):
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
            top_class = run.get("top1_class") or run.get("tags", {}).get("top1_class", "—")
            conf = run.get("top1_confidence") or run.get("metrics", {}).get("top1_confidence")
            endpoint = run.get("endpoint") or run.get("tags", {}).get("endpoint", "predict")
            conf_str = f" ({conf*100:.0f}%)" if conf is not None else ""
            # Human-readable: "Predicted: Cat (92%) via predict"
            if endpoint == "clip":
                st.caption(f"CLIP match: **{top_class}**{conf_str}")
            else:
                st.caption(f"Predicted: **{top_class}**{conf_str}")
    else:
        st.caption("No predictions recorded yet.")

    st.divider()

    # --- Active Learning (collapsed to save space) ---
    with st.expander("Active Learning Pipeline"):
        try:
            al_data = get_active_learning_summary()
            total = al_data.get("total_feedback", 0)
            flagged = al_data.get("flagged", 0)
            rate = al_data.get("flag_rate", 0.0)
            st.metric("Flagged / Total", f"{flagged} / {total}")
            if total > 0:
                st.progress(min(rate, 1.0), text=f"Flag rate: {rate*100:.1f}%")
            else:
                st.caption("No feedback submitted yet.")
        except Exception:
            st.caption("Data unavailable.")

# ----------------------------
# Main UI
# ----------------------------
st.title("Insight AI")
st.caption(
    "Upload an image to get AI predictions, visual explanations, and provide "
    "feedback to improve the model."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG.",
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = ImageOps.exif_transpose(img)

    img_hash = image_hash(uploaded_file)
    reset_state_on_new_image(img_hash)

    # Image preview + metadata
    col_img, col_meta = st.columns([2, 1])
    with col_img:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with col_meta:
        w, h = img.size
        st.markdown("**Image info**")
        st.caption(f"Filename: `{uploaded_file.name}`")
        st.caption(f"Dimensions: {w} × {h} px")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
        st.caption(f"Model: {_backend_model_type.title()}")

    st.divider()

    # ============================================================
    # STEP 1: ANALYZE  (Predict + Caption together)
    # ============================================================
    st.subheader("Step 1 — Analyze Image")

    if st.button("Analyze Image", type="primary", use_container_width=True):
        # Run predict and caption as sequential calls (caption is fast)
        with st.spinner("Running predictions..."):
            try:
                result = call_backend_predict(
                    uploaded_file,
                    top_k=TOP_K,
                    model_type=st.session_state.get("selected_model", _backend_model_type),
                )
                preds = result.get("predictions", [])
                st.session_state.predictions = preds
                st.session_state.predict_meta = {
                    "calibrated": result.get("calibrated", False),
                    "temperature": result.get("temperature", 1.0),
                }
                if preds and preds[0]["confidence"] < ACTIVE_LEARNING_THRESHOLD:
                    st.session_state.active_learning_flag = True
                    st.session_state.flag_reason = "low_confidence"
                # Invalidate MLflow cache so sidebar refreshes
                st.session_state.mlflow_runs = None
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        with st.spinner("Generating caption..."):
            try:
                caption_result = call_backend_caption(uploaded_file)
                st.session_state.caption = caption_result
            except Exception:
                # Caption failure is non-fatal
                st.session_state.caption = None

    # --- Show prediction results ---
    if st.session_state.get("predictions"):
        preds = st.session_state.predictions
        meta = st.session_state.get("predict_meta", {})
        calibrated = meta.get("calibrated", False)
        temp = meta.get("temperature", 1.0)

        # Low-confidence warning BEFORE the prediction list
        if (
            st.session_state.get("active_learning_flag")
            and st.session_state.get("flag_reason") == "low_confidence"
        ):
            st.warning(
                f"Low-confidence prediction (top score < {ACTIVE_LEARNING_THRESHOLD*100:.0f}%) — "
                "this image has been flagged for human review."
            )

        badge = " *(calibrated)*" if calibrated else ""
        st.markdown(f"**Top Predictions{badge}**")

        for i, pred in enumerate(preds, start=1):
            conf = pred["confidence"] * 100
            st.write(f"{i}. **{pred['class_name']}** — {conf:.1f}%")
            st.caption("Confidence:")
            st.progress(min(pred["confidence"], 1.0))

        if calibrated:
            st.caption(f"Scores adjusted via temperature scaling (T={temp})")

    # --- Show caption ---
    if st.session_state.get("caption"):
        st.markdown("**Image Caption (BLIP)**")
        st.write(f"> {st.session_state.caption.get('caption', '')}")

    # ============================================================
    # STEP 2: EXPLAINABILITY  (only after predictions)
    # ============================================================
    if st.session_state.get("predictions"):
        st.divider()
        st.subheader("Step 2 — Explainability")
        st.caption("Visualize which parts of the image drove the prediction.")

        col_gc, col_sh = st.columns(2)

        with col_gc:
            if st.button("Grad-CAM", use_container_width=True):
                with st.spinner("Computing Grad-CAM heatmaps..."):
                    try:
                        result = call_backend_gradcam(
                            uploaded_file,
                            top_k=TOP_K,
                            model_type=st.session_state.get("selected_model", _backend_model_type),
                        )
                        st.session_state.gradcams = result.get("gradcams", [])
                    except Exception as e:
                        st.error(f"Grad-CAM failed: {e}")

        with col_sh:
            if st.button("SHAP  (may take ~1 min)", use_container_width=True):
                with st.spinner("Computing SHAP attributions — this may take 1–2 minutes on free-tier CPU..."):
                    try:
                        result = shap_explain(
                            uploaded_file,
                            model_type=st.session_state.get("selected_model", _backend_model_type),
                        )
                        st.session_state.shap_result = result
                    except Exception as e:
                        st.error(f"SHAP failed: {e}")

        # Show results in tabs if either is available
        show_gradcam = bool(st.session_state.get("gradcams"))
        show_shap = bool(st.session_state.get("shap_result"))

        if show_gradcam or show_shap:
            tab_labels = []
            if show_gradcam:
                tab_labels.append("Grad-CAM")
            if show_shap:
                tab_labels.append("SHAP")

            tabs = st.tabs(tab_labels)
            tab_idx = 0

            if show_gradcam:
                with tabs[tab_idx]:
                    alpha = st.slider(
                        "Heatmap intensity", 0.0, 1.0, 0.4, 0.05, key="gradcam_alpha"
                    )
                    gcam_entries = st.session_state.gradcams
                    gcam_tab_labels = [
                        f"{g['class_name']} ({g['confidence']*100:.1f}%)" for g in gcam_entries
                    ]
                    gcam_tabs = st.tabs(gcam_tab_labels)
                    for gcam_tab, gradcam_data in zip(gcam_tabs, gcam_entries):
                        with gcam_tab:
                            b64 = gradcam_data["heatmap_base64"]
                            raw = b64.split(",")[1] if "," in b64 else b64
                            overlay_img = Image.open(io.BytesIO(base64.b64decode(raw)))
                            # Blend heatmap with original
                            original = img.convert("RGB")
                            overlay = overlay_img.convert("RGB").resize(original.size)
                            blended = Image.blend(original, overlay, alpha)
                            st.image(blended, caption="Grad-CAM Overlay", use_container_width=True)
                tab_idx += 1

            if show_shap:
                with tabs[tab_idx]:
                    shap = st.session_state.shap_result
                    st.caption(f"Method: {shap.get('method', 'SHAP GradientExplainer')}")
                    b64 = shap.get("shap_plot_base64", "")
                    if b64 and len(b64) > 30:
                        raw = b64.split(",")[1] if "," in b64 else b64
                        shap_img = Image.open(io.BytesIO(base64.b64decode(raw)))
                        st.image(shap_img, caption="SHAP Attribution Map", use_container_width=True)
                    st.info(
                        shap.get(
                            "explanation",
                            "Red regions increase the prediction; blue regions decrease it.",
                        )
                    )
                    if shap.get("top_class"):
                        st.caption(
                            f"Top predicted class: **{shap['top_class']}** "
                            f"({shap.get('top_confidence', 0)*100:.1f}%)"
                        )

    # ============================================================
    # STEP 3: FEEDBACK  (only after predictions)
    # ============================================================
    if st.session_state.get("predictions"):
        st.divider()
        preds = st.session_state.predictions

        feedback_count = st.session_state.get("feedback_count", 0)
        if feedback_count > 0:
            st.subheader(f"Step 3 — Feedback  *(submitted {feedback_count}×)*")
            st.caption("You can submit additional feedback for this image.")
        else:
            st.subheader("Step 3 — Feedback")
            st.caption(
                "Help improve the model by telling us which prediction was correct "
                "and rating the overall quality."
            )

        # Which prediction was correct?
        pred_options = [f"{p['class_name']} ({p['confidence']*100:.1f}%)" for p in preds]
        pred_options.append("None of the above")

        correct_choice = st.radio(
            "Which prediction was correct?",
            pred_options,
            index=0,
            key=f"correct_pred_{img_hash}_{feedback_count}",
        )

        correct_class = None
        if correct_choice == "None of the above":
            correct_class = st.text_input(
                "What is the correct label?",
                placeholder="e.g. melanoma, golden retriever, stop sign...",
                key=f"correct_class_input_{img_hash}_{feedback_count}",
            )
        else:
            # Extract just the class name (strip the confidence part)
            correct_class = correct_choice.rsplit(" (", 1)[0]

        feedback_text = st.text_area(
            "Additional comments (optional)",
            placeholder="e.g. The image shows a benign nevus, not melanoma. "
            "Lighting was poor which may have affected the result.",
            key=f"feedback_text_{img_hash}_{feedback_count}",
        )

        rating = st.slider(
            "Overall rating",
            1,
            5,
            3,
            help="1 = Poor  ·  2 = Fair  ·  3 = Neutral  ·  4 = Good  ·  5 = Excellent",
            key=f"rating_{img_hash}_{feedback_count}",
        )
        st.caption(f"Rating: {'★' * rating}{'☆' * (5 - rating)}  ({['', 'Poor', 'Fair', 'Neutral', 'Good', 'Excellent'][rating]})")

        # Auto-flag checkbox — explain if pre-checked
        auto_flagged = st.session_state.get("active_learning_flag", False)
        flag_label = (
            "Flag for human review  *(auto-flagged — low confidence)*"
            if auto_flagged
            else "Flag for human review"
        )
        manual_flag = st.checkbox(
            flag_label,
            value=auto_flagged,
            help="Flagged images are added to the active learning queue for retraining.",
            key=f"manual_flag_{img_hash}_{feedback_count}",
        )

        if st.button("Submit Feedback", key=f"submit_feedback_{img_hash}_{feedback_count}"):
            top_conf = preds[0]["confidence"] if preds else None
            top_class = preds[0]["class_name"] if preds else None
            flag_reason = (
                st.session_state.get("flag_reason")
                if auto_flagged
                else ("user_flagged" if manual_flag else None)
            )

            entry = {
                "feedback": feedback_text,
                "rating": rating,
                "correct_class": correct_class,
                "active_learning_flag": manual_flag,
                "flag_reason": flag_reason,
                "top1_confidence": top_conf,
                "predicted_class": top_class,
            }
            try:
                post_feedback_to_backend(uploaded_file, entry)
                st.session_state.mlflow_runs = None
                st.session_state.feedback_submitted = True
                st.session_state.feedback_count = feedback_count + 1
                st.success("Feedback submitted — thank you!")
                st.rerun()
            except Exception as e:
                st.error(f"Feedback submission failed: {e}")

    # ============================================================
    # STEP 4: CLIP ZERO-SHOT CLASSIFICATION
    # ============================================================
    st.divider()
    st.subheader("Step 4 — Zero-Shot Classification (CLIP)")
    st.info(
        "CLIP (Contrastive Language-Image Pretraining) can classify your image against "
        "any labels you provide — no training required. Enter comma-separated labels "
        "below and CLIP will rank them by visual similarity."
    )

    clip_input = st.text_input(
        "Custom labels (comma-separated)",
        placeholder="e.g. melanoma, benign lesion, normal skin, dermatitis",
        key="clip_label_input",
    )

    if st.button("Classify with CLIP", use_container_width=True, type="primary"):
        labels = [lbl.strip() for lbl in clip_input.split(",") if lbl.strip()]
        if not labels:
            st.warning("Enter at least one label before classifying.")
        elif len(labels) > 20:
            st.warning("Please enter 20 or fewer labels.")
        else:
            with st.spinner("Running CLIP zero-shot classification..."):
                try:
                    result = clip_classify(uploaded_file, labels)
                    st.session_state.clip_results = result
                    st.session_state.mlflow_runs = None
                except Exception as e:
                    st.error(f"CLIP classification failed: {e}")

    if st.session_state.get("clip_results"):
        clip_data = st.session_state.clip_results
        results = clip_data.get("results", [])
        model_name = clip_data.get("model", "CLIP ViT-B/32")
        st.caption(f"Model: {model_name}")
        if results:
            for r in results:
                score_pct = r["score"] * 100
                st.write(f"**{r['label']}** — {score_pct:.1f}%")
                st.caption("Similarity score:")
                st.progress(min(r["score"], 1.0))

        st.info(
            "Your CLIP classifications and label corrections are added to the active "
            "learning queue and will inform future model retraining."
        )

else:
    st.info("Upload an image above to begin analysis.")

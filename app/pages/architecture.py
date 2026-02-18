import streamlit as st

st.set_page_config(page_title="Architecture · Insight AI", layout="centered")

st.title("Architecture Overview")
st.caption("How data flows through Insight AI — from image upload to human-readable explanation, with MLops tracking and a human feedback loop.")

st.markdown(
    """
    ```text
    ┌─────────────────────────────────────────────────────────────┐
    │                    Streamlit Frontend                       │
    │              (Streamlit Cloud · free tier)                  │
    └───────────────────────┬─────────────────────────────────────┘
                            │  REST API  (multipart/form-data)
                            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   FastAPI Backend  /api/v1                  │
    │          (Hugging Face Spaces · Docker · port 7860)         │
    │                                                             │
    │  User Image ──▶ Preprocessing (resize 224×224, normalize)  │
    │                        │                                    │
    │         ┌──────────────┼──────────────┬──────────────┐     │
    │         ▼              ▼              ▼              ▼     │
    │    CNN Predict    Grad-CAM        SHAP Explainer   BLIP    │
    │   (MobileNetV2    (heatmap        (GradientExpl.  Caption  │
    │    or EfficientB0  overlay)        ranked_outputs=1)       │
    │    + calibration)                                          │
    │         │              │              │              │     │
    │         └──────────────┴──────────────┴──────────────┘     │
    │                        │                                    │
    │              CLIP Zero-Shot (optional)                      │
    │        (ViT-B/32 · user-defined text labels)               │
    │                        │                                    │
    │              MLflow Experiment Logging                      │
    │          (params, metrics, tags per request)               │
    │                        │                                    │
    │              Active Learning Flagging                       │
    │          (auto-flag if confidence < 0.5)                   │
    │                        │                                    │
    └────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
                    User Feedback Form
               (rating, correction, flag)
                             │
                             ▼
                  feedback_log.json  ↺
           (persisted · influences future sessions)
    ```
    """
)

st.divider()

st.subheader("Component Breakdown")

with st.expander("1  Image Preprocessing", expanded=True):
    st.markdown(
        """
        - Uploaded images are decoded and converted to RGB (PIL)
        - Resized to 224×224 to match CNN input requirements
        - Normalized per model type: MobileNetV2 uses `preprocess_input` (scale to [-1, 1]),
          EfficientNetB0 expects [0, 255] (normalization internal to the model)
        - Shared preprocessing pipeline across predict, Grad-CAM, SHAP, and CLIP endpoints
        """
    )

with st.expander("2  CNN Prediction + Confidence Calibration", expanded=False):
    st.markdown(
        """
        - **ImageNet model**: MobileNetV2 pretrained on ImageNet-1K (1000 classes, ~71.8% top-1)
        - **Medical model**: EfficientNetB0 fine-tuned on ISIC 2020 skin lesion dataset (binary: benign/melanoma)
        - Model selected via `MODEL_TYPE` environment variable (`imagenet` or `medical`)
        - **Temperature scaling**: raw logits divided by T=1.5 (imagenet) or T=1.2 (medical)
          before softmax — reduces overconfidence typical of pre-trained networks
        - Returns top-K predictions with calibrated confidence scores
        - Background loading in a separate thread — server accepts requests immediately at startup
        """
    )

with st.expander("3  Grad-CAM Explainability", expanded=False):
    st.markdown(
        """
        - Gradient-weighted Class Activation Mapping (Grad-CAM) highlights regions that
          most influenced the predicted class
        - Hooks into the last convolutional layer (`Conv_1` for MobileNetV2)
        - Produces a heatmap overlaid on the original image at 0.4 alpha blend
        - Supports top-K classes — returns one heatmap per predicted class
        - Gradient validation and division-by-zero protection prevent silent failures
        """
    )

with st.expander("4  SHAP GradientExplainer", expanded=False):
    st.markdown(
        """
        - Uses `shap.GradientExplainer` — best compatibility with TensorFlow/Keras functional models
        - Background dataset: 10 random samples (224×224×3), generated once and cached as
          `models/shap_background.npy` on first startup
        - `ranked_outputs=1` limits computation to the top-1 predicted class only,
          reducing gradient passes from 1000 → 1 (~1000× speedup on CPU)
        - Output: signed feature attribution map (RdBu_r colormap) blended with original image
        - Red regions = pixels that pushed toward the predicted class
        - Blue regions = pixels that suppressed the predicted class
        """
    )

with st.expander("5  BLIP Captioning (Vision → Language)", expanded=False):
    st.markdown(
        """
        - `Salesforce/blip-image-captioning-base` loaded via HuggingFace Transformers
        - Generates a free-form natural-language description of the image content
        - Model (~1GB) downloaded and cached in the Docker image at build time for fast startup
        - Compatible with PyTorch 2.1.2 + Transformers 4.35.2
        - Optional — disabled gracefully if `ENABLE_BLIP=false`
        """
    )

with st.expander("6  CLIP Zero-Shot Classification", expanded=False):
    st.markdown(
        """
        - `openai/clip-vit-base-patch32` via HuggingFace Transformers
        - Encodes the image and a user-supplied list of text labels into a shared embedding space
        - Returns cosine similarity scores for each label — no training required
        - Works on any concept the user can describe in text
        - Scores are relative (softmax over provided labels), not absolute probabilities
        """
    )

with st.expander("7  MLflow Experiment Tracking", expanded=False):
    st.markdown(
        """
        - Local file-based tracking (`file:./mlruns`) — no external server required
        - Every `/predict`, `/shap`, `/clip` call is logged as a named MLflow run
        - Logged per request: model type, top-1 class, confidence, temperature, endpoint name
        - `/api/v1/mlflow/recent-runs` exposes the last N runs for the frontend sidebar
        - Runs are ephemeral on HF Spaces (container restarts reset them) — acceptable for demo
        """
    )

with st.expander("8  Active Learning & Feedback Loop", expanded=False):
    st.markdown(
        """
        - Predictions with top-1 confidence < 0.5 are automatically flagged in the UI
        - Users can manually flag any prediction via checkbox
        - Feedback entries include: rating (1–5), correction text, `active_learning_flag`,
          `flag_reason` (low_confidence | user_flagged), and `top1_confidence`
        - All entries appended to `feedback_log.json` with timestamped image saves
        - `/api/v1/active-learning/summary` returns total flags and flag rate
        """
    )

st.divider()

st.subheader("Design Principles")

st.markdown(
    """
    - **Explainability-first** — every prediction is paired with at least one explanation method
    - **Modularity** — each component (SHAP, CLIP, MLflow, medical model) is independently toggled via env vars
    - **Production-aware** — non-blocking startup, Docker containerization, CI/CD via GitHub Actions
    - **Responsible AI** — model card documenting bias, limitations, and out-of-scope uses
    - **Observable** — health endpoint exposes real-time model loading status per component
    - **Human-centered** — outputs designed to be interpretable, not just accurate
    """
)

st.divider()

st.subheader("Deployment Architecture")

st.markdown(
    """
    **Frontend — Streamlit Cloud**
    - Lightweight Streamlit UI (no ML dependencies)
    - Calls backend via `INSIGHT_BACKEND_URL` secret
    - Falls back to mock API client if backend is unreachable
    - `app/requirements.txt`: streamlit, requests, pillow only

    **Backend — Hugging Face Spaces (Docker)**
    - FastAPI + Uvicorn on port 7860
    - All ML models loaded at container startup in a background thread
    - `api/requirements.txt`: TensorFlow, PyTorch, Transformers, SHAP, MLflow
    - BLIP model baked into Docker image at build time (~1GB)

    **CI/CD — GitHub Actions**
    - Trigger: push to `main` touching `api/**` or `models/**`, or manual `workflow_dispatch`
    - Clones HF Space repo, syncs files, configures Git LFS for binary models, pushes
    - HF Spaces detects the push and rebuilds the Docker image automatically

    **API versioning**
    - All endpoints under `/api/v1/` prefix
    - Health check at root `/` (unversioned)
    """
)

st.divider()

st.caption("Author: Sheron Schley · github.com/O-S-O-K/insight_ai_app")
st.caption("© Insight AI · Architecture Diagram & Design")
st.caption("Live demo: https://insight-ai-v1.streamlit.app · Backend: https://o-s-o-k-insight-ai-backend.hf.space")

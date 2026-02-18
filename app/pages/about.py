import streamlit as st

st.set_page_config(page_title="About · Insight AI", layout="centered")

st.title("About Insight AI")

st.markdown(
    """
    **Insight AI** is an interactive, explainable computer vision application that bridges the gap between
    *model predictions* and *human understanding*. Upload any image and receive a multi-modal
    explanation of what the model saw — and why it made that decision.

    **What you can do:**
    - **Classify** images with a pretrained CNN (MobileNetV2, ImageNet-1K or EfficientNetB0, medical)
    - **Explain** predictions visually with Grad-CAM heatmaps and SHAP feature attribution
    - **Caption** images in natural language using BLIP (vision → language)
    - **Compare** predictions against any custom labels using CLIP zero-shot classification
    - **Flag** uncertain predictions for human review (active learning)
    - **Track** every inference run automatically via MLflow experiment logging
    - **Submit feedback** to log corrections and ratings for future model improvement
    """
)

st.divider()

st.subheader("Feature Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Explainability**
        - Grad-CAM — visual heatmap overlay showing where the model looked
        - SHAP GradientExplainer — signed pixel-level feature attribution
        - Calibrated confidence — temperature scaling (T=1.5) reduces overconfidence

        **Vision–Language**
        - BLIP — generates a natural-language caption for any image
        - CLIP zero-shot — rank user-defined text labels by image similarity
        """
    )

with col2:
    st.markdown(
        """
        **MLOps & Feedback**
        - MLflow — every prediction logged with metrics and parameters
        - Active learning — low-confidence predictions auto-flagged for review
        - Feedback loop — corrections stored in JSON, no retraining required

        **Models**
        - ImageNet General — MobileNetV2, 1000 classes, ~71.8% top-1 accuracy
        - Medical Imaging — EfficientNetB0 fine-tuned on ISIC 2020 skin lesions
        """
    )

st.divider()

st.subheader("Human-in-the-Loop Design")

st.markdown(
    """
    Rather than retraining models online, Insight AI captures **structured supervision at inference time**.

    - Prediction correctness is validated by the user after each inference
    - Low-confidence predictions (< 50%) are automatically flagged for review
    - Users can manually flag any prediction regardless of confidence
    - Corrections and ratings are persisted in `feedback_log.json`
    - Active learning flags surface uncertain samples for future labeling

    This design prioritizes **safety, auditability, and explainability** while
    preserving the stability of deployed models.
    """
)

st.divider()

st.subheader("Tech Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        **Frontend**
        - Streamlit 1.53.0
        - Requests 2.32+
        - Pillow 10.3+
        - Streamlit Cloud (hosting)
        """
    )

with col2:
    st.markdown(
        """
        **Backend**
        - FastAPI 0.110.2 + Uvicorn
        - TensorFlow 2.15.0 (CPU)
        - tf-keras 2.15 (Keras 2 compat)
        - PyTorch 2.1.2
        - Transformers 4.35.2
        - SHAP 0.44+
        - MLflow 2.12+
        - Hugging Face Spaces (hosting)
        """
    )

with col3:
    st.markdown(
        """
        **Infrastructure**
        - Docker (containerized backend)
        - GitHub Actions (CI/CD deploy)
        - Git LFS (binary model files)
        - JSON (metadata + feedback logs)
        """
    )

st.divider()

st.subheader("Project Motivation")

st.markdown(
    """
    Many machine learning applications stop at prediction accuracy.
    Insight AI goes a step further by asking:

    > *Why did the model make this decision — and how can a human meaningfully respond to it?*

    By combining visual explanations (Grad-CAM, SHAP), language-based reasoning (BLIP, CLIP),
    experiment tracking (MLflow), and a human-in-the-loop feedback mechanism, this project
    demonstrates how explainable AI systems can be both **transparent** and **interactive**
    in real-world deployment settings.

    **Best practices demonstrated:**
    - Explainable AI (XAI) — Grad-CAM, SHAP, CLIP, confidence calibration
    - MLOps — MLflow experiment tracking, active learning flagging
    - Production ML — non-blocking model loading, Docker, CI/CD, API versioning
    - Responsible AI — model card, bias documentation, clinical disclaimers
    - Frontend/backend separation — independent scaling, cost-effective free-tier hosting
    """
)

st.divider()

st.caption("Author: Sheron Schley · github.com/O-S-O-K/insight_ai_app")
st.caption("© Insight AI · Explainable Vision with Human-in-the-Loop Feedback")
st.caption("Live demo: https://insight-ai-v1.streamlit.app · Backend: https://o-s-o-k-insight-ai-backend.hf.space")

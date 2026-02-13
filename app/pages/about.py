import streamlit as st

st.set_page_config(page_title="About · Insight AI", layout="centered")

st.title("🧠 About Insight AI")

st.markdown(
    """
    **Insight AI** is an interactive, explainable computer vision application designed to bridge the gap between
    *model predictions* and *human understanding*.

   **How it works:**
   - Users upload an image and receive:
      - 🔍 A **model prediction** from a pretrained CNN (MobileNetV2)
      - 🖼️ A **visual explanation** using Grad-CAM heatmaps
      - 📝 An **optional natural-language caption** (BLIP vision–language model)
      - 💡 A **human-friendly insight layer** mapping captions/predictions to interpretable concepts

    **Human-in-the-loop feedback:**
   - Users validate/correct predictions and captions
   - Feedback is logged and used to influence future inference (dynamic mapping, no retraining)
   - Ratings (1-5 scale) provide confidence signals for system improvement

   **Architecture:**
   - The app is split into a lightweight Streamlit frontend and a FastAPI backend (for heavy inference)
   - The backend can run locally or on Render (cloud), and the frontend can be deployed on Streamlit Cloud
   - This separation enables free, reliable hosting and scalable inference
   - Robust model loading with automatic SavedModel to H5 fallback prevents deployment errors
    """
)

st.divider()

st.subheader("🔧 Architecture Overview")

st.markdown(
    """
    The application is structured as a modular, production-aware ML system:

    1. **Image Preprocessing**
       Uploaded images are resized and normalized for compatibility across models.

    2. **CNN Prediction**
       A pretrained CNN produces class probabilities for the input image with top-K results.

    3. **Grad-CAM Explainability**
       Gradient-weighted class activation maps highlight regions of the image
       that most influenced the model's prediction. Includes validation to prevent errors.

    4. **BLIP Captioning (Vision → Language)**
       A BLIP vision–language model generates a natural-language description
       of the image content.

    5. **Insight Mapping Layer**
       Captions and predictions are mapped to domain-relevant keywords to
       produce human-readable explanations.

    6. **User Feedback Loop**
       Users validate predictions and captions, optionally correct labels,
       and provide comments. Feedback is logged and used to dynamically
       influence future inference behavior via a lightweight mapping layer,
       without retraining the underlying models.
    """
)

st.divider()

st.subheader("🔁 Human-in-the-Loop Design")

st.markdown(
    """
    Rather than retraining models online, Insight AI captures **structured supervision at inference time**.

    - Prediction correctness is validated by the user
    - Caption quality is independently evaluated
    - Corrections are persisted in JSON and CSV logs
    - Future sessions benefit from prior feedback through dynamic mapping

    This design prioritizes **safety, auditability, and explainability** while
    preserving the stability of deployed models.
    """
)

st.divider()

st.subheader("⚡ Recent Improvements")

st.markdown(
    """
    **Latest Updates:**

    - ✅ **Robust Model Loading**: Automatic fallback from SavedModel to H5 format
    - ✅ **Error Prevention**: Comprehensive validation prevents `_UserObject` loading issues
    - ✅ **Deployment Hardening**: Enhanced Dockerfile with build-time validation
    - ✅ **Better Diagnostics**: Health endpoint returns model type, layer count, TF version
    - ✅ **Gradient Safety**: Added None-gradient handling in Grad-CAM
    - ✅ **Clear Logging**: Startup messages show model loading status and fallbacks
    - ✅ **Bug Fixes**: UTF-16 encoding fixed, missing imports added, mock API synchronized
    """
)

st.divider()

st.subheader("🚀 Tech Stack")

st.markdown(
    """
    **Frontend:**
    - **Streamlit 1.53.0** - Web UI framework
    - **Requests 2.32+** - HTTP client for API calls
    - **Pillow 10.3+** - Image processing

    **Backend:**
    - **FastAPI 0.110.2** - REST API framework
    - **Uvicorn 0.23.2** - ASGI server
    - **TensorFlow 2.10.1** - CNN inference (CPU-optimized)
    - **PyTorch 2.1.2** - BLIP model
    - **Transformers 4.52+** - Vision-language models
    - **NumPy, Matplotlib, OpenCV** - Numerical & image processing

    **Infrastructure:**
    - **Docker & Docker Compose** - Containerization
    - **Render.com** - Backend hosting
    - **Streamlit Cloud** - Frontend hosting

    **Data:**
    - **JSON** - Dynamic mappings and model metadata
    - **CSV** - Feedback audit logs
    """
)

st.divider()

st.subheader("📌 Project Motivation")

st.markdown(
    """
    Many machine learning applications stop at prediction accuracy.
    Insight AI goes a step further by asking:

    > *Why did the model make this decision — and how can a human meaningfully respond to it?*

    By combining visual explanations, language-based reasoning, and a
    human-in-the-loop feedback mechanism, this project demonstrates how
    explainable AI systems can be both **transparent** and **interactive**
    in real-world deployment settings.

    This project shows modern best practices in:
    - Explainable AI (XAI)
    - Production ML deployment
    - Frontend/backend separation
    - Robust error handling
    - Human-AI collaboration
    """
)

st.divider()

st.caption("Author: Sheron Schley | github.com/O-S-O-K")
st.caption("© Insight AI · Explainable Vision with Human-in-the-Loop Feedback")
st.caption("🌐 Live Demo: https://insight-ai-v1.streamlit.app")

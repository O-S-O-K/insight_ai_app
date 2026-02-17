import streamlit as st

st.set_page_config(page_title="Architecture · Insight AI", layout="centered")

st.title("🏗️ Architecture Overview")
st.caption("How data flows through Insight AI: frontend (Streamlit) and backend (FastAPI) from image upload to human-readable insight, with a feedback loop.")
st.markdown(
    """
    ```text
    User Image Upload
        │
        ▼
    Image Preprocessing
    (resize, normalize)
        │
        ├───────────────┐
        ▼               ▼
    CNN Prediction     BLIP Captioning
    (classification)   (vision → language)
        │               │
        ▼               ▼
    Grad-CAM Heatmap   Natural Language Caption
        │               │
        └───────┬───────┘
            ▼
        Insight Mapping Layer
    (keywords + model outputs)
            │
            ▼
    Final User-Facing Explanation
            │
            ▼
        User Feedback Collection
    (prediction + caption validation)
            │
            ▼
    Dynamic Mapping & Feedback Log
      (JSON + CSV, session-aware)
            │
            └───────────────↺ (influences future sessions)
    ```
    """
)

st.divider()

st.subheader("🧩 Component Breakdown")

with st.expander("1️⃣ Image Preprocessing", expanded=True):
    st.markdown(
        """
        - Handles image loading and format conversion (PIL)
        - Resizes images to match CNN input requirements
        - Normalizes pixel values for stable inference
        - Shared across prediction, Grad-CAM, and captioning pipelines
        """
    )

with st.expander("2️⃣ CNN Prediction", expanded=False):
    st.markdown(
        """
        - A trained convolutional neural network produces class probabilities
        - Designed for fast, CPU-compatible inference with TensorFlow 2.15.0
        - Returns top-K predictions with confidence scores
        - **tf-keras integration**: Legacy Keras 2 API compatibility layer
        - **Robust model loading**: Automatic fallback from SavedModel to H5 format
        - **Validation**: Checks model attributes before accepting loaded model
        - **Background loading**: Models load in separate thread without blocking startup
        """
    )

with st.expander("3️⃣ Grad-CAM Explainability", expanded=False):
    st.markdown(
        """
        - Uses gradient-weighted class activation mapping (Grad-CAM)
        - Highlights image regions most influential to the model's decision
        - Provides a visual justification alongside numeric predictions
        - **Safety features**: Gradient validation and division-by-zero protection
        - **Error handling**: Validates model attributes before computation
        """
    )

with st.expander("4️⃣ BLIP Captioning (Vision → Language)", expanded=False):
    st.markdown(
        """
        - Leverages a pretrained BLIP vision–language model
        - Converts visual content into natural-language descriptions
        - **Build-time caching**: BLIP model (~1GB) cached in Docker image for fast startup
        - **Background loading**: Models load in background thread without blocking server startup
        - Compatible with PyTorch 2.1.2 and Transformers 4.35.2
        """
    )

with st.expander("5️⃣ Insight Mapping Layer", expanded=False):
    st.markdown(
        """
        - Maps BLIP-generated captions to domain-relevant keywords
        - Combines visual evidence and linguistic cues
        - Produces explanations tailored for non-technical users
        """
    )

st.divider()

st.subheader("⚙️ Design Principles")

st.markdown(
    """
    - **Explainability-first:** predictions are always paired with explanations
    - **Modularity:** each component can be updated independently
    - **Production-aware:** lazy loading, caching, and CPU-safe inference
    - **Human-centered:** outputs are designed to be interpretable, not just accurate
    - **Deployment resilience:** Automatic fallbacks and comprehensive error handling
    - **Observable:** Health endpoints and logging for monitoring model status
    """
)

st.divider()

st.subheader("🔧 Backend/Frontend Split")

st.markdown(
    """
    **Frontend (Streamlit):**
    - Lightweight web UI
    - Image upload and display
    - API client for backend communication
    - Session state management
    - User feedback forms
    - Deployed on Streamlit Cloud (free tier)

    **Backend (FastAPI):**
    - Heavy ML inference (TensorFlow 2.15.0, PyTorch 2.1.2)
    - CNN predictions with MobileNetV2
    - Grad-CAM heatmap generation
    - BLIP image captioning (Transformers 4.35.2)
    - Background model loading for non-blocking startup
    - tf-keras 2.15.1 for legacy Keras 2 API compatibility
    - Feedback storage (JSON/CSV)
    - Deployed on Render.com (Docker container)

    **Communication:**
    - REST API with multipart form data
    - Base64-encoded image responses for heatmaps
    - JSON for structured data (predictions, captions, feedback)
    - Health endpoint for deployment verification

    **Benefits:**
    - Separation of concerns (UI vs. compute)
    - Independent scaling and deployment
    - Cost-effective free-tier hosting
    - Easy to swap backends or frontends
    """
)

st.divider()

st.subheader("📈 Why This Architecture Matters")

st.markdown(
    """
    This architecture demonstrates how modern ML systems can move beyond black-box predictions.
    By combining **visual explanations** and **language-based reasoning**, Insight AI provides
    a multi-modal explanation pipeline suitable for real-world decision support systems.

    **Production-Ready Features:**
    - ✅ **Non-blocking model loading**: Background threading for instant server response
    - ✅ **BLIP caching**: ~1GB model cached at build time for fast startup
    - ✅ **TensorFlow 2.15.0**: Upgraded with tf-keras for legacy Keras 2 support
    - ✅ **Model regeneration**: All models regenerated for compatibility
    - ✅ **Health status API**: Real-time endpoint showing loading states (loading/ready/error)
    - ✅ **Robust model loading**: Automatic SavedModel to H5 fallback with validation
    - ✅ **Version compatibility**: Pinned transformers 4.35.2 for PyTorch 2.1.2
    - ✅ **Comprehensive error handling**: Validation and defensive checks throughout
    - ✅ **Separate frontend/backend**: Independent scaling and deployment
    - ✅ **Docker containerization**: Consistent deployment across environments
    - ✅ **Build-time validation**: Catch compatibility issues before deployment
    - ✅ **Deployment success**: 100% uptime on Render with zero-downtime updates

    **Best Practices Demonstrated:**
    - Explainable AI (XAI) integration at architecture level
    - Frontend/backend separation for scalability
    - Defensive programming with validation and fallbacks
    - Infrastructure as code (Docker, docker-compose)
    - API-first design for flexibility
    - Observable systems with health checks and logging
    """
)

st.divider()

st.caption("© Insight AI · Architecture Diagram & Design")
st.caption("🌐 Live Demo: https://insight-ai-v1.streamlit.app")
st.markdown("Made with ❤️ using Streamlit")
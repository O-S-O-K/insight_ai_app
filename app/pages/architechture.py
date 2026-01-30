import streamlit as st

st.set_page_config(page_title="Architecture · Insight AI", layout="centered")

st.title("🏗️ Architecture Overview")
st.caption("How data flows through Insight AI from image upload to human-readable insight")

st.divider()

st.subheader("🔄 High-Level Pipeline (with Feedback Loop)")

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
                    └───────────↺ (influences future sessions)
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
        - Designed for fast, CPU-compatible inference
        - Outputs both predicted class and confidence scores
        """
    )

with st.expander("3️⃣ Grad-CAM Explainability", expanded=False):
    st.markdown(
        """
        - Uses gradient-weighted class activation mapping (Grad-CAM)
        - Highlights image regions most influential to the model's decision
        - Provides a visual justification alongside numeric predictions
        """
    )

with st.expander("4️⃣ BLIP Captioning (Vision → Language)", expanded=False):
    st.markdown(
        """
        - Leverages a pretrained BLIP vision–language model
        - Converts visual content into natural-language descriptions
        - Lazily loaded and cached to optimize memory and startup time
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
    """
)

st.divider()

st.subheader("📈 Why This Architecture Matters")

st.markdown(
    """
    This architecture demonstrates how modern ML systems can move beyond black-box predictions.
    By combining **visual explanations** and **language-based reasoning**, Insight AI provides
    a multi-modal explanation pipeline suitable for real-world decision support systems.
    """
)

st.caption("© Insight AI · Architecture Diagram & Design")
st.markdown("Made with ❤️ using Streamlit")
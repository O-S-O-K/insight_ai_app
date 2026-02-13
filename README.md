# InsightAI: Interactive Image Classification with Feedback

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-v1.27-orange)](https://streamlit.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://insight-ai-v1.streamlit.app) [![Last Commit](https://img.shields.io/github/last-commit/O-S-O-K/insight_ai_app)](https://github.com/O-S-O-K/insight_ai_app)

### An interactive Streamlit app combining CNN predictions, Grad-CAM explanations, optional BLIP captions, and human-in-the-loop feedback for smarter, interpretable AI.

InsightAI is an interactive explainable AI application that:
- Classifies images using a CNN (MobileNetV2, pretrained or fine-tuned)
- Visualizes model attention with Grad-CAM heatmaps
- Optionally generates natural-language image captions (BLIP)
- Allows human-in-the-loop feedback for improving predictions
- Runs fully in the browser (mobile-friendly)

👉 Try it live: **no install required**

**InsightAI** is an end-to-end Explainable AI (XAI) system that combines:
- **Deep learning (CNNs)**
- **Explainability (Grad-CAM)**
- **Vision-language models (BLIP, optional)**
- **Human-in-the-loop feedback**
- **Top-3 prediction selection & "Other" class input**
- **Dynamic, user-driven Grad-CAM overlays**
- **Interactive Streamlit interface**

The result is a project that demonstrates not only *model accuracy*, but *model understanding, inspection, and improvement over time*.

---

## Table of Contents

- [InsightAI: Interactive Image Classification with Feedback](#insightai-interactive-image-classification-with-feedback)
    - [An interactive Streamlit app combining CNN predictions, Grad-CAM explanations, optional BLIP captions, and human-in-the-loop feedback for smarter, interpretable AI.](#an-interactive-streamlit-app-combining-cnn-predictions-grad-cam-explanations-optional-blip-captions-and-human-in-the-loop-feedback-for-smarter-interpretable-ai)
  - [Table of Contents](#table-of-contents)
  - [Project Motivation](#project-motivation)
  - [High-Level System Overview](#high-level-system-overview)
  - [Key Features](#key-features)
  - [Recent Improvements](#recent-improvements)
  - [Model Architecture](#model-architecture)
    - [CNN Image Classifier](#cnn-image-classifier)
  - [Datasets Used](#datasets-used)
    - [ImageNet / Fine-Tuned Dataset](#imagenet--fine-tuned-dataset)
    - [User-Provided Images (Inference)](#user-provided-images-inference)
  - [Explainability: Grad-CAM](#explainability-grad-cam)
  - [Vision-Language Integration (BLIP, Optional)](#vision-language-integration-blip-optional)
  - [Dynamic Class Mapping](#dynamic-class-mapping)
  - [Backend/Frontend Architecture](#backendfrontend-architecture)
  - [Installation](#installation)
  - [Tech Stack](#tech-stack)
  - [Why This Project Matters](#why-this-project-matters)
  - [Example Test Session (End-to-End)](#example-test-session-end-to-end)
  - [🌍 Deployment](#-deployment)
  - [License](#license)

---

## Project Motivation

Most entry-level ML projects stop at:  
> “Here is my model accuracy.”

**InsightAI** goes further and asks:
- Why did the model make this prediction?
- Does the model’s reasoning align with human intuition?
- Can users correct the model if predictions are wrong?
- How do vision and language models complement each other?

It demonstrates *real-world ML system thinking*, integrating explainability, human feedback, and semantic understanding.

---

## High-Level System Overview

**Pipeline Overview:**

User Image Upload → Image Preprocessing (resize, normalize) → CNN Prediction & BLIP Captioning → Grad-CAM Heatmap & Natural Language Caption → Insight Mapping Layer (keywords + model outputs) → Final User-Facing Explanation → User Feedback Collection → Dynamic Mapping & Feedback Log (JSON + CSV, session-aware) → influences future sessions.

This design shows how production ML systems can evolve over time, not remain static.

---

## Key Features

- Interactive Streamlit web interface
- Real-time CNN image classification with top-3 predictions
- Dynamic Grad-CAM heatmap visualization
- Optional BLIP-generated captions for semantic understanding
- Keyword-to-class mapping
- Human-in-the-loop feedback with top-3 selection and "Other" input
- Session-safe prediction and Grad-CAM overlays
- Persistent feedback logging for future semantic alignment
- Modular, production-style codebase
- Robust model loading with automatic H5/SavedModel fallback
- Comprehensive error handling and validation

---

## Recent Improvements

**Deployment & Performance Optimizations (Latest)**
- ✅ **Non-Blocking Model Loading**: Models load in background thread for instant port binding
- ✅ **BLIP Caching**: BLIP model (~1GB) cached in Docker image at build time for fast startup
- ✅ **TensorFlow 2.15.0**: Upgraded from 2.10.1 with tf-keras for legacy Keras 2 support
- ✅ **Model Regeneration**: All models regenerated for TensorFlow 2.15.0/tf-keras compatibility
- ✅ **Background Threading**: Server responds immediately while models load asynchronously
- ✅ **Health Status**: Health endpoint shows model loading status (loading/ready/error)
- ✅ **Deployment Success**: Successfully deployed on Render with 100% uptime

**Model Loading & Compatibility**
- ✅ **Robust Model Loading**: Automatic fallback from SavedModel to H5 format with validation
- ✅ **tf-keras Integration**: Added tf-keras 2.15.1 for legacy Keras 2 API compatibility
- ✅ **Error Handling**: Comprehensive error handling with detailed logging
- ✅ **Transformers Compatibility**: Pinned transformers==4.35.2 for torch 2.1.2 compatibility

**Infrastructure & DevOps**
- ✅ **Dockerfile Optimization**: Build-time model downloads and validation
- ✅ **Keras 3 Support**: Fallback imports for both Keras 3 and legacy tf.keras
- ✅ **Debug Output**: Comprehensive startup logging for troubleshooting
- ✅ **Better Error Messages**: Clear failure messages with full tracebacks

**Bug Fixes**
- Fixed UTF-16 encoding issues in `requirements.txt` files
- Added missing `Model` import in `api/main.py` for Grad-CAM functionality
- Fixed mock API client response format to match real API
- Removed duplicate content in architecture page
- Created `.streamlit/config.toml` and example secrets file
- Removed deprecated `safe_mode` parameter from model loading
- Fixed `batch_shape` incompatibility by regenerating models

---

## Model Architecture

### CNN Image Classifier

- Architecture: MobileNetV2 (pretrained on ImageNet, optionally fine-tuned)
- Input shape: (224, 224, 3)
- Convolution + Depthwise separable blocks
- Dense fully connected layers
- Softmax output layer
- Optimizer: Adam, Loss: Categorical Crossentropy

This architecture emphasizes **system design, explainability, and feedback integration** over raw model complexity.

---

## Datasets Used

### ImageNet / Fine-Tuned Dataset
- Used to train or fine-tune the CNN
- Standardized image resizing and normalization

### User-Provided Images (Inference)
- Arbitrary real-world images uploaded by users
- Resized and normalized for inference
- Used for prediction, Grad-CAM, and feedback logging
- No retraining occurs from these images

---

## Explainability: Grad-CAM

- Visualizes where the CNN focuses when making predictions
- Heatmaps update dynamically for **user-selected labels**
- Helps verify model focus and detect spurious correlations
- Adjustable heatmap intensity slider

---

## Vision-Language Integration (BLIP, Optional)

- Generates captions describing uploaded images
- Adds semantic context beyond class labels
- Supports keyword-to-class mapping and feedback alignment

---

## Dynamic Class Mapping

- BLIP captions are parsed into keywords mapped to possible classes
- User corrections (via top-3 or "Other") update session mapping
- Feedback persists in CSV and JSON
- Grad-CAM overlays reflect user-selected labels
- No model retraining required

---


## Backend/Frontend Architecture

This project is split into two main components:
- **Frontend:** Streamlit app (see this README)
- **Backend:** FastAPI service for heavy inference (see [api/README.md](api/README.md))

For backend-specific setup, deployment, and API details, see [api/README.md](api/README.md).

---

## Installation

### Quick Start (Local Development)

**Prerequisites:**
- Python 3.10
- Git

**Option 1: Docker Compose (Recommended)**

Run both frontend and backend together:
```bash
git clone https://github.com/O-S-O-K/insight_ai_app.git
cd insight_ai_app
docker-compose up
```
- Frontend: http://localhost:8501
- Backend: http://localhost:8000

**Option 2: Separate Frontend/Backend**

1. **Clone the repository:**
```bash
git clone https://github.com/O-S-O-K/insight_ai_app.git
cd insight_ai_app
```

2. **Create Python environment:**
```bash
conda create -n xai-app python=3.10
conda activate xai-app
```

3. **Run Backend (Terminal 1):**
```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

4. **Run Frontend (Terminal 2):**
```bash
pip install -r requirements.txt

# Create .streamlit/secrets.toml with:
# INSIGHT_BACKEND_URL = "http://localhost:8000"

streamlit run app/app.py
```

**Option 3: Frontend Only (Mock Mode)**

For UI development without backend:
```bash
pip install -r requirements.txt
export USE_MOCK_API=true
streamlit run app/app.py
```

---

## Tech Stack

**Frontend:**
- Streamlit 1.53.0 (Web UI)
- Requests 2.32+ (HTTP client for backend communication)
- Pillow 10.3+ (Image processing)

**Backend:**
- FastAPI 0.110.2 (REST API framework)
- Uvicorn 0.23.2 (ASGI server)
- TensorFlow 2.15.0 CPU (Deep learning)
- tf-keras 2.15.1 (Legacy Keras 2 API compatibility)
- PyTorch 2.1.2 (For BLIP model)
- Transformers 4.35.2 (BLIP vision-language model)
- NumPy 1.23.5 (Numerical computing)
- Matplotlib 3.8.4 (Heatmap colormaps)
- OpenCV 4.8.1 (Image processing)

**Infrastructure:**
- Docker & Docker Compose (Containerization)
- Render.com (Backend hosting with auto-deploy)
- Streamlit Cloud (Frontend hosting)

**Data Persistence:**
- JSON for dynamic mappings and model metadata
- CSV for feedback audit logs

---

## Why This Project Matters

- Demonstrates CNN training and predictions
- Applies explainable AI via Grad-CAM
- Integrates vision-language reasoning (BLIP)
- Implements human-in-the-loop feedback for semantic alignment
- Handles ambiguous predictions with top-3 and "Other" input
- Dynamically updates Grad-CAM for user-selected classes
- Improves system behavior over time without retraining
- Modular, production-oriented Python code
- Deployable interactive ML app with Streamlit

Shows how ML systems can **learn from users**, adapt to ambiguity, and become more interpretable.

---

## Example Test Session (End-to-End)

1. Upload an image of a dog on a couch  
2. CNN Prediction:
   - german_shepherd: 74.02%
   - tabby: 1.81%
   - tiger_cat: 1.07%
3. BLIP Caption: "a cat and dog sitting on a couch"  
4. Initial BLIP → Class Mapping: ["beagle", "bloodhound", "golden_retriever"]  
5. User Feedback: Top prediction wrong → selects "Other" → correct class `german_shepherd`  
6. Grad-CAM overlay updates for `german_shepherd`  
7. System Action: Updates dynamic mapping file  
8. Result: User-driven explainability and feedback loop, no retraining required

---

## 🌍 Deployment

### Live Demo
👉 **https://insight-ai-v1.streamlit.app**

Mobile-compatible and runs entirely in the browser.

---

### Deploy Your Own Instance

**Frontend (Streamlit Cloud)**

1. Fork this repository on GitHub
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your fork
4. Add secrets in Streamlit Cloud dashboard:
   ```toml
   INSIGHT_BACKEND_URL = "https://your-backend.onrender.com"
   ```
5. Deploy! Streamlit will use `requirements.txt` automatically

**Backend (Render.com)**

1. Sign up at [Render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. **Docker deployment (recommended):**
   - Build Command: `docker build -f api/Dockerfile -t backend .`
   - Start Command: Uses Dockerfile CMD
5. **OR non-Docker deployment:**
   - Root Directory: `api`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Set environment variable in Render:
   - `HF_TOKEN` = Your Hugging Face token (for BLIP captions)
7. Deploy and note the service URL

**Connect Frontend to Backend:**
- Copy your Render backend URL
- Add it to Streamlit Cloud secrets as `INSIGHT_BACKEND_URL`
- Restart Streamlit app

---

### Local Docker Deployment

**Full stack:**
```bash
docker-compose up
```

**Backend only:**
```bash
docker build -f api/Dockerfile -t insight-backend .
docker run -p 8000:8000 insight-backend
```

**Frontend only:**
```bash
docker build -f app/Dockerfile -t insight-frontend .
docker run -p 8501:8501 -e INSIGHT_BACKEND_URL=http://backend:8000 insight-frontend
```

---

### Troubleshooting Deployment

**Issue: Model loads as `_UserObject` without layers**

This happens when SavedModel was created incorrectly. The application automatically falls back to H5 format, but you can regenerate the SavedModel:

```bash
python regenerate_savedmodel.py
```

See `SAVEDMODEL_FIX.md` for detailed troubleshooting.

**Issue: Build fails in Dockerfile**

The Dockerfile validates TensorFlow/Keras at build time. If it fails:
- Check TensorFlow version in `api/requirements.txt`
- Ensure `tensorflow-cpu==2.10.1` for CPU-only deployment
- Review build logs for specific error

**Issue: Health endpoint shows model not loaded**

Check the `/` endpoint:
```bash
curl https://your-backend.onrender.com/
```

Should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "has_layers": true,
  "num_layers": 155
}
```

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details. You are free to use, modify, and redistribute this project with attribution.

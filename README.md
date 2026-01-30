# InsightAI: Interactive Image Classification with Feedback

### An interactive Streamlit app combining CNN predictions, Grad-CAM explanations, BLIP captions, and human-in-the-loop feedback for smarter, interpretable AI.

![App Screenshot](screenshots/xai_app_screenshot.png)

**Insight AI** is an end-to-end Explainable AI (XAI) system that goes beyond static image classification. It combines:

- **Deep learning (CNNs)**
- **Explainability (Grad-CAM)**
- **Vision-language models (BLIP)**
- **Human-in-the-loop feedback**
- **Top-3 prediction selection & "Other" class input**
- **Dynamic, user-driven Grad-CAM overlays**
- **Interactive Streamlit interface**

The result is a project that demonstrates not only *model accuracy*, but *model understanding, inspection, and improvement over time*.

---

## Table of Contents

- [InsightAI: Interactive Image Classification with Feedback](#insightai-interactive-image-classification-with-feedback)
    - [An interactive Streamlit app combining CNN predictions, Grad-CAM explanations, BLIP captions, and human-in-the-loop feedback for smarter, interpretable AI.](#an-interactive-streamlit-app-combining-cnn-predictions-grad-cam-explanations-blip-captions-and-human-in-the-loop-feedback-for-smarter-interpretable-ai)
  - [Table of Contents](#table-of-contents)
  - [Project Motivation](#project-motivation)
  - [High-Level System Overview](#high-level-system-overview)
  - [Key Features](#key-features)
  - [Model Architecture](#model-architecture)
    - [CNN Image Classifier](#cnn-image-classifier)
  - [Datasets Used](#datasets-used)
    - [CIFAR-10 (Training)](#cifar-10-training)
    - [User-Provided Images (Inference)](#user-provided-images-inference)
  - [Explainability: Grad-CAM](#explainability-grad-cam)
  - [Vision-Language Integration (BLIP)](#vision-language-integration-blip)
  - [Dynamic Class Mapping](#dynamic-class-mapping)
  - [Installation](#installation)
  - [Tech Stack](#tech-stack)
  - [Why This Project Matters](#why-this-project-matters)
  - [Example Test Session (End-to-End)](#example-test-session-end-to-end)
  - [Author](#author)

---

## Project Motivation

Most entry-level ML projects stop at:
> “Here is my model accuracy.”

**Insight AI** goes further and asks:

- Why did the model make this prediction?
- Does the model’s reasoning align with human intuition?
- Can users correct the model if predictions are wrong?
- How do vision and language models complement each other?

It demonstrates *real-world ML system thinking*, integrating explainability, human feedback, and semantic understanding.

---

## High-Level System Overview

**Pipeline Overview:**

```text
User Image Upload
        │
        ▼
Image Preprocessing (resize, normalize)
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

This design shows how production ML systems can evolve over time, not remain static.

---

## Key Features

- Interactive Streamlit web interface
- Real-time CNN image classification with top-3 predictions
- Dynamic Grad-CAM heatmap visualization
- BLIP-generated captions for semantic understanding
- Keyword-to-class mapping
- Human-in-the-loop feedback with top-3 selection and "Other" input
- Session-safe prediction boosting
- Persistent feedback logging for future semantic alignment
- Modular, production-style codebase

---

## Model Architecture

### CNN Image Classifier

- Input shape: `(32, 32, 3)`
- Convolution + MaxPooling blocks
- Dense fully connected layers
- Softmax output layer
- Optimizer: Adam, Loss: Categorical Crossentropy
- Test Accuracy: ~73–74% (baseline CIFAR-style CNN)

The model is intentionally simple to emphasize **system design, explainability, and feedback integration**.

---

## Datasets Used

### CIFAR-10 (Training)
- 60,000 color images across 10 classes
- 50,000 training / 10,000 test images
- Normalized to `[0,1]`

### User-Provided Images (Inference)
- Arbitrary real-world images
- Resized and normalized
- Used for inference and feedback, not retraining

---

## Explainability: Grad-CAM

- Visualizes where the CNN is "looking" when making predictions
- Heatmaps update dynamically for **user-selected labels**
- Helps verify model focus and detect spurious correlations

---

## Vision-Language Integration (BLIP)

- Generates captions describing uploaded images
- Adds semantic context beyond CIFAR labels
- Supports verification and keyword-to-class mapping

---

## Dynamic Class Mapping

- BLIP captions are parsed into keywords mapped to possible classes
- User corrections (via top-3 or "Other") update session mapping
- Feedback persists in JSON + CSV
- Grad-CAM overlays reflect user-selected labels
- No model retraining required

---

## Installation

```bash
git clone https://github.com/O-S-O-K/explainable-ai-app.git
cd explainable-ai-app
conda create -n xai-app python=3.10
conda activate xai-app
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Tech Stack

- Python 3.10
- TensorFlow / Keras (CNN)
- Streamlit (Web Interface)
- Grad-CAM (Dynamic visualizations)
- BLIP (Vision–Language Model)
- NumPy, OpenCV, PIL
- JSON + CSV persistence for feedback-driven learning

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

This project shows how ML systems can **learn from users**, adapt to ambiguity, and become more trustworthy and interpretable.

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

## Author

Sheron Schley  
Focus Areas: Data Science, Deep Learning, Explainable AI, Applied Machine Learning Systems

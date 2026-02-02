# InsightAI: Interactive Image Classification with Feedback

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-v1.27-orange)](https://streamlit.io/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://insight-ai-v1.streamlit.app) [![Last Commit](https://img.shields.io/github/last-commit/O-S-O-K/insight-ai-app)](https://github.com/O-S-O-K/insight-ai-app)

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
  - [Model Architecture](#model-architecture)
    - [CNN Image Classifier](#cnn-image-classifier)
  - [Datasets Used](#datasets-used)
    - [ImageNet / Fine-Tuned Dataset](#imagenet--fine-tuned-dataset)
    - [User-Provided Images (Inference)](#user-provided-images-inference)
  - [Explainability: Grad-CAM](#explainability-grad-cam)
  - [Vision-Language Integration (BLIP, Optional)](#vision-language-integration-blip-optional)
  - [Dynamic Class Mapping](#dynamic-class-mapping)
  - [Installation](#installation)
  - [Tech Stack](#tech-stack)
  - [Why This Project Matters](#why-this-project-matters)
  - [Example Test Session (End-to-End)](#example-test-session-end-to-end)
  - [🌍 Deployment](#-deployment)
  - [Author](#author)
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

## Installation

Clone the repository and navigate into it:  
    git clone https://github.com/O-S-O-K/insight-ai-app.git  
    cd explainable-ai-app  

Create a Python environment and activate it:  
    conda create -n xai-app python=3.10  
    conda activate xai-app  

Install dependencies:  
    pip install -r requirements.txt  

Run the app:  
    streamlit run app/app.py  

(Optional) Place a fine-tuned model in `models/cnn_model_finetuned.h5`.

---

## Tech Stack

- Python 3.10
- TensorFlow / Keras (CNN)
- Streamlit (Web Interface)
- Grad-CAM (Dynamic visualizations)
- Optional BLIP (Vision–Language Model)
- NumPy, PIL, OpenCV
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

Deployed using **Streamlit Cloud**:  
👉 https://insight-ai-v1.streamlit.app  

Mobile-compatible and runs entirely in the browser.

---

## Author

Sheron Schley  
Focus Areas: Data Science, Deep Learning, Explainable AI, Applied Machine Learning Systems

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details. You are free to use, modify, and redistribute this project with attribution.

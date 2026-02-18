# Insight AI — Model Card

**Model Version:** v2.0 (ImageNet General) / v1.0 (Medical Imaging)
**Last Updated:** 2026-02-17
**Author:** Sheron Schley
**Contact:** [GitHub Issues](https://github.com/O-S-O-K/insight_ai_app/issues)

---

## Model Details

### ImageNet General Model
| Field | Value |
|-------|-------|
| **Architecture** | MobileNetV2 (Functional API) |
| **Pretrained on** | ImageNet-1K (1.2M images, 1000 classes) |
| **Input shape** | 224 × 224 × 3 (RGB) |
| **Output** | Softmax probabilities over 1000 classes |
| **Calibration** | Temperature scaling (T=1.5) applied post-inference |
| **Explainability** | Grad-CAM, SHAP GradientExplainer |
| **Additional** | CLIP zero-shot (ViT-B/32), BLIP captioning |
| **File** | `models/cnn_baseline_functional.h5` (14 MB) |
| **Framework** | TensorFlow 2.15.0 + tf-keras |

### Medical Imaging Model
| Field | Value |
|-------|-------|
| **Architecture** | EfficientNetB0 (fine-tuned) |
| **Pretrained on** | ImageNet-1K |
| **Fine-tuned on** | ISIC 2020 Skin Lesion Dataset |
| **Task** | Binary classification: Melanoma vs. Benign |
| **Input shape** | 224 × 224 × 3 (RGB) |
| **Output** | Softmax probability: [P(benign), P(melanoma)] |
| **Calibration** | Temperature scaling (T=1.2) |
| **File** | `models/medical_model.h5` |
| **Framework** | TensorFlow 2.15.0 + tf-keras |

---

## Intended Use

### Primary Use Cases
- **Portfolio demonstration** of computer vision and XAI techniques for software engineering recruitment
- **Educational** exploration of explainable AI methods (Grad-CAM, SHAP, CLIP)
- **Research prototyping** for image classification pipelines

### Out-of-Scope Uses
- **NOT for clinical diagnosis** — the Medical Imaging model is NOT a certified medical device
- **NOT for safety-critical decisions** — ImageNet model does not have real-world accuracy guarantees
- **NOT for production medical use** — no regulatory approval (FDA, CE) has been obtained
- **NOT for surveillance** — not designed or validated for facial recognition or person identification

---

## Ethical Considerations

### Medical Model
The skin lesion classifier trained on ISIC 2020 **must not be used for clinical decision-making**. Reasons:

1. **Class imbalance**: ISIC 2020 has ~55:1 benign-to-melanoma ratio. The model uses class weights to compensate but may still be biased toward false negatives on rare classes.
2. **Patient demographics**: ISIC 2020 is predominantly light-skinned patients. Model performance may degrade on darker skin tones — a known bias in dermatology AI datasets.
3. **Image quality dependency**: Performance assumes dermoscopy-quality images; smartphone photos may produce unreliable results.
4. **No clinical validation**: No prospective clinical trials have been conducted.

### General Model (ImageNet)
- Trained on ImageNet which contains demographic biases in object and person representation
- The 1000-class taxonomy reflects cultural assumptions from 2010s internet images
- Class labels (e.g., "African Grey" parrot) reflect common names that may have evolved in sensitivity

### Active Learning System
- Predictions flagged for human review are stored in `feedback_log.json`
- No personally identifiable information is collected by the system
- Flagged images are not shared externally by default

---

## Factors

### Model Performance Factors (Medical Model)
| Factor | Impact |
|--------|--------|
| Skin tone | Potential bias (ISIC skews light-skinned) |
| Image resolution | Higher resolution improves accuracy |
| Dermoscopy vs. clinical photo | Model trained on dermoscopy images |
| Lesion size | Smaller lesions harder to classify |
| Hair/artifacts | Occlusion reduces accuracy |

### Model Performance Factors (General Model)
| Factor | Impact |
|--------|--------|
| Image quality | Motion blur, low light reduce accuracy |
| Multiple objects | Predicts dominant object only |
| ImageNet class coverage | Works best on the 1000 training classes |
| Aspect ratio | Standardized to 224×224 by center-cropping |

---

## Metrics

### ImageNet General Model
| Metric | Value |
|--------|-------|
| Top-1 Accuracy (ImageNet val) | ~71.8% (standard MobileNetV2) |
| Top-5 Accuracy (ImageNet val) | ~91.0% |
| Calibration (expected) | Temperature T=1.5 reduces ECE |
| Inference time (CPU) | ~50-100 ms per image |

### Medical Imaging Model
| Metric | Value |
|--------|-------|
| Val Accuracy | *(update after training)* |
| Val AUC | *(update after training)* |
| Val Sensitivity | *(update after training)* |
| Val Specificity | *(update after training)* |

Run `python scripts/models/train_medical.py` to populate these metrics, which are then written to `models/medical_metadata.json`.

---

## Training Data

### ImageNet General Model
- **Dataset**: ImageNet Large Scale Visual Recognition Challenge (ILSVRC 2012)
- **Size**: 1.28M training images, 50K validation images
- **Classes**: 1000 (animals, objects, scenes, food, vehicles, etc.)
- **License**: Research / educational use
- **Source**: Pretrained weights from Keras Applications (Google)

### Medical Imaging Model
- **Dataset**: ISIC 2020 Skin Lesion Analysis Challenge
- **URL**: https://challenge2020.isic-archive.com/
- **Size**: 33,126 dermoscopy images
- **Classes**: Melanoma (584), Benign (32,542)
- **License**: CC BY-NC 4.0 (non-commercial)
- **Annotation**: Expert dermatologist labels

---

## Evaluation Data

### ImageNet General Model
The model uses standard ImageNet validation split for reporting performance.

### Medical Imaging Model
Evaluation uses an 80/20 stratified train/val split of the ISIC 2020 training set (no test labels available publicly for ISIC 2020). Held-out test performance should be validated on an independent clinical dataset before any application.

---

## Quantitative Analyses

### Confidence Calibration
Both models apply temperature scaling before returning confidence scores:
- **General model**: T=1.5 (reduces overconfidence typical of pre-trained networks)
- **Medical model**: T=1.2
- Effect: Softens probability distributions; a 90% raw confidence becomes ~78% calibrated

### Explainability Methods
| Method | What it shows | Scope |
|--------|--------------|-------|
| Grad-CAM | Pixels that activate the predicted class | Last conv layer |
| SHAP GradientExplainer | Feature attribution (signed importance) | All layers |
| CLIP Zero-Shot | Similarity to user-defined text labels | Embedding space |

---

## Caveats and Recommendations

1. **Always run predictions on the correct model**: Use `MODEL_TYPE=imagenet` for general objects, `MODEL_TYPE=medical` only for research on dermoscopy images.
2. **Calibrated confidence ≠ ground truth probability**: Temperature scaling improves calibration but does not make the model infallible.
3. **Active learning flags are advisory**: Low-confidence predictions are automatically flagged for review — they are not automatically rejected or discarded.
4. **CLIP zero-shot is comparative**: CLIP scores are relative to the provided labels, not absolute probabilities. Providing more descriptive labels (e.g., "a close-up photo of a golden retriever dog") improves accuracy.
5. **Grad-CAM highlights class-discriminative regions**: These are not guaranteed to correspond to clinically meaningful regions in the medical context.

---

## Citation

If you use this project in academic work:

```bibtex
@software{insight_ai_2026,
  author = {Sheron Schley},
  title = {Insight AI: Explainable Image Classification with CLIP, SHAP, and Grad-CAM},
  year = {2026},
  url = {https://github.com/O-S-O-K/insight_ai_app}
}
```

---

## License

MIT License — see [LICENSE](../LICENSE) for details.

The ISIC 2020 dataset used for medical model training is licensed CC BY-NC 4.0 (non-commercial).

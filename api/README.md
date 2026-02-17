# InsightAI Backend API

FastAPI backend service for InsightAI, handling heavy ML inference tasks including CNN predictions, Grad-CAM visualizations, and BLIP image captioning.

---

## Overview

The backend API provides:
- **CNN Image Classification**: Top-K predictions with confidence scores
- **Grad-CAM Visualization**: Heatmap overlays showing model attention
- **BLIP Captioning**: Natural language descriptions of images
- **Feedback Collection**: Storage of user corrections and ratings
- **Robust Model Loading**: Automatic fallback from SavedModel to H5 format
- **Health Monitoring**: Diagnostic endpoint for deployment verification

---

## API Endpoints

### `GET /`
Health check endpoint with diagnostic information.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "<class 'keras.engine.functional.Functional'>",
  "has_layers": true,
  "num_layers": 155,
  "gradcam_layer": "Conv_1",
  "tensorflow_version": "2.10.1"
}
```

### `POST /predict`
Image classification with top-K predictions.

**Request:** Multipart form data with `file` (image)

**Response:**
```json
{
  "predictions": [
    {
      "class_idx": 0,
      "class_name": "Cat",
      "confidence": 0.92
    }
  ]
}
```

### `POST /caption`
Generate natural language caption for image.

**Request:** Multipart form data with `file` (image)

**Response:**
```json
{
  "caption": "a cat sitting on a couch"
}
```

### `POST /gradcam`
Generate Grad-CAM heatmaps for top predictions.

**Request:**
- Multipart form data with `file` (image)
- Optional `top_k` parameter (default: 3)

**Response:**
```json
{
  "gradcams": [
    {
      "class_idx": 0,
      "class_name": "Cat",
      "confidence": 0.92,
      "heatmap_base64": "data:image/png;base64,..."
    }
  ]
}
```

### `POST /feedback`
Submit human feedback for predictions.

**Request:**
- Multipart form data with `file` (image)
- Form field `entry` (JSON string):
  ```json
  {
    "feedback": "Correct prediction",
    "rating": 5
  }
  ```

**Response:**
```json
{
  "status": "success"
}
```

---

## Local Development

### Prerequisites
- Python 3.10
- Models in `models/` directory:
  - `cnn_model.h5` (required)
  - `cnn_baseline_savedmodel/` (optional, with fallback)
  - `model_metadata.json` (class labels)

### Setup

1) **Create a Python environment:**
```bash
conda create -n xai-backend python=3.10
conda activate xai-backend
```

2) **Install dependencies:**
```bash
pip install -r api/requirements.txt
```

3) **Set environment variables (optional):**
```bash
export HF_TOKEN=your_hf_token_here  # For BLIP captions
```

4) **Run the backend:**
```bash
# From project root
uvicorn api.main:app --reload --port 8000
```

5) **Test the health endpoint:**
```bash
curl http://localhost:8000/
```

6) **Run the Streamlit frontend (in another terminal):**
```bash
pip install -r requirements.txt
export INSIGHT_BACKEND_URL=http://localhost:8000
streamlit run app/app.py
```

---

## Docker Deployment

### Build and Run Locally

```bash
# Build
docker build -f api/Dockerfile -t insight-backend .

# Run
docker run -p 8000:8000 insight-backend

# Test
curl http://localhost:8000/
```

### Docker Compose

Run both frontend and backend:
```bash
docker-compose up
```

---

## Cloud Deployment (Render)

See `DEPLOY_RENDER.md` for detailed instructions.

**Quick Steps:**
1. Create new Web Service on Render
2. Connect GitHub repository
3. Select Dockerfile deployment: `api/Dockerfile`
4. Set environment variables:
   - `HF_TOKEN` (optional, for BLIP captions)
5. Deploy and note the service URL
6. Configure frontend with backend URL

**Environment Variables:**
- `HF_TOKEN` - Hugging Face token for BLIP captions (optional)
- `PORT` - Port number (auto-set by Render)
- `TF_USE_LEGACY_KERAS=1` - Set in Dockerfile
- `CUDA_VISIBLE_DEVICES=-1` - CPU-only mode, set in Dockerfile

---

## Model Loading

The backend uses a robust model loading strategy:

1. **Attempts SavedModel format** (`models/cnn_baseline_savedmodel/`)
   - Validates model has Keras attributes (`.layers`, `.predict`)
   - If invalid (loads as `_UserObject`), falls back to H5

2. **Falls back to H5 format** (`models/cnn_model.h5`)
   - Standard Keras model format
   - More reliable for cross-platform deployment

3. **Validation**
   - Checks for required attributes before accepting model
   - Fails fast with clear error messages if neither format works

**Regenerate SavedModel:**
If you need to fix an invalid SavedModel:
```bash
python regenerate_savedmodel.py
```

See `SAVEDMODEL_FIX.md` for troubleshooting.

---

## Troubleshooting

**Model loading errors:**
- Check logs for "Attempting to load SavedModel..." messages
- Verify `cnn_model.h5` exists in `models/` directory
- Run health endpoint to see model status

**Port binding issues:**
- Render sets `$PORT` environment variable automatically
- Locally, default is 8000
- Docker: Ensure port mapping matches (`-p 8000:8000`)

**BLIP caption errors:**
- Requires significant CPU/RAM
- Set `HF_TOKEN` if using Hugging Face API
- May be slow on first load (model download)

**Import errors:**
- Dockerfile validates TensorFlow at build time
- Check `api/requirements.txt` for version compatibility
- Ensure `tensorflow-cpu==2.10.1` for CPU deployment

---

## Project Structure

```
api/
├── main.py                 # FastAPI application
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── DEPLOY_RENDER.md      # Render deployment guide
└── utils/                # Utility modules (mirrors root utils/)
    ├── preprocessing.py
    ├── gradcam.py
    ├── blip_caption.py
    └── keyword_mapping.py
```

---

## For More Information

- **Project Overview**: See [root README](../README.md)
- **Frontend Setup**: See [app/app.py](../app/app.py)
- **Model Troubleshooting**: See [SAVEDMODEL_FIX.md](../SAVEDMODEL_FIX.md)
- **Render Deployment**: See [DEPLOY_RENDER.md](DEPLOY_RENDER.md)

---

## Tech Stack

- **Framework**: FastAPI 0.110.2
- **Server**: Uvicorn 0.23.2
- **ML**: TensorFlow 2.10.1 CPU, PyTorch 2.1.2
- **Vision-Language**: Transformers 4.52+ (BLIP)
- **Processing**: NumPy 1.23.5, OpenCV 4.8.1, Matplotlib 3.8.4
- **Deployment**: Docker, Render.com

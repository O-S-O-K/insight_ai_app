# Insight AI - Setup Guide

Complete setup instructions for local development and deployment.

## Prerequisites

- **Python**: 3.10 (required for TensorFlow 2.15.0 compatibility)
- **Git**: For cloning the repository
- **Docker** (optional): For containerized deployment
- **Hugging Face Account** (optional): For BLIP captioning functionality

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/O-S-O-K/insight_ai_app.git
cd insight_ai_app
```

### 2. Environment Setup

Copy the environment template and configure:

```bash
cp .env.example .env
```

Edit `.env` to set:
- `INSIGHT_BACKEND_URL=http://localhost:8000` (for local development)
- `ENABLE_BLIP=true` (set to `false` on low-memory systems)
- Other variables as needed

### 3. Local Development (Recommended)

**Option A: Docker Compose (Full Stack)**

Run both frontend and backend with one command:

```bash
docker-compose up
```

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Option B: Manual Setup**

Create a Python virtual environment:

```bash
# Using conda
conda create -n xai-app python=3.10
conda activate xai-app

# OR using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

**Start Backend** (Terminal 1):

```bash
pip install -r api/requirements.txt
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start Frontend** (Terminal 2):

```bash
pip install -r requirements.txt

# Create Streamlit secrets
mkdir -p .streamlit
echo 'INSIGHT_BACKEND_URL = "http://localhost:8000"' > .streamlit/secrets.toml

streamlit run app/app.py
```

**Option C: Frontend Only (Mock Mode)**

For UI development without running the backend:

```bash
pip install -r requirements.txt
export USE_MOCK_API=true  # Windows: set USE_MOCK_API=true
streamlit run app/app.py
```

## Model Files

The repository includes pre-trained models in `/models`:

- `cnn_baseline_functional.h5` (14 MB) - Primary MobileNetV2 model
- `cnn_baseline_savedmodel/` (12 MB) - SavedModel format (fallback)
- `model_metadata.json` (24 KB) - ImageNet-1K class labels (1000 classes)

These models are ready to use. No training required.

## Configuration Details

### Backend Configuration

The FastAPI backend (`api/main.py`) uses these environment variables:

- `ENABLE_BLIP`: Enable BLIP image captioning (default: `true`)
  - Set to `false` on systems with <1GB RAM
  - BLIP requires PyTorch and ~500MB additional RAM
- `CUDA_VISIBLE_DEVICES`: GPU control (default: `-1` for CPU-only)
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging verbosity (default: `2`)
- `TF_USE_LEGACY_KERAS`: Use legacy Keras 2.x API (default: `1`)

### Frontend Configuration

The Streamlit app (`app/app.py`) uses:

- `INSIGHT_BACKEND_URL`: Backend API endpoint
- `USE_MOCK_API`: Use mock data instead of real backend (default: `false`)

## Deployment

### Streamlit Cloud (Frontend)

1. Fork this repository on GitHub
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app pointing to your fork
4. Add secrets in dashboard:
   ```toml
   INSIGHT_BACKEND_URL = "https://your-backend-url.com"
   ```
5. Deploy - Streamlit automatically uses `requirements.txt`

### Render.com (Backend)

**Docker Deployment (Recommended)**

1. Sign up at [Render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Configure service:
   - **Environment**: Docker
   - **Dockerfile Path**: `api/Dockerfile`
   - **Plan**: Free tier (512 MB RAM) works with `ENABLE_BLIP=false`
5. Optional environment variables:
   - `ENABLE_BLIP=false` (for free tier)
   - `HF_TOKEN=your_token` (if using BLIP)
6. Deploy and note service URL

**Manual Deployment**

1. Create Web Service on Render
2. Configure:
   - **Root Directory**: `api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Set environment variables as above
4. Deploy

### Connect Frontend to Backend

After deploying backend:

1. Copy backend URL from Render (e.g., `https://insight-backend.onrender.com`)
2. Add to Streamlit Cloud secrets:
   ```toml
   INSIGHT_BACKEND_URL = "https://insight-backend.onrender.com"
   ```
3. Restart Streamlit app

## Verification

### Check Backend Health

```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "status": "ready",
  "model_loaded": true,
  "blip_loaded": true,
  "message": "Backend is ready"
}
```

### Check Frontend

Visit http://localhost:8501 and:

1. Upload an image (JPG, JPEG, or PNG)
2. Click **Predict** - should show top 3 predictions
3. Click **Caption** - should generate image description (if BLIP enabled)
4. Click **Grad-CAM** - should show attention heatmaps

## Troubleshooting

### Backend Won't Start

**Issue**: TensorFlow/Keras import errors

**Solution**: Ensure Python 3.10 and correct package versions:
```bash
pip install tensorflow-cpu==2.15.0 tf-keras==2.15.1
```

**Issue**: Out of memory errors

**Solution**: Disable BLIP
```bash
export ENABLE_BLIP=false
```

### Frontend Cannot Connect to Backend

**Issue**: `Backend URL is not configured`

**Solution**: Create `.streamlit/secrets.toml`:
```toml
INSIGHT_BACKEND_URL = "http://localhost:8000"
```

**Issue**: Connection refused

**Solution**: Verify backend is running:
```bash
curl http://localhost:8000/
```

### Model Loading Failures

**Issue**: Model loads without layers

**Solution**: The app automatically falls back to H5 format. To regenerate SavedModel:
```bash
python scripts/models/regenerate_savedmodel.py
```

**Issue**: Model input shape errors

**Solution**: Use provided models which have correct 224x224 input shape

### Deployment Issues

**Issue**: Render free tier out of memory

**Solution**: Set `ENABLE_BLIP=false` in Render environment variables

**Issue**: Streamlit app shows "Backend Currently Offline"

**Solution**:
- Render free tier instances sleep after inactivity
- Wait 2-3 minutes for backend to wake up
- Refresh page

## Project Structure

```
insight_ai_app/
├── api/                    # FastAPI backend
│   ├── main.py            # API server
│   ├── requirements.txt   # Backend dependencies
│   └── utils/             # Backend utilities (Grad-CAM, BLIP, preprocessing)
├── app/                    # Streamlit frontend
│   ├── app.py             # Main UI
│   ├── requirements.txt   # Frontend dependencies
│   └── utils/             # Frontend utilities
├── models/                 # Pre-trained models
│   ├── cnn_baseline_functional.h5
│   ├── cnn_baseline_savedmodel/
│   └── model_metadata.json
├── scripts/                # Utility scripts
│   └── models/            # Model conversion/validation scripts
├── docs/                   # Documentation
├── .env.example           # Environment template
├── docker-compose.yml     # Full-stack Docker setup
└── requirements.txt       # Frontend-only dependencies
```

## Next Steps

- **Development**: See [docs/DEVELOPMENT.md](DEVELOPMENT.md) for architecture details
- **Scripts**: See [scripts/README.md](../scripts/README.md) for utility scripts
- **API Documentation**: Visit http://localhost:8000/docs when backend is running
- **Live Demo**: Try the deployed app at https://insight-ai-v1.streamlit.app

## Support

- **GitHub Issues**: [O-S-O-K/insight_ai_app/issues](https://github.com/O-S-O-K/insight_ai_app/issues)
- **Documentation**: See additional docs in `/docs` directory
- **Author**: Sheron Schley

## License

MIT License - See [LICENSE](../LICENSE) for details

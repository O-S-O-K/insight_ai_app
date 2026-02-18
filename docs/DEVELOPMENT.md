# Insight AI - Development Guide

Internal documentation index and development workflow for contributors and maintainers.

## Documentation Index

This directory contains technical documentation for system internals, deployment, and troubleshooting.

### Setup & Deployment

- **[SETUP.md](SETUP.md)** - Complete setup guide for local development and deployment
- **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)** - Deployment architecture and optimization history

### Technical References

- **[MODEL_LOADING_FIXES.md](MODEL_LOADING_FIXES.md)** - Model loading troubleshooting and fallback mechanisms
- **[SAVEDMODEL_FIX.md](SAVEDMODEL_FIX.md)** - SavedModel regeneration and compatibility fixes
- **[../models/MODEL_CARD.md](../models/MODEL_CARD.md)** - Responsible AI model card (intended use, ethics, metrics)

### Project Documentation

- **[README.md](../README.md)** - Main project overview and features
- **[api/README.md](../api/README.md)** - Backend API documentation

## Architecture Overview

### System Design

Insight AI uses a **microservices architecture** with separated frontend and backend:

```
┌─────────────────────┐         ┌─────────────────────┐
│   Streamlit App     │  HTTP   │   FastAPI Backend   │
│   (Frontend)        │ ◄─────► │   (Inference)       │
│                     │         │                     │
│  - UI/UX            │         │  - CNN Model        │
│  - Image Upload     │         │  - Grad-CAM         │
│  - Response Display │         │  - BLIP Captions    │
│  - Feedback Forms   │         │  - Preprocessing    │
└─────────────────────┘         └─────────────────────┘
         │                               │
         │                               │
         ▼                               ▼
   Streamlit Cloud              Render.com / Local
   (Static Hosting)             (Compute Instance)
```

**Benefits:**
- Independent scaling (frontend can scale separately from compute-heavy backend)
- Deployment flexibility (frontend on Streamlit Cloud, backend anywhere)
- Clear separation of concerns
- Easier testing (mock API for frontend development)

### Technology Stack

**Frontend** (`app/`)
- **Streamlit 1.53.0**: Web UI framework
- **Requests**: HTTP client for backend communication
- **Pillow**: Image processing and display
- **Simple, stateful**: Session state for multi-step workflows

**Backend** (`api/`)
- **FastAPI 0.110.2**: Modern async API framework
- **TensorFlow 2.15.0 + tf-keras**: Deep learning inference
- **PyTorch 2.1.2**: BLIP model execution
- **Transformers 4.35.2**: Hugging Face BLIP integration
- **Uvicorn**: ASGI server

**Infrastructure**
- **Docker**: Containerization for consistent deployments
- **GitHub Actions**: (Future) CI/CD automation
- **Render.com**: Backend hosting with auto-deploy
- **Streamlit Cloud**: Frontend hosting

## Project Structure

```
insight_ai_app/
├── api/                          # Backend FastAPI service
│   ├── main.py                   # API server & model loading
│   ├── requirements.txt          # Backend dependencies
│   ├── Dockerfile                # Backend containerization
│   └── utils/                    # Backend utilities
│       ├── preprocessing.py      # Image preprocessing pipeline
│       ├── gradcam.py            # Grad-CAM implementation
│       └── blip_caption.py       # BLIP captioning wrapper
│
├── app/                          # Frontend Streamlit app
│   ├── app.py                    # Main UI and workflow
│   ├── requirements.txt          # Frontend dependencies
│   ├── Dockerfile                # Frontend containerization
│   └── utils/                    # Frontend utilities
│       ├── api_client.py         # Backend HTTP client
│       └── mock_api_client.py    # Mock API for testing
│
├── models/                       # Pre-trained models
│   ├── cnn_baseline_functional.h5      # Primary model (H5 format)
│   ├── cnn_baseline_savedmodel/        # SavedModel format (fallback)
│   └── model_metadata.json             # Class labels (ImageNet-1K)
│
├── scripts/                      # Utility scripts
│   └── models/                   # Model management scripts
│       ├── check_model_shape.py
│       ├── check_tf_keras.py
│       ├── check_tf_version.py
│       ├── convert_to_functional_api.py
│       ├── fix_models_for_tf2.10.py
│       └── regenerate_savedmodel.py
│
├── docs/                         # Documentation (this directory)
│   ├── DEVELOPMENT.md           # This file
│   ├── SETUP.md                 # Setup guide
│   ├── DEPLOYMENT_SUMMARY.md    # Deployment docs
│   ├── MODEL_LOADING_FIXES.md   # Troubleshooting
│   ├── SAVEDMODEL_FIX.md        # Model fixes
│   └── RENDER_QUICKSTART.md     # Render deployment
│
├── feedback_images/              # User feedback data (gitignored)
├── .streamlit/                   # Streamlit configuration (gitignored)
├── .env.example                  # Environment variable template
├── docker-compose.yml            # Full-stack Docker setup
└── requirements.txt              # Frontend-only dependencies (root)
```

## Development Workflow

### Local Development Setup

1. **Clone and setup** (see [SETUP.md](SETUP.md))
2. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Development modes**:

   **Full stack** (Docker Compose):
   ```bash
   docker-compose up
   ```

   **Backend only**:
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

   **Frontend only** (Mock mode):
   ```bash
   export USE_MOCK_API=true
   streamlit run app/app.py
   ```

4. **Make changes** and test locally
5. **Commit with conventional commits**:
   ```bash
   git add .
   git commit -m "feat: add new feature X"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style (formatting, no logic change)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Build process, dependencies, tooling

Examples:
```bash
git commit -m "feat: add Grad-CAM intensity slider"
git commit -m "fix: resolve UTF-8 encoding in requirements.txt"
git commit -m "docs: update deployment guide for Render"
git commit -m "chore: upgrade TensorFlow to 2.15.0"
```

## Key Systems & Patterns

### 1. Model Loading Strategy

**Location**: `api/main.py` lines 96-175

**Strategy**: Robust fallback mechanism for model loading

```python
# Priority order:
1. cnn_baseline_functional.h5 (primary)
2. cnn_baseline_savedmodel/ (SavedModel fallback)
3. Alternative H5 paths (legacy models)
```

**Why**: Different TensorFlow/Keras versions have compatibility issues. Multi-format fallback ensures reliability.

**Details**: See [MODEL_LOADING_FIXES.md](MODEL_LOADING_FIXES.md)

### 2. Memory Optimization

**Location**: `api/main.py` line 18

**Pattern**: Conditional loading of heavy models

```python
ENABLE_BLIP = os.environ.get("ENABLE_BLIP", "true").lower() == "true"
```

**Why**: BLIP model requires ~500MB RAM. Free tier deployments (512MB RAM) can disable it.

**Usage**: Set `ENABLE_BLIP=false` in `.env` for low-memory environments

### 3. Backend Health Check

**Location**: `app/app.py` lines 56-104

**Pattern**: Graceful degradation with maintenance pages

**Flow**:
1. Frontend checks backend health on load
2. If unhealthy, shows maintenance page with status
3. If `status=loading`, shows "AI Models Loading" message
4. If `status=ready`, proceeds to main app

**Why**: Render free tier instances sleep after inactivity. Takes 2-3 minutes to wake up.

### 4. Mock API Client

**Location**: `app/utils/mock_api_client.py`

**Pattern**: Development mode for frontend-only work

**Usage**:
```bash
export USE_MOCK_API=true
streamlit run app/app.py
```

**Why**: Allows UI development without running backend

### 5. Session State Management

**Location**: `app/app.py` lines 252-258

**Pattern**: State reset on new image upload

```python
def reset_state_on_new_image(new_hash: str):
    if st.session_state.get("image_hash") != new_hash:
        st.session_state.image_hash = new_hash
        st.session_state.feedback_submitted = False
        st.session_state.predictions = None
        # ... reset other state
```

**Why**: Prevents stale predictions when user uploads new image

### 6. Image Preprocessing Pipeline

**Location**: `api/utils/preprocessing.py`

**Pipeline**:
1. PIL Image → NumPy array
2. Resize to (224, 224)
3. MobileNetV2 preprocessing ([-1, 1] normalization)
4. Batch dimension expansion

**Consistency**: Same preprocessing in both real and mock API

## Testing

### Manual Testing Checklist

**Backend** (`http://localhost:8000`):
- [ ] `/` - Health check returns `status: ready`
- [ ] `/docs` - Swagger UI loads
- [ ] `/predict` - Returns top-K predictions with confidences
- [ ] `/caption` - Returns BLIP caption (if enabled)
- [ ] `/gradcam` - Returns base64 heatmap images

**Frontend** (`http://localhost:8501`):
- [ ] Image upload works (JPG, PNG, JPEG)
- [ ] Predict button shows top 3 predictions
- [ ] Caption button generates description
- [ ] Grad-CAM button shows heatmaps
- [ ] Heatmap intensity slider works
- [ ] Feedback form submits successfully
- [ ] Feedback persists on page refresh (same image)
- [ ] New image resets state

### Future: Automated Testing

Planned improvements:
- **Unit tests**: pytest for utility functions
- **Integration tests**: Backend API endpoint testing
- **E2E tests**: Streamlit app testing with selenium
- **CI/CD**: GitHub Actions for automated testing on PR

## Deployment

### Development Deployment

See [SETUP.md](SETUP.md) for local development setup.

### Production Deployment

See [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) for full deployment architecture.

**Quick overview**:
1. **Backend**: Deploy to Render.com with Docker
2. **Frontend**: Deploy to Streamlit Cloud
3. **Connect**: Add backend URL to Streamlit secrets

### Deployment Checklist

Before deploying:
- [ ] Update version in README badges
- [ ] Test locally with Docker Compose
- [ ] Verify environment variables in `.env.example`
- [ ] Check model files are committed
- [ ] Update documentation if architecture changed
- [ ] Test with `ENABLE_BLIP=false` for free tier compatibility

## Common Development Tasks

### Add New Model

1. Train/export model to H5 or SavedModel format
2. Copy to `models/` directory
3. Update `api/main.py` MODEL_PATH or add to ALT_MODEL_PATHS
4. Update `model_metadata.json` if classes changed
5. Test loading with `scripts/models/check_model_shape.py`
6. Update documentation

### Add New API Endpoint

1. Add endpoint in `api/main.py`
2. Add client method in `app/utils/api_client.py`
3. Add mock method in `app/utils/mock_api_client.py`
4. Update frontend UI in `app/app.py`
5. Test with both real and mock backends
6. Update API documentation

### Update Dependencies

1. Update `api/requirements.txt` or `requirements.txt`
2. Test locally in clean environment
3. Update `.env.example` if new env vars needed
4. Update Dockerfile if base image changes
5. Test Docker build: `docker-compose build`
6. Document breaking changes

### Fix Model Compatibility Issue

See [MODEL_LOADING_FIXES.md](MODEL_LOADING_FIXES.md) for detailed troubleshooting.

Quick reference:
1. Check TensorFlow/Keras versions match training environment
2. Try regenerating with `scripts/models/regenerate_savedmodel.py`
3. Add fallback path to `ALT_MODEL_PATHS` in `api/main.py`
4. Verify input shape matches 224x224x3

## Code Style & Best Practices

### Python Style

- **PEP 8**: Follow standard Python style guide
- **Type hints**: Use for function signatures where helpful
- **Docstrings**: Add for complex functions
- **Comments**: Explain *why*, not *what*
- **Imports**: Group standard lib, third-party, local

### FastAPI Patterns

- **Async when possible**: Use `async def` for I/O operations
- **Dependency injection**: Use FastAPI dependencies for shared logic
- **Error handling**: Return appropriate HTTP status codes
- **Validation**: Use Pydantic models for request/response

### Streamlit Patterns

- **Session state**: Use for persistent data across reruns
- **Caching**: Use `@st.cache_data` for expensive operations
- **Error handling**: Show user-friendly messages with `st.error()`
- **State reset**: Clear state when user starts new workflow

## Troubleshooting

### Common Issues

See dedicated documentation:
- **Model loading**: [MODEL_LOADING_FIXES.md](MODEL_LOADING_FIXES.md)
- **SavedModel**: [SAVEDMODEL_FIX.md](SAVEDMODEL_FIX.md)
- **Deployment**: [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)
- **Setup**: [SETUP.md](SETUP.md)

### Getting Help

1. **Check documentation** in `/docs` directory
2. **Search GitHub issues**: Existing solutions may exist
3. **Create new issue**: Provide error logs and environment details
4. **Contact maintainer**: See README for contact info

## Future Roadmap

Planned improvements (not prioritized):

- [ ] **Testing**: pytest suite for backend, frontend tests
- [ ] **CI/CD**: GitHub Actions for automated testing and deployment
- [ ] **Monitoring**: Error tracking and performance monitoring
- [ ] **Model versioning**: Track model performance over time
- [ ] **Feedback loop**: Use feedback data to improve predictions
- [ ] **Multi-model support**: Switch between different architectures
- [ ] **API authentication**: Secure backend endpoints
- [ ] **Rate limiting**: Prevent abuse of public API
- [ ] **User accounts**: Personalized feedback history
- [ ] **Batch processing**: Process multiple images at once

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`feature/amazing-feature`)
3. Follow code style and commit conventions
4. Write/update tests if applicable
5. Update documentation
6. Submit pull request

## License

MIT License - See [LICENSE](../LICENSE)

## Contact

**Author**: Sheron Schley
**GitHub**: [O-S-O-K/insight_ai_app](https://github.com/O-S-O-K/insight_ai_app)
**Issues**: [GitHub Issues](https://github.com/O-S-O-K/insight_ai_app/issues)

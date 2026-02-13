# Use lightweight Python image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_USE_LEGACY_KERAS=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy backend requirements and install
COPY api/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Fail fast if TensorFlow not importable and validate model loading
RUN python - <<'PY'
import tensorflow as tf
print('TF OK', tf.__version__)

# Handle both Keras 3 (separate package) and legacy tf.keras
try:
    # Try legacy tf.keras first (works with TF_USE_LEGACY_KERAS=1)
    from tensorflow.keras import layers, models
    print('tf.keras OK (legacy)')
except (ImportError, ModuleNotFoundError):
    # Fall back to standalone Keras 3
    from keras import layers, models
    print('keras OK (standalone)')

# Verify keras models can be loaded properly
test_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation='softmax')
])
print('Test model creation OK')

# Verify model has expected attributes
assert hasattr(test_model, 'layers'), "Model missing 'layers' attribute"
assert hasattr(test_model, 'predict'), "Model missing 'predict' method"
assert hasattr(test_model, 'get_layer'), "Model missing 'get_layer' method"
assert hasattr(test_model, 'inputs'), "Model missing 'inputs' attribute"
print('Model attributes validation OK')
print(f'Model has {len(test_model.layers)} layers')
PY

# Copy backend code and models
COPY api/ ./api
COPY models/ ./models

# Prepare writable data directories
RUN mkdir -p /app/feedback_images \
    && chown -R appuser:appuser /app

# Switch to non-root
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (honor Render's PORT if provided)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Use lightweight Python image
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

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

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

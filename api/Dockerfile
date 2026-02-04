FROM python:3.10-slim

# Install system deps for PIL and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user + venv
RUN useradd -m appuser \
    && python -m venv /home/appuser/venv \
    && chown -R appuser:appuser /home/appuser \
    && su appuser -c "/home/appuser/venv/bin/pip install --upgrade pip"

WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN chown -R appuser:appuser /home/appuser/venv \
    && su appuser -c "/home/appuser/venv/bin/pip install --no-cache-dir -r requirements.txt"

# Copy full repo (because build context = ..)
COPY .. /app
RUN chown -R appuser:appuser /app /home/appuser

ENV PATH="/home/appuser/venv/bin:${PATH}"
USER appuser

# Render-compatible startup
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

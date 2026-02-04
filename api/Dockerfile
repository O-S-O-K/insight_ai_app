FROM python:3.10-slim

# Install system deps for PIL and build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and a virtual environment in their home
RUN useradd -m appuser \
    && python -m venv /home/appuser/venv \
    && chown -R appuser:appuser /home/appuser \
    && su appuser -c "/home/appuser/venv/bin/pip install --upgrade pip"

WORKDIR /app
COPY requirements.txt ./
# Install Python requirements into the venv as appuser (avoids pip running as root)
RUN chown -R appuser:appuser /home/appuser/venv \
    && su appuser -c "/home/appuser/venv/bin/pip install --no-cache-dir -r requirements.txt"

# Copy project files and set ownership to the non-root user
COPY .. /app
RUN chown -R appuser:appuser /app /home/appuser

# Use the virtualenv's binaries and run as non-root user
ENV PATH="/home/appuser/venv/bin:${PATH}"
USER appuser

EXPOSE 8000
# Use uvicorn with an explicit app directory to avoid module path issues
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/api"]

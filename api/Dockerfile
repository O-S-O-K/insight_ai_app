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
    && /home/appuser/venv/bin/pip install --upgrade pip

WORKDIR /app
COPY requirements.txt ./
# Install Python requirements into the venv (avoids running pip as root at runtime)
RUN /home/appuser/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy project files and set ownership to the non-root user
COPY .. /app
RUN chown -R appuser:appuser /app /home/appuser

# Use the virtualenv's binaries and run as non-root user
ENV PATH="/home/appuser/venv/bin:${PATH}"
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

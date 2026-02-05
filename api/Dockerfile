# Use a lightweight Python image
FROM python:3.10-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and virtual environment
RUN useradd -m appuser \
    && python -m venv /home/appuser/venv

# Set working directory
WORKDIR /app

# Copy API requirements and install them
COPY api/requirements.txt .
RUN /home/appuser/venv/bin/pip install --upgrade pip \
    && /home/appuser/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api/ ./api
# Copy utils folder so imports work
COPY utils/ ./utils
# Copy main.py from archived root_files
COPY _archive_runtime/root_files/main.py . 
# Copy models folder
COPY models/ ./models

# Switch to non-root user
USER appuser

# Expose default FastAPI port
EXPOSE 8000

# Run the API using uvicorn
CMD ["/home/appuser/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

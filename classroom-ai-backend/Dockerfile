# ===========================================
# OPTIMIZED MULTI-STAGE DOCKERFILE
# Classroom AI Backend with GPU Support
# ===========================================

# ============ BASE STAGE ============
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# ============ DEPENDENCIES STAGE ============
FROM base AS dependencies

# Copy requirements files
COPY requirements/base.txt requirements/ml.txt ./requirements/

# Install base dependencies
RUN pip3 install --no-cache-dir -r requirements/base.txt

# Install ML dependencies
RUN pip3 install --no-cache-dir -r requirements/ml.txt

# ============ DEVELOPMENT STAGE ============
FROM dependencies AS development

# Install dev dependencies
COPY requirements/dev.txt ./requirements/
RUN pip3 install --no-cache-dir -r requirements/dev.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/cache /app/data/uploads /app/data/models && \
    chown -R app:app /app/data

# Switch to app user
USER app

# Copy application code (will be overridden by volume mount in docker-compose)
COPY --chown=app:app . .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run application
CMD ["python3", "main.py"]

# ============ PRODUCTION STAGE ============
FROM dependencies AS production

# Copy only necessary application files
COPY --chown=root:root backend/ ./backend/
COPY --chown=root:root main.py ./
COPY --chown=root:root .env.example .env

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    mkdir -p /app/data/logs /app/data/cache /app/data/uploads /app/data/models && \
    chown -R app:app /app/data

# Switch to app user
USER app

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Production command with Uvicorn
CMD ["uvicorn", "backend.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]

# ============ MODEL-DOWNLOADER STAGE ============
# Special stage for downloading AI models into a volume
FROM dependencies AS model-downloader

# Copy model download script
COPY scripts/download_models.py ./scripts/

# Set cache directory
ENV HF_HOME=/models/.cache/huggingface \
    TRANSFORMERS_CACHE=/models/.cache/huggingface

# Create models directory
RUN mkdir -p /models/.cache/huggingface

# This stage is meant to be run once to populate the model volume
# Usage: docker-compose --profile setup run model-downloader
CMD ["python3", "scripts/download_models.py"]

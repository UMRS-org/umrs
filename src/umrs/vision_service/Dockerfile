FROM python:3.11-slim AS base

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager) — kept at runtime for custom
# adapter dependency installation via requirements.txt auto-discovery.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

WORKDIR /app

# Install PyTorch with CUDA 13.0 support (GPU inference on Blackwell/aarch64)
# The cu130 wheels bundle CUDA runtime libs; host driver is mounted via CDI.
RUN uv pip install \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies
COPY src/umrs/vision_service/requirements-vision.txt .
RUN uv pip install -r requirements-vision.txt

# Install decord for future video support (fall back to decord2 shim if needed)
RUN uv pip install decord==0.6.0 || \
    { uv pip install decord2==3.2.0; }

# Copy UMRS source (models, vision service, etc.)
COPY src/umrs/ /app/umrs/

# BentoML service entrypoint
EXPOSE 3000
CMD ["bentoml", "serve", "umrs.vision_service.service:VisionModelService", "--port", "3000"]

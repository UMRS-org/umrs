FROM python:3.11-slim AS base

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv from PyPI instead of a GHCR multi-stage copy; compute-node
# rootless podman builds can pull Python packages but may not resolve GHCR.
RUN python -m pip install --no-cache-dir uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1

WORKDIR /app

# Install PyTorch with CUDA 13.0 support (GPU inference on Blackwell/aarch64)
# The cu130 wheels bundle CUDA runtime libs; host driver is mounted via CDI.
RUN uv pip install \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies
COPY src/umrs/vision_service/requirements-vision.txt .
RUN uv pip install -r requirements-vision.txt
RUN uv pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git@8e451d5eb43c817b64ae7577fb7b9ae223db88a9"

# Copy UMRS source (models, vision service, etc.)
COPY src/umrs/ /app/umrs/

# BentoML service entrypoint
EXPOSE 3000
CMD ["bentoml", "serve", "umrs.vision_service.service:VisionModelService", "--port", "3000"]

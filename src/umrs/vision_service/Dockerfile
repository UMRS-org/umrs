FROM python:3.11-slim AS base

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 13.0 support (GPU inference on Blackwell/aarch64)
# The cu130 wheels bundle CUDA runtime libs; host driver is mounted via CDI.
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install remaining Python dependencies
COPY src/umrs/vision_service/requirements-vision.txt .
RUN pip install --no-cache-dir -r requirements-vision.txt

# Copy UMRS source (models, vision service, etc.)
COPY src/umrs/ /app/umrs/

# BentoML service entrypoint
EXPOSE 3000
CMD ["bentoml", "serve", "umrs.vision_service.service:VisionModelService", "--port", "3000"]

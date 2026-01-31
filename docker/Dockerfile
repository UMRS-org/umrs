# Dockerfile for UMRS server (Python)
FROM python:3.10-slim

WORKDIR /app

# Build argument to select requirements file (default: requirements.txt, use requirements-ci.txt for CI)
ARG REQUIREMENTS_FILE=requirements.txt

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy both requirements files (ci version may not exist locally)
COPY ./src/requirements*.txt /app/
RUN pip install --no-cache-dir -r /app/${REQUIREMENTS_FILE} && \
    python -m spacy download en_core_web_sm

# Install MinIO Client (mc) for user/bucket management used by minio.py
RUN curl -sSL -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x /usr/local/bin/mc \
    && mc --version || true

COPY ./src /app/src
COPY ./doc /app/doc
COPY ./docker/data /root/data

# Shared models directory used by both server and worker (simulating
# the shared filesystem on HPC where model weights live).
RUN mkdir -p /shared/models \
    && cp /root/data/models/nvidia_resnet50_200821.pth.tar /shared/models/nvidia_resnet50_200821.pth.tar || true

# Use the server configuration (server schema) and include users_json
COPY ./docker/conf_umrs.json /root/conf_umrs.json
# Use docker user_files (mounted in compose for test container); server image does not need users_json
COPY ./docker/user_files /root/user_files

# s3cfg files
COPY ./docker/s3cfg /root/.s3cfg
COPY ./docker/s3cfg_user1 /root/.s3cfg_user1
COPY ./docker/s3cfg_user2 /root/.s3cfg_user2

# specify the umrs config directory
ENV UMRS_CONFIG /root/
# or use the default /root/.config/umrs/

EXPOSE 8000
CMD ["python", "src/server_umrs.py", "-c", "/root/conf_umrs.json"]

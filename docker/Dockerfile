# Dockerfile for UMRS server (Python)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -m spacy download en_core_web_sm

# Install MinIO Client (mc) for user/bucket management used by minio.py
RUN curl -sSL -o /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x /usr/local/bin/mc \
    && mc --version || true

COPY ./src /app/src
COPY ./doc /app/doc
COPY ./docker/data /app/data

# Use the server configuration (server schema) and include users_json
COPY ./docker/conf_umrs.json /app/conf_umrs.json
# Use docker user_files (mounted in compose for test container); server image does not need users_json
COPY ./docker/user_files /app/user_files

# s3cfg files 
COPY ./docker/s3cfg /root/.s3cfg
COPY ./docker/s3cfg_user1 /root/.s3cfg_user1
COPY ./docker/s3cfg_user2 /root/.s3cfg_user2

EXPOSE 8000
CMD ["python", "src/server_umrs.py"]

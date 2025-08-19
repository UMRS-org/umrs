# Dockerfile for UMRS server (Python)
FROM python:3.10-slim

WORKDIR /app


# Install system dependencies and MinIO Client (mc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -m spacy download en_core_web_sm

COPY ./src /app/src
#COPY ./Infinia_demo/doc /app/doc
#COPY ./Infinia_demo/demo.md /app/demo.md
COPY ./data /app/data

# copy the model directory
COPY ./models /app/models
# copy the initial configuration file
COPY ./docker/conf_umrs_init.json /app/conf_umrs_init.json
COPY ./docker/conf_umrs_ejeannot.json /app/docker/conf_umrs_ejeannot.json
COPY ./docker/s3cfg /root/.s3cfg
# copy SSL certificates for HTTPS support


# Copy SSL certificates for HTTPS support
COPY ./docker/ssl /app/ssl


# Trust MinIO self-signed certificate as CA and force CA bundle update
COPY ./docker/ssl/public.crt /usr/local/share/ca-certificates/minio_ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates --fresh \
    && rm -rf /var/lib/apt/lists/*

# Force Python to use the correct CA bundle globally
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

EXPOSE 8000
EXPOSE 8443
CMD ["python", "src/server_umrs.py", "--conf", "conf_umrs_init.json", "--use-https", "--ssl-cert", "/app/ssl/public.crt", "--ssl-key", "/app/ssl/private.key", "--port", "8443"]

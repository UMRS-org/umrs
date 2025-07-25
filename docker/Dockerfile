# Dockerfile for UMRS server (Python)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
COPY ./docker/s3cfg /root/.s3cfg
# copy env file contains environment variables
COPY ./docker/env.default /app/src/.env

EXPOSE 8000
CMD ["python", "src/server_umrs.py", "--conf", "conf_umrs_init.json", "--web"]

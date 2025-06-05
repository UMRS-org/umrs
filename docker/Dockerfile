# Dockerfile for UMRS server (Python)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./Infinia_demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ./Infinia_demo/src /app/src
COPY ./Infinia_demo/doc /app/doc
COPY ./Infinia_demo/demo.md /app/demo.md
COPY ./data /app/data
#COPY ./data/models /app/models # this models is a symlink from NIM_test/models
COPY ./NIM_test/models /app/models
COPY ./docker/conf_umrs.json /app/conf_umrs.json

EXPOSE 8000
CMD ["python", "src/server_umrs.py"]

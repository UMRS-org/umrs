#!/bin/bash
cd /home/users/masaru.nagaso/umrs-deploy/deploy/rccs-cloud
export DB_SUPERUSER=umrs
export DB_SUPERUSER_PASSWORD=umrs_rccs
export POSTGRES_USER=umrs
export POSTGRES_PASSWORD=umrs_rccs
exec /home/users/masaru.nagaso/umrs-deploy/deploy/rccs-cloud/.venv/bin/python3 \
    /home/users/masaru.nagaso/umrs-deploy/src/server_umrs.py \
    -c /home/users/masaru.nagaso/umrs-deploy/deploy/rccs-cloud/conf_umrs.json

#!/bin/sh
mc alias set myminio http://127.0.0.1:9000 ejeannot jemdaz-satwyz-xojtA6


mc admin config set myminio notify_postgres:primary \
    format=namespace \
    connection_string="postgres://ejeannot@127.0.0.1:5432/minio_events?sslmode=disable" \
    table="events" \
    queue_limit="10000" \
    max_open_connections="2"


mc admin config set myminio notify_postgres:primary enable="on"

mc admin service restart myminio

sleep 5

mc admin config get myminio notify_postgres

mc ls myminio/test-postgres || mc mb myminio/test-postgres

mc event list myminio/test-postgres || mc event add myminio/test-postgres arn:minio:sqs::primary:postgresql --event put,delete

mc event list myminio/test-postgres

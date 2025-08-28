#!/bin/bash
mc alias set myminio http://127.0.0.1:9000 ejeannot jemdaz-satwyz-xojtA6

# # Apply webhook settings with additional parameters
# mc admin config set myminio notify_webhook:primary \
#    endpoint="http://127.0.0.1:5000/minio-events" \
#    queue_limit="10000" \
#    auth_token="Bearer mysecrettoken" \
#    queue_dir="/tmp/minio/events" \
#    client_cert="/Users/ejeannot/.minio/certs/client-cert.pem" \
#    client_key="/Users/ejeannot/.minio/certs/client-key.pem" \
#    comment="ZeroMQ webhook for S3 event notifications"
# Apply webhook settings with additional parameters
mc admin config set myminio notify_webhook:primary \
   endpoint="http://127.0.0.1" \
   queue_limit="10000" \
   # auth_token="Bearer mysecrettoken" \
   # queue_dir="/tmp/minio/events" \
   # client_cert="/Users/ejeannot/.minio/certs/client-cert.pem" \
   # client_key="/Users/ejeannot/.minio/certs/client-key.pem" \
   # comment="ZeroMQ webhook for S3 event notifications"


mc admin config set myminio notify_webhook:primary enable="on"
# arn:<partition>:<service>:<region>:<accountID>:<resource>'.
# mc event add myminio/test-wh arn:minio:sqs::primary:webhook --event put,delete


mc admin service restart myminio

sleep 5

mc admin config get myminio notify_webhook

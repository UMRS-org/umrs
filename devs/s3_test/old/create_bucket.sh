#!/usr/bin/sh

# Check if the user provided a bucket name
if [ -z "$1" ]; then
    echo "❌ Usage: $0 <bucket-name>"
    exit 1
fi

bucket_name=$1  # ✅ Use the first command-line argument as the bucket name

# Read credentials from the configuration file
access_key=$(awk '/s3_key/ {print $2}' ~/infinia_cred.yml)
secret_key=$(awk '/s3_secret/ {print $2}' ~/infinia_cred.yml)

vhost="ejeannot-s3-vm.infinia.lab.local"
port="8111"

# ✅ Create the specified bucket
s3cmd --host=$vhost:$port \
    --host-bucket=$vhost:$port \
    --access_key=$access_key \
    --secret_key=$secret_key \
    --bucket-location=ap-northeast-1 \
    --ssl --no-check-certificate \
    mb s3://$bucket_name/

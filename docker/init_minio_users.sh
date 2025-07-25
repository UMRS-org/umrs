#!/bin/bash
set -e

# This script initializes MinIO users and policies for multi-user UMRS setup
# Requires MinIO Client (mc) installed and configured to access the MinIO server

# MinIO server alias (must be configured with mc alias set)
MINIO_ALIAS="localminio"
MINIO_URL="http://minio:9000"
MINIO_ROOT_USER="minioadmin"
MINIO_ROOT_PASSWORD="minioadmin"

# User credentials
USER1="user1"
USER1_SECRET="user1secret"
USER2="user2"
USER2_SECRET="user2secret"

# Setup mc alias if not already present
mc alias set $MINIO_ALIAS $MINIO_URL $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD || true

# Create users
USERS=("user1" "user2")
SECRETS=("user1secret" "user2secret")

# Create users and policies for each user (full access to their buckets and ListAllMyBuckets)
for i in "${!USERS[@]}"; do
  USER="${USERS[$i]}"
  SECRET="${SECRETS[$i]}"
  mc admin user add $MINIO_ALIAS $USER $SECRET
  cat > /tmp/${USER}_policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListAllMyBuckets"],
      "Resource": ["arn:aws:s3:::*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:*"],
      "Resource": [
        "arn:aws:s3:::${USER}*",
        "arn:aws:s3:::${USER}*/*"
      ]
    }
  ]
}
EOF
  mc admin policy create $MINIO_ALIAS ${USER}-policy /tmp/${USER}_policy.json
  mc admin policy attach $MINIO_ALIAS ${USER}-policy --user $USER
done

# Output credentials for updating s3cfg files
echo ""
echo "S3 Credentials for all users:"
for i in "${!USERS[@]}"; do
  echo "  Access Key: ${USERS[$i]}"
  echo "  Secret Key: ${SECRETS[$i]}"
  echo ""
done
echo "Update /home/user1/.s3cfg and /home/user2/.s3cfg accordingly."
echo "MinIO users and policies created."

#!/bin/bash

# Multi-user test script for UMRS system
# This script tests the multi-user functionality by:
# 1. Creating users and buckets on S3
# 2. Processing texts and images for each user
# 3. Running retrieval tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== UMRS Multi-User Test Script ===${NC}"

# Function to wait for services to be ready
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Waiting for $service_name to be ready...${NC}"

    while ! nc -z $host $port; do
        if [ $attempt -eq $max_attempts ]; then
            echo -e "${RED}Failed to connect to $service_name after $max_attempts attempts${NC}"
            exit 1
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done

    echo -e "${GREEN}$service_name is ready!${NC}"
}

# Function to create MinIO user and policy
#create_minio_user() {
#    local username=$1
#    local password=$2
#    local access_key=$3
#    local secret_key=$4
#
#    echo -e "${YELLOW}Creating MinIO user: $username${NC}"
#
#    # Create user policy (allow full access to user's buckets)
#    cat > /tmp/${username}_policy.json << EOF
#{
#    "Version": "2012-10-17",
#    "Statement": [
#        {
#            "Effect": "Allow",
#            "Action": [
#                "s3:*"
#            ],
#            "Resource": [
#                "arn:aws:s3:::${username}-bucket-*",
#                "arn:aws:s3:::${username}-bucket-*/*"
#            ]
#        }
#    ]
#}
#EOF
#
#    # Note: In a real deployment, you would use MinIO client (mc) to create users
#    # For now, we'll use the admin credentials and create buckets directly
#    echo "User $username would be created in production MinIO setup"
#}

# Buckets are created and managed by UMRS client automatically based on config files.
# No direct S3 commands are needed.

# Function to run UMRS client test
run_umrs_test() {
    local username=$1
    local config_file=$2
    local password=$3
    local s3_secret=$4

    su - $username <<EOF
    echo "==> whoami"
    whoami
    echo "==> Running UMRS client for user: $username"
    echo "==> python3 -m venv /tmp/venv"
    python3 -m venv /tmp/venv
    echo "==> Activating venv"
    . /tmp/venv/bin/activate
    echo "==> pip install requirements-client.txt"
    pip install --break-system-packages -r /client/requirements-client.txt

    # Create user via UMRS admin API
    export UMRS_ADMIN_TOKEN="test_admin_token"
    echo "==> UMRS_ADMIN_TOKEN: $UMRS_ADMIN_TOKEN"
    echo "==> python3 /home/$username/src/client_umrs.py init-user ..."
    python3 /home/$username/src/client_umrs.py init-user --server-url http://umrs-server:8000 --admin-token "\$UMRS_ADMIN_TOKEN" --username $username --password $password --s3-secret $s3_secret

    echo "==> which python3"
    which python3
    if [ "$username" = "user2" ]; then
        # Use create-buckets-from-config for user2
        echo "==> python3 /home/$username/src/client_umrs.py create-buckets-from-config ..."
        python3 /home/$username/src/client_umrs.py create-buckets-from-config \
          --server-url http://umrs-server:8000 \
          --token "$username" \
          --username $username \
          -c $config_file

        # Upload sample text to text bucket
        echo "==> python3 /home/$username/src/client_umrs.py upload (text) ..."
        python3 /home/$username/src/client_umrs.py upload --token "$username" -c $config_file /data/texts/beast_in_the_jungle.txt ${username}-bucket-2

        # Upload sample image to image bucket
        echo "==> python3 /home/$username/src/client_umrs.py upload (image) ..."
        python3 /home/$username/src/client_umrs.py upload --token "$username" -c $config_file /data/img_test/cat.jpeg ${username}-bucket-1
    else
        # Manual bucket creation for user1 (default/override example)
        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (text) ..."
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url http://umrs-server:8000 \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-2 \
          --xp-name "Text and Images" \
          --modality text \
          --model "all-mpnet-base-v2"

        echo "==> python3 /home/$username/src/client_umrs.py upload (text) ..."
        python3 /home/$username/src/client_umrs.py upload --token "$username" -c $config_file /data/texts/beast_in_the_jungle.txt ${username}-bucket-2

        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (image) ..."
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url http://umrs-server:8000 \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-1 \
          --xp-name "Text and Images" \
          --modality image \
          --model "clip-ViT-B-32"

        echo "==> python3 /home/$username/src/client_umrs.py upload (image) ..."
        python3 /home/$username/src/client_umrs.py upload --token "$username" -c $config_file /data/img_test/cat.jpeg ${username}-bucket-1
    fi

    # Test list-buckets API for this user
    echo "==> Testing list-buckets for $username..."
    echo "==> python3 /home/$username/src/client_umrs.py list-buckets ..."
    python3 /home/$username/src/client_umrs.py list-buckets --server-url http://umrs-server:8000 --token "$username" --username $username
EOF
}

# Main execution
main() {
    echo -e "${GREEN}Starting multi-user test...${NC}"

    # Wait for services to be ready
    wait_for_service "MinIO" "minio" 9000
    wait_for_service "PostgreSQL" "postgres" 5432
    wait_for_service "UMRS Server" "umrs-server" 8000

    # Install netcat for service checking
    if ! command -v nc &> /dev/null; then
        echo -e "${YELLOW}Installing netcat...${NC}"
        apt-get update && apt-get install -y netcat-openbsd
    fi


    # Create users via UMRS admin API
    echo -e "${YELLOW}Creating users via UMRS admin API...${NC}"
    export UMRS_ADMIN_TOKEN="test_admin_token"
    # Wait for UMRS server API to be ready
    until curl -sf http://umrs-server:8000/health; do
        echo "Waiting for UMRS server API..."; sleep 2;
    done

    # Test User 1
    echo -e "${GREEN}=== Testing User 1 ===${NC}"
    run_umrs_test "user1" "/home/user1/conf_umrs.json" "user1pass" "user1secret"

    # Test User 2
    echo -e "${GREEN}=== Testing User 2 ===${NC}"
    run_umrs_test "user2" "/home/user2/conf_umrs.json" "user2pass" "user2secret"

    echo -e "${GREEN}=== Multi-user test completed! ===${NC}"
}

# Run main function
main "$@"

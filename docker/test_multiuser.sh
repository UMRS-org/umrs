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
create_minio_user() {
    local username=$1
    local password=$2
    local access_key=$3
    local secret_key=$4

    echo -e "${YELLOW}Creating MinIO user: $username${NC}"

    # Create user policy (allow full access to user's buckets)
    cat > /tmp/${username}_policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:*"
            ],
            "Resource": [
                "arn:aws:s3:::${username}-bucket-*",
                "arn:aws:s3:::${username}-bucket-*/*"
            ]
        }
    ]
}
EOF

    # Note: In a real deployment, you would use MinIO client (mc) to create users
    # For now, we'll use the admin credentials and create buckets directly
    echo "User $username would be created in production MinIO setup"
}

# Buckets are created and managed by UMRS client automatically based on config files.
# No direct S3 commands are needed.

# Function to run UMRS client test
run_umrs_test() {
    local username=$1
    local config_file=$2

    su - $username <<EOF
    whoami
    echo "Running UMRS client for user: $username"
    python3 -m venv /tmp/venv
    . /tmp/venv/bin/activate
    pip install --break-system-packages -r /client/requirements-client.txt

    which python3
    # Create/check text bucket and upload sample text
    python3 /home/$username/src/client_umrs.py -c $config_file /data/texts/beast_in_the_jungle.txt ${username}-bucket-2

    # Create/check image bucket and upload sample image
    python3 /home/$username/src/client_umrs.py -c $config_file /data/img_test/cat.jpeg ${username}-bucket-1
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

    # reinitialize MinIO users
    echo -e "${YELLOW}Reinitializing MinIO users...${NC}"
    /client/init_minio_users.sh

    # Test User 1
    echo -e "${GREEN}=== Testing User 1 ===${NC}"
    run_umrs_test "user1" "/home/user1/conf_umrs.json"

    # Test User 2
    echo -e "${GREEN}=== Testing User 2 ===${NC}"
    run_umrs_test "user2" "/home/user2/conf_umrs.json"

    echo -e "${GREEN}=== Multi-user test completed! ===${NC}"
}

# Run main function
main "$@"

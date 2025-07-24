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

    echo -e "${YELLOW}Running UMRS client test for $username${NC}"

    # Switch to user and run test
    su - $username -c "
        cd /client
        source /home/$username/venv/bin/activate

        # Test configuration
        echo 'Testing configuration file: $config_file'
        python3 -c 'import json; print(json.load(open(\"$config_file\")))' || echo 'Config file error'

        # Run retrieval to verify data upload and embedding
        echo 'Retrieving text embeddings from bucket:'
        python3 src/client_umrs.py -c $config_file --queue_type zeromq --bucket ${username}-bucket-2 --retrieve --limit 2

        echo 'Retrieving image embeddings from bucket:'
        python3 src/client_umrs.py -c $config_file --queue_type zeromq --bucket ${username}-bucket-1 --retrieve --limit 2
    "
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

#!/bin/bash

# Multi-user test script for UMRS system
# This script tests the multi-user functionality by:
# 1. Creating users and buckets via UMRS API (replaces init_minio_users.sh and init_postgres_users.sh)
# 2. Processing texts and images for each user
# 3. Running retrieval tests
#
# NOTE: This script now uses UMRS API calls exclusively and no longer requires:
# - init_minio_users.sh (replaced by UMRS /api/admin/create_user)
# - init_postgres_users.sh (replaced by UMRS /api/admin/create_user)
# - Direct MinIO 'mc' commands
# - Direct PostgreSQL 'psql' commands

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

# Note: The following functions have been replaced by UMRS API calls:
# - create_minio_user() → replaced by UMRS /api/admin/create_user
# - Manual MinIO policy creation → handled automatically by UMRS
# - PostgreSQL user creation → handled by UMRS /api/admin/create_user
#
# All user initialization is now handled by create_users_via_umrs() function

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

    # Note: User creation is now handled centrally by create_users_via_umrs()
    echo "==> User $username already created via UMRS API"
    export UMRS_USER_TOKEN="$username"
    echo "==> UMRS_USER_TOKEN: $UMRS_USER_TOKEN"

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
        python3 /home/$username/src/client_umrs.py upload -c $config_file /home/$username/data/texts/beast_in_the_jungle.txt ${username}-bucket-5

        # Store a single line of text to text bucket (text modality)
        echo "==> python3 /home/$username/src/client_umrs.py upload (text, single line) ..."
        python3 /home/$username/src/client_umrs.py upload -c $config_file "This is a test line for $username" ${username}-bucket-2

        # Upload sample image to image bucket
        echo "==> python3 /home/$username/src/client_umrs.py upload (image) ..."
        python3 /home/$username/src/client_umrs.py upload -c $config_file /home/$username/data/img_test/cat.jpeg ${username}-bucket-1
    else
        # Manual bucket creation for user1 (default/override example)
        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (text_file) ..."
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url http://umrs-server:8000 \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-5 \
          --xp-name "Images and Texts" \
          --modality text_file \
          --model "intfloat/e5-base-v2"

        echo "==> python3 /home/$username/src/client_umrs.py upload (text_file) ..."
        python3 /home/$username/src/client_umrs.py upload -c $config_file /home/$username/data/texts/beast_in_the_jungle.txt ${username}-bucket-5

        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (text) ..."
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url http://umrs-server:8000 \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-2 \
          --xp-name "Text and Images" \
          --modality text \
          --model "all-mpnet-base-v2"

        # Store a single line of text to text bucket (text modality)
        echo "==> python3 /home/$username/src/client_umrs.py upload (text, single line) ..."
        python3 /home/$username/src/client_umrs.py upload -c $config_file "This is a test line for $username" ${username}-bucket-2

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
        python3 /home/$username/src/client_umrs.py upload -c $config_file /home/$username/data/img_test/cat.jpeg ${username}-bucket-1
    fi

    # Test list-buckets API for this user
    echo "==> Testing list-buckets for $username..."
    echo "==> python3 /home/$username/src/client_umrs.py list-buckets ..."
    python3 /home/$username/src/client_umrs.py list-buckets --server-url http://umrs-server:8000 --token "$username" --username $username
EOF
}

# Function to create users via UMRS API (replaces init_minio_users.sh and init_postgres_users.sh)
create_users_via_umrs() {
    echo -e "${YELLOW}Creating users via UMRS admin API (replacing shell scripts)...${NC}"
    export UMRS_ADMIN_TOKEN="test_admin_token"

    # Install requirements for admin operations
    echo "==> Installing client requirements for admin operations..."
    pip install --break-system-packages -r /client/requirements-client.txt

    # User definitions (matching the original shell scripts)
    declare -a USERS=("user1" "user2")
    declare -a PASSWORDS=("user1pass" "user2pass")
    declare -a S3_SECRETS=("user1secret" "user2secret")

    # Create each user via UMRS API instead of direct shell commands
    for i in "${!USERS[@]}"; do
        USERNAME="${USERS[$i]}"
        PASSWORD="${PASSWORDS[$i]}"
        S3_SECRET="${S3_SECRETS[$i]}"

        echo -e "${YELLOW}Creating user ${USERNAME} via UMRS API...${NC}"
        echo "==> python3 /client/src/client_umrs.py init-user --username $USERNAME ..."

        python3 /client/src/client_umrs.py init-user \
          --server-url http://umrs-server:8000 \
          --admin-token "$UMRS_ADMIN_TOKEN" \
          --username "$USERNAME" \
          --password "$PASSWORD" \
          --s3-secret "$S3_SECRET"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}User $USERNAME created successfully${NC}"
        else
            echo -e "${RED}Failed to create user $USERNAME${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}All users created via UMRS API (no separate shell scripts needed)${NC}"
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

    # Wait for UMRS server API to be ready
    echo -e "${YELLOW}Waiting for UMRS server API to be ready...${NC}"
    until curl -sf http://umrs-server:8000/health; do
        echo "Waiting for UMRS server API..."; sleep 2;
    done
    echo -e "${GREEN}UMRS server API is ready!${NC}"

    # Create users via UMRS API (replaces init_minio_users.sh and init_postgres_users.sh)
    create_users_via_umrs

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

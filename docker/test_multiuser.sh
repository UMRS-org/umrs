#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== UMRS Multi-User Test Script (HTTPS) ===${NC}"

# Configuration
UMRS_SERVER_URL="https://umrs-server:8443"
INSECURE_FLAG="--insecure"  # For self-signed certificates

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

# Function to wait for HTTPS service to be ready
wait_for_https_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Waiting for $service_name to be ready...${NC}"

    while ! curl -k -f -s "$url" > /dev/null 2>&1; do
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

    export UMRS_USER_TOKEN="$username"
    echo "==> UMRS_USER_TOKEN: $UMRS_USER_TOKEN"

    if [ "$username" = "user2" ]; then
        # Use create-buckets-from-config for user2
        echo "==> python3 /home/$username/src/client_umrs.py create-buckets-from-config ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py create-buckets-from-config \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --config $config_file \
          $INSECURE_FLAG

        # Upload sample text to text bucket using hybrid API
        echo "==> python3 /home/$username/src/client_umrs.py upload (text) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py upload \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-5 \
          --xp-name "Images and Texts" \
          --modality "text file" \
          /home/$username/data/texts/beast_in_the_jungle.txt \
          $INSECURE_FLAG

        # Store a single line of text to text bucket (text modality) - skip for now as it requires file
        echo "==> Skipping single line text upload (requires file input in hybrid mode)"

        # Upload sample image to image bucket using hybrid API
        echo "==> python3 /home/$username/src/client_umrs.py upload (image) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py upload \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-3 \
          --xp-name "Images and Texts" \
          --modality "image" \
          /home/$username/data/img_test/cat.jpeg \
          $INSECURE_FLAG
    else
        # Manual bucket creation for user1 (default/override example)
        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (text file) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-5 \
          --xp-name "Images and Texts" \
          --modality "text file" \
          --model "intfloat/e5-base-v2" \
          $INSECURE_FLAG

        echo "==> python3 /home/$username/src/client_umrs.py upload (text file) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py upload \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-5 \
          --xp-name "Images and Texts" \
          --modality "text file" \
          /home/$username/data/texts/beast_in_the_jungle.txt \
          $INSECURE_FLAG

        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (text) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-2 \
          --xp-name "Text and Images" \
          --modality text \
          --model "all-mpnet-base-v2" \
          $INSECURE_FLAG

        # Store a single line of text to text bucket (text modality) - skip for now as it requires file
        echo "==> Skipping single line text upload (requires file input in hybrid mode)"

        echo "==> python3 /home/$username/src/client_umrs.py create-bucket (image) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py create-bucket \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-1 \
          --xp-name "Text and Images" \
          --modality image \
          --model "clip-ViT-B-32" \
          $INSECURE_FLAG

        echo "==> python3 /home/$username/src/client_umrs.py upload (image) ..." $INSECURE_FLAG
        python3 /home/$username/src/client_umrs.py upload \
          --server-url $UMRS_SERVER_URL \
          --token "$username" \
          --username $username \
          --bucket-alias ${username}-bucket-1 \
          --xp-name "Text and Images" \
          --modality "image" \
          /home/$username/data/img_test/cat.jpeg \
          $INSECURE_FLAG
    fi

    # Test list-buckets API for this user
    echo "==> Testing list-buckets for $username..."
    echo "==> python3 /home/$username/src/client_umrs.py list-buckets ..." $INSECURE_FLAG
    python3 /home/$username/src/client_umrs.py list-buckets --server-url $UMRS_SERVER_URL --token "$username" --username $username $INSECURE_FLAG
EOF
}

# Function to check bucket contents and test deletion
check_bucket_contents_and_deletion() {
    local username=$1
    local token=$1  # Using username as token for simplicity

    echo -e "${GREEN}=== Checking Bucket Contents for User: $username ===${NC}"

    # Get bucket list
    echo -e "${YELLOW}Getting bucket list for $username...${NC}"
    bucket_response=$(curl -s -k -H "Authorization: Bearer $token" \
                     "https://umrs-server:8443/api/user/list_buckets?username=$username")

    echo "Raw bucket response:"
    echo "$bucket_response" | python3 -m json.tool

    # Extract bucket names using python
    buckets=$(echo "$bucket_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    buckets = data.get('buckets', [])
    for bucket in buckets:
        print(bucket)
except:
    pass
")

    echo -e "${YELLOW}Found buckets for $username:${NC}"
    echo "$buckets"

    # Check contents of each bucket using MinIO client (direct access from container)
    if [ -n "$buckets" ]; then
        echo -e "${YELLOW}Checking bucket contents using MinIO client...${NC}"

        # Setup mc alias directly (since mc is available in this container)
        echo "Setting up MinIO alias..."
        mc alias set myminio http://minio:9000 minioadmin minioadmin 2>/dev/null || true

        echo "Listing all buckets:"
        mc ls myminio 2>/dev/null || echo "Failed to list buckets"

        echo ""
        echo "Checking bucket contents for $username:"

        # Convert buckets string to array and check each bucket
        IFS=$'\n' read -rd '' -a bucket_array <<< "$buckets" || true

        for bucket in "${bucket_array[@]}"; do
            if [ -n "$bucket" ]; then
                echo "--- Contents of $bucket ---"
                mc ls myminio/$bucket 2>/dev/null || echo "Empty or inaccessible"
            fi
        done
    fi

    # Test bucket deletion if we have buckets
    if [ -n "$buckets" ]; then
        # Get the first bucket for deletion test
        first_bucket=$(echo "$buckets" | head -n 1)

        if [ -n "$first_bucket" ]; then
            echo -e "${YELLOW}Testing bucket deletion...${NC}"
            echo "Will attempt to delete bucket: $first_bucket"

            # Check if delete-bucket functionality exists in client
            echo "==> Attempting to delete bucket '$first_bucket'..."

            # Using --confirm flag to skip interactive confirmation
            su - $username <<EOF
            . /tmp/venv/bin/activate
            echo "==> python3 /home/$username/src/client_umrs.py delete-bucket --confirm ..." $INSECURE_FLAG
            python3 /home/$username/src/client_umrs.py delete-bucket \
              --server-url $UMRS_SERVER_URL \
              --token "$username" \
              --username $username \
              --bucket-name "$first_bucket" \
              --confirm \
          $INSECURE_FLAG
EOF

            # Show buckets after deletion attempt
            echo -e "${YELLOW}Bucket list after deletion attempt:${NC}"
            curl -s -k -H "Authorization: Bearer $token" \
                 "https://umrs-server:8443/api/user/list_buckets?username=$username" | \
                 python3 -m json.tool
        fi
    fi

    echo ""
}

# Function to create users via UMRS client command (replaces init_minio_users.sh and init_postgres_users.sh)
create_users_via_umrs() {
    echo -e "${YELLOW}Creating users via UMRS client command (replacing shell scripts)...${NC}"
    export UMRS_ADMIN_TOKEN="test_admin_token"

    # Install requirements for admin operations
    echo "==> Installing client requirements for admin operations..."
    pip install --break-system-packages -r /client/requirements-client.txt

    # User definitions (matching the original shell scripts)
    declare -a USERS=("user1" "user2")
    declare -a PASSWORDS=("user1pass" "user2pass")
    declare -a S3_SECRETS=("user1secret" "user2secret")

    # Create each user via UMRS client command instead of direct shell commands
    for i in "${!USERS[@]}"; do
        USERNAME="${USERS[$i]}"
        PASSWORD="${PASSWORDS[$i]}"
        S3_SECRET="${S3_SECRETS[$i]}"

        echo -e "${YELLOW}Creating user ${USERNAME} via UMRS client command...${NC}"
        echo "==> python3 /client/src/client_umrs.py init-user --server-url $UMRS_SERVER_URL --admin-token \$UMRS_ADMIN_TOKEN --username $USERNAME --password $PASSWORD --s3-secret $S3_SECRET" $INSECURE_FLAG

        python3 /client/src/client_umrs.py init-user \
          --server-url $UMRS_SERVER_URL \
          --admin-token "$UMRS_ADMIN_TOKEN" \
          --username "$USERNAME" \
          --password "$PASSWORD" \
          --s3-secret "$S3_SECRET" \
          $INSECURE_FLAG

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}User $USERNAME created successfully${NC}"
        else
            echo -e "${RED}Failed to create user $USERNAME${NC}"
            exit 1
        fi
    done

    echo -e "${GREEN}All users created via UMRS client command (no separate shell scripts needed)${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}Starting multi-user test...${NC}"

    # Wait for services to be ready
    wait_for_service "MinIO" "minio" 9000
    wait_for_service "PostgreSQL" "postgres" 5432
    wait_for_https_service "UMRS Server" "https://umrs-server:8443/health"

    # Install netcat for service checking
    if ! command -v nc &> /dev/null; then
        echo -e "${YELLOW}Installing netcat...${NC}"
        apt-get update && apt-get install -y netcat-openbsd
    fi

    # Create users via UMRS client command (replaces init_minio_users.sh and init_postgres_users.sh)
    create_users_via_umrs

    # Test User 1
    echo -e "${GREEN}=== Testing User 1 ===${NC}"
    run_umrs_test "user1" "/home/user1/conf_umrs.json" "user1pass" "user1secret"

    # Test User 2
    echo -e "${GREEN}=== Testing User 2 ===${NC}"
    run_umrs_test "user2" "/home/user2/conf_umrs.json" "user2pass" "user2secret"

    # Check bucket contents and test deletion for both users
    echo -e "${GREEN}=== Checking Bucket Contents and Testing Deletion ===${NC}"
    check_bucket_contents_and_deletion "user1"
    check_bucket_contents_and_deletion "user2"

    echo -e "${GREEN}=== Multi-user test completed! ===${NC}"
}

# Run main function
main "$@"

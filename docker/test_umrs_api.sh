#!/bin/bash

# UMRS API automated test script for Docker container (Bash version)
# Tests all endpoints for two users using their config files and the data directory.
# Also tests admin endpoints for user and bucket creation.


set -e

CONFIG_PATH=${1:-conf_umrs_user1.json}
DATA_DIR=${2:-./data}
PY_CLIENT="python3 client_umrs_api.py"
BASE_URL=${UMRS_API_URL:-https://umrs-server:8443}

function test_admin_endpoints() {
  local config_path="$1"
  echo -e "\n=== Testing admin endpoints for user config: $config_path ==="
  echo "[create_user]"
  $PY_CLIENT --base_url "$BASE_URL" create_user "$config_path"
  echo
  echo "[create_buckets]"
  $PY_CLIENT --base_url "$BASE_URL" create_buckets "$config_path"
  echo
}

function test_all_endpoints() {
  local config_path="$1"
  local data_dir="$2"
  local username=$(jq -r '.username' "$config_path")
  echo -e "\n=== Testing for user: $username ==="

  echo "[health]"
  $PY_CLIENT --base_url "$BASE_URL" health

  echo "[status]"
  $PY_CLIENT --base_url "$BASE_URL" status

  echo "[list_buckets]"
  $PY_CLIENT --base_url "$BASE_URL" list_buckets "$username"

  # Upload text
  BUCKET_ALIAS=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text") | .s3.bucket_alias' "$config_path")
  echo "[upload_text]"
  $PY_CLIENT --base_url "$BASE_URL" upload_text "Hello from test script!" "$BUCKET_ALIAS" "$username" '{"test":true}'

  # Upload image
  IMG_PATH="$data_dir/img/Cat_1.jpeg"
  BUCKET_ALIAS_IMG=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="image") | .s3.bucket_alias' "$config_path")
  echo "[upload_image]"
  $PY_CLIENT --base_url "$BASE_URL" upload_image "$IMG_PATH" "$BUCKET_ALIAS_IMG" "$username" '{"test":true}'

  # Upload textfile
  TXTFILE_PATH="$data_dir/texts/beast_in_the_jungle.txt"
  BUCKET_ALIAS_TXT=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text file") | .s3.bucket_alias' "$config_path")
  echo "[upload_textfile]"
  $PY_CLIENT --base_url "$BASE_URL" upload_textfile "$TXTFILE_PATH" "$BUCKET_ALIAS_TXT" "$username" '{"test":true}'

  echo "[recompute_embeddings]"
  $PY_CLIENT --base_url "$BASE_URL" recompute_embeddings

  echo "[update_embeddings]"
  $PY_CLIENT --base_url "$BASE_URL" update_embeddings

  echo "[check_consistency]"
  $PY_CLIENT --base_url "$BASE_URL" check_consistency

  echo -e "=== All tests completed for $username ===\n"
}

echo "--- BEGIN: Running test_admin_endpoints ---"
test_admin_endpoints "$CONFIG_PATH"
echo "--- END: test_admin_endpoints completed ---"

# Test hot_reload to add user2
echo "--- BEGIN: Running hot_reload to add user2 ---"
USER2_CONFIG="conf_umrs_user2.json"
$PY_CLIENT --base_url "$BASE_URL" add_user_hot_reload "$USER2_CONFIG"
echo "--- END: hot_reload for user2 completed ---"

# Then, test all regular endpoints
test_all_endpoints "$CONFIG_PATH" "$DATA_DIR"

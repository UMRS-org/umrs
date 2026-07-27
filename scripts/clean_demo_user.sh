#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# clean_demo_user.sh — Wipe all demo_user state (server + local)
#
# Usage:
#   ./clean_demo_user.sh              # clean both server and local
#   ./clean_demo_user.sh --local      # local only
#   ./clean_demo_user.sh --server     # server only
#   DEMO_USER=other_user ./clean_demo_user.sh  # different username
# ──────────────────────────────────────────────────────────────
set -euo pipefail

USER_TO_CLEAN="${DEMO_USER:-demo_user}"
REMOTE="rccs-cloud"
DEPLOY_DIR="/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs/deploy/rccs-cloud"

do_local=true
do_server=true

case "${1:-}" in
  --local)  do_server=false ;;
  --server) do_local=false  ;;
  --help|-h)
    sed -n '2,8p' "$0" | sed 's/^# //'
    exit 0
    ;;
esac

# Bucket prefix: underscores→hyphens, lowercased
bucket_prefix="$(echo "$USER_TO_CLEAN" | tr '[:upper:]' '[:lower:]' | tr '_' '-')."

green() { printf '\033[32m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }

# ── Server cleanup ────────────────────────────────────────────
if $do_server; then
  bold "=== Server cleanup ($REMOTE) ==="

  # 1. Remove user JSON files
  echo "Removing user files..."
  ssh "$REMOTE" "rm -fv \
    $DEPLOY_DIR/user_files/${USER_TO_CLEAN}.json \
    $DEPLOY_DIR/user_files/${USER_TO_CLEAN}.client.json \
    ~/.config/umrs/user_files/${USER_TO_CLEAN}.json \
    ~/.config/umrs/user_files/${USER_TO_CLEAN}.client.json \
    2>/dev/null || true"

  # 2. Remove s3cfg
  echo "Removing s3cfg..."
  ssh "$REMOTE" "rm -fv ~/.config/umrs/s3cfg_${USER_TO_CLEAN} 2>/dev/null || true"

  # 3. Remove from conf_umrs.json
  echo "Cleaning conf_umrs.json..."
  ssh "$REMOTE" "python3 <<'PYEOF'
import json, sys, os
cfg_path = \"$DEPLOY_DIR/conf_umrs.json\"
with open(cfg_path) as f:
    cfg = json.load(f)
before = list(cfg.get(\"users\", []))
cfg[\"users\"] = [u for u in cfg.get(\"users\", []) if \"$USER_TO_CLEAN\" not in u]
if before != cfg[\"users\"]:
    with open(cfg_path, \"w\") as f:
        json.dump(cfg, f, indent=2)
        f.write(\"\\n\")
    print(\"  removed from conf_umrs.json\")
else:
    print(\"  not in conf_umrs.json (already clean)\")
PYEOF
"

  # 4. Delete MinIO buckets
  echo "Deleting MinIO buckets..."
  ssh "$REMOTE" "python3 <<'PYEOF'
import boto3
s3 = boto3.client('s3',
    endpoint_url='http://127.0.0.1:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    region_name='us-east-1',
)
prefix = '$bucket_prefix'
buckets = s3.list_buckets()['Buckets']
targets = [b['Name'] for b in buckets if b['Name'].startswith(prefix)]
if not targets:
    print('  no buckets to delete')
for bname in targets:
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bname):
        for obj in page.get('Contents', []):
            s3.delete_object(Bucket=bname, Key=obj['Key'])
    s3.delete_bucket(Bucket=bname)
    print('  deleted bucket %s' % bname)
PYEOF
"

  green "Server cleanup done."
fi

# ── Local cleanup ─────────────────────────────────────────────
if $do_local; then
  bold "=== Local cleanup ==="
  cfg_dir="${XDG_CONFIG_HOME:-$HOME/.config}/umrs"

  for f in \
    "$cfg_dir/users/${USER_TO_CLEAN}.json" \
    "$cfg_dir/users/${USER_TO_CLEAN}.client.json" \
    "$cfg_dir/s3cfg_${USER_TO_CLEAN}"; do
    if [ -f "$f" ]; then
      rm -v "$f"
    else
      dim "  (not found: $f)"
    fi
  done

  # Reset current_user_config pointer if it points to this user
  pointer="$cfg_dir/current_user_config"
  if [ -f "$pointer" ] && grep -q "$USER_TO_CLEAN" "$pointer" 2>/dev/null; then
    rm -v "$pointer"
    echo "  (pointer reset — run 'umrs login' to re-authenticate)"
  fi

  green "Local cleanup done."
fi

echo ""
bold "Clean state for '$USER_TO_CLEAN'. To re-onboard:"
echo "  1. Server: cd $DEPLOY_DIR && umrs-server admin user add $USER_TO_CLEAN --role user --slurm-account rccs-dmp"
echo "  2. Client: umrs login --host <HOST> --api-key <TOKEN>"
echo "  3. Add experiences: umrs config import <experience.json>"
echo "  4. Reload: umrs reload"

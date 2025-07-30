#!/bin/bash
# init_postgres_users.sh
# Script to create additional PostgreSQL users (user1, user2) after deployment.
# Usage: ./init_postgres_users.sh [PGHOST] [PGPORT] [PGADMINUSER] [PGADMINPASS]

set -e

PGHOST=${1:-postgres}
PGPORT=${2:-5432}
PGADMINUSER=${3:-postgres}
PGADMINPASS=${4:-postgres}

# User definitions (username:password)
USERS=(
  "user1:user1pass"
  "user2:user2pass"
)

export PGPASSWORD="$PGADMINPASS"

for entry in "${USERS[@]}"; do
  IFS=":" read -r username userpass <<< "$entry"
  echo "Creating PostgreSQL user: $username"
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGADMINUSER" -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='$username'" | grep -q 1 || \
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGADMINUSER" -d postgres -c "CREATE USER $username WITH PASSWORD '$userpass';"
  # Optionally, grant privileges or create databases for each user:
  # psql -h "$PGHOST" -p "$PGPORT" -U "$PGADMINUSER" -d postgres -c "CREATE DATABASE ${username}_db OWNER $username;"
  # psql -h "$PGHOST" -p "$PGPORT" -U "$PGADMINUSER" -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE ${username}_db TO $username;"
done

echo "PostgreSQL users created successfully."

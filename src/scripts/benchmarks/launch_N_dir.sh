#!/bin/bash

N=$1

if [ -z "$N" ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

# Répertoire du script (benchmark/)
BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"

# Fichier de log
LOG_DIR="$BENCHMARK_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xp_dir_2_${N}.log"

# Supprimer le log s'il existe déjà
rm -f "$LOG_FILE"

# Lancer le monitor en tâche de fond
python "$BENCHMARK_DIR/monitor_users.py" "$N" --log-file "$LOG_FILE" &
MONITOR_PID=$!

# Attendre que le monitor ait fait son premier poll (il crée le fichier)
echo "Waiting for monitor to complete first poll..."
while [ ! -f "$LOG_FILE" ]; do
    sleep 0.5
done
echo "Monitor ready."

# Lancer les N clients en parallèle
# Date calculée juste avant le lancement
echo "Started: $(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])')" >> "$LOG_FILE"

for i in $(seq 1 "$N"); do
    ./client_umrs.py \
        -c ~/.config/umrs/n_users/users/user1_${i}.json \
        --output text \
        upload_dir --blocking ../data/img bucket-3/new-xp_5_-$N > /dev/null &
    sleep 1
done
echo "All client launched"

#!/bin/bash

N=$1
RUN=${2:-1}

if [ -z "$N" ]; then
    echo "Usage: $0 <N> [run]"
    exit 1
fi

# Sous-répertoire unique par exécution : nouvelles clés S3, donc
# filter_already_indexed ne saute rien d'une exécution à l'autre
TAG="$(date +%Y%m%d-%H%M%S)-N${N}-run${RUN}"

# Répertoire du script (benchmark/)
BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"

# Fichier de log
LOG_DIR="$BENCHMARK_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xp_dir_${N}_run${RUN}.log"

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
echo "Tag: $TAG" >> "$LOG_FILE"

# Identification de la machine (clients ; le serveur peut tourner ailleurs)
echo "Host: $(uname -a)" >> "$LOG_FILE"
CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null \
    || lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p')
[ -n "$CPU_MODEL" ] && echo "CPU: $CPU_MODEL" >> "$LOG_FILE"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/GPU: /' >> "$LOG_FILE"
fi
echo "Started: $(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])')" >> "$LOG_FILE"

for i in $(seq 1 "$N"); do
    ./client_umrs.py \
        -c ~/.config/umrs/n_users/users/user1_${i}.json \
        --output text \
        upload_dir --blocking ../data/img "bucket-3/${TAG}" > /dev/null &
done
echo "All client launched"
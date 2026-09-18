#!/bin/bash

N=$1

if [ -z "$N" ]; then
    echo "Usage: $0 <N>"
    exit 1
fi


# Lancer les N clients en parallèle
# Date calculée juste avant le lancement
echo "Started: $(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])')" >> "$LOG_FILE"
for i in $(seq 1 "$N"); do
    ./client_umrs.py \
        -c ~/.config/umrs/n_users/users/user1_${i}.json \
        --output text \
        upload_dir --blocking ../data/img bucket-3 &
done

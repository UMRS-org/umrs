#!/bin/bash

CLIENT="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/src/client_umrs.py"
CONFIG="~/.config/umrs/n_users/users/user1_1.json"
IMAGE="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/data/img_test/cat.jpeg"
BUCKET_BASE="bucket-3"
LOG_DIR="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/src/scripts/log"
LOG_FILE="$LOG_DIR/knn_sweep.log"
REPEATS=10

K_VALUES=(1 2 4 8 16 32 64 128 256 512 1000)
N_VALUES=(1000 2000 3000 4000 5000 6000 7000)

mkdir -p "$LOG_DIR"
> "$LOG_FILE"  # clear log

total=$(( ${#K_VALUES[@]} * ${#N_VALUES[@]} * REPEATS ))
count=0

for rep in $(seq 1 $REPEATS); do
    for n in "${N_VALUES[@]}"; do
        for k in "${K_VALUES[@]}"; do
            count=$((count + 1))
            echo "[$count/$total] rep=$rep n=$n k=$k"

            echo "=== rep=$rep n=$n k=$k ===" >> "$LOG_FILE"
            { time $CLIENT \
                -c $CONFIG \
                --output text \
                knn $k $IMAGE ${BUCKET_BASE}/${n} ; } >> "$LOG_FILE" 2>&1
            echo "" >> "$LOG_FILE"
        done
    done
done

echo "Done. Results in $LOG_FILE"

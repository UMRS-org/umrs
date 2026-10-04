#!/bin/bash

CLIENT="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/src/client_umrs.py"
IMAGE="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/data/img_test/cat.jpeg"
CONFIG_DIR="/home/emmanuel.jeannot/.config/umrs/n_users/users"
LOG_DIR="/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/src/scripts/benchmarks/log"
LOG_FILE="$LOG_DIR/knn_cold_warm.log"

CLIENTS=(user1_1 user1_2 user1_3 user1_10 user1_11 user1_12 user1_13)
K_VALUES=(5 20 200)
BUCKET="bucket-3"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

for user in "${CLIENTS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for call in cold hot; do
            echo "=== user=$user k=$k call=$call ===" | tee -a "$LOG_FILE"
            { time python "$CLIENT" \
                -c "$CONFIG_DIR/${user}.json" \
                --output text \
                knn $k "$IMAGE" $BUCKET ; } >> "$LOG_FILE" 2>&1
            echo "" >> "$LOG_FILE"
        done
    done
done

echo "Done. Results in $LOG_FILE"

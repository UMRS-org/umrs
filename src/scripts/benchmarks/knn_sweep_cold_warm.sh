#!/bin/bash
# KNN cold/hot sweep: for each k, repeat, user, run the same KNN query twice
# (cold, then hot) and keep the "KNN results" and `time` lines.
# Upstream shipped this as knn_sweep_cold_warm.py (a bash script); paths that
# were hard-coded there are environment overrides here.
#   UMRS_BENCH_CLIENT      client_umrs.py            (default: <repo>/src/client_umrs.py)
#   UMRS_BENCH_IMAGE       query image               (default: <repo>/data/img_test/cat.jpeg)
#   UMRS_BENCH_CONFIG_DIR  user1_<i>.json configs    (default: ~/.config/umrs/n_users/users)
#   UMRS_BENCH_LOG_DIR     output directory          (default: <this dir>/log)
# Needs the KNN client command (client_umrs.py knn).

BENCHMARK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$BENCHMARK_DIR/../../.." && pwd)"

CLIENT="${UMRS_BENCH_CLIENT:-$REPO_DIR/src/client_umrs.py}"
IMAGE="${UMRS_BENCH_IMAGE:-$REPO_DIR/data/img_test/cat.jpeg}"
CONFIG_DIR="${UMRS_BENCH_CONFIG_DIR:-$HOME/.config/umrs/n_users/users}"
LOG_DIR="${UMRS_BENCH_LOG_DIR:-$BENCHMARK_DIR/log}"
LOG_FILE="$LOG_DIR/knn_cold_warm_redo.log"

CLIENTS=(user1_7 user1_8 user1_9 user1_10 user1_11 user1_12 user1_13 user1_14)
K_VALUES=(5 20 200)
BUCKET="bucket-3"
REPEATS=100

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

for k in "${K_VALUES[@]}"; do
    for rep in $(seq 1 $REPEATS); do
        for user in "${CLIENTS[@]}"; do
            for call in cold hot; do
                echo "=== rep=$rep user=$user k=$k call=$call ===" >> "$LOG_FILE"
                { time python "$CLIENT" \
                    -c "$CONFIG_DIR/${user}.json" \
                    --output text \
                    knn $k "$IMAGE" $BUCKET ; } 2>&1 | \
                    grep -E "KNN results|^real" >> "$LOG_FILE"
                echo "" >> "$LOG_FILE"
            done
        done
    done
done

echo "Done. Results in $LOG_FILE"

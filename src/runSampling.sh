#!/usr/bin/env bash
set -euo pipefail

# Sequential sampling runner (no screens), matching runExperiment.sh style

# Configure what to run
EXPERIMENT_TYPES=("diabetes")   # add: "hepatitis" "kidney"
MODEL_NAMES=("gpt-4o" "gpt-4o-mini")          
SEEDS=(0 42 100 123 456)

# Resolve path to the Python entrypoint relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_ENTRY="${SCRIPT_DIR}/run_sampling.py"

if [[ ! -f "$PY_ENTRY" ]]; then
    echo "Error: Python entrypoint not found at $PY_ENTRY" >&2
    exit 1
fi

start_ts=$(date +%s)
for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[RUN] sampling experiment=${experiment_type} model=${model_name} seed=${seed}"
            python "$PY_ENTRY" "$experiment_type" "$model_name" "$seed"
            echo "[DONE] sampling experiment=${experiment_type} model=${model_name} seed=${seed}"
        done
    done
done

end_ts=$(date +%s)
echo "All sampling runs completed in $(( end_ts - start_ts ))s."
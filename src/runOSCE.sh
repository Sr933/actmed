#!/usr/bin/env bash
set -euo pipefail

# Sequential OSCE runner (no screens), matching runExperiment.sh style

# Configure model names and seeds
MODEL_NAMES=("gpt-4o" "gpt-4o-mini")
SEEDS=(0 42 100 123 456)

# Resolve path to the Python entrypoint relative to this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_ENTRY="${SCRIPT_DIR}/runOSCE.py"

if [[ ! -f "$PY_ENTRY" ]]; then
    echo "Error: Python entrypoint not found at $PY_ENTRY" >&2
    exit 1
fi

start_ts=$(date +%s)
for model_name in "${MODEL_NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "[RUN] OSCE model=${model_name} seed=${seed}"
        python "$PY_ENTRY" "$model_name" "$seed"
        echo "[DONE] OSCE model=${model_name} seed=${seed}"
    done
done

end_ts=$(date +%s)
echo "All OSCE runs completed in $(( end_ts - start_ts ))s."
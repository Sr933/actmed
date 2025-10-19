#!/bin/bash
# filepath: /mnt/pdata/silas/Sepsis/src/runSampling.sh

# Define the experiment types, model names, and seeds
EXPERIMENT_TYPES=("diabetes")
MODEL_NAMES=("gpt-4o" "gpt-4o-mini")
SEEDS=(0 42 100 123 456)

# Path to conda.sh
CONDA_PATH="/home/sr933/miniconda/etc/profile.d/conda.sh"

# Function to run a sampling test in a new screen
run_sampling() {
    local experiment_type=$1
    local model_name=$2
    local seed=$3
    screen -dmS "${experiment_type}_${model_name}_${seed}_sampling2" bash -c "
        source $CONDA_PATH && \
        conda activate actmed && \
        python /home/sr933/BayesianReasoning/src/TestSampling.py ${experiment_type} ${model_name} ${seed} && \
        exec bash
    "
}

# Start all sampling tests in parallel
for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Starting sampling for $experiment_type with $model_name and seed $seed..."
            run_sampling "$experiment_type" "$model_name" "$seed"
        done
    done
done

# Wait for all screens to finish
echo "Waiting for all sampling screens to complete..."
while screen -ls | grep -q Detached; do
    sleep 600  # Check every 600 seconds if screens are still running
done

# Close all screens
echo "Closing all sampling screens..."
screen -ls | grep Detached | awk '{print $1}' | xargs -I {} screen -S {} -X quit

echo "All sampling tests completed and screens closed."
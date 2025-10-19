#!/bin/bash

EXPERIMENT_TYPES=("osce")
#MODEL_NAMES=("gpt-4.1-mini" "gpt-4o-mini")
SEEDS=(0 42 100 123 456)

MODEL_NAMES=("gpt-4o" "gpt-4o-mini")

# Path to conda.sh
CONDA_PATH="/home/sr933/miniconda/etc/profile.d/conda.sh"

# Function to run an experiment in a new screen
run_experiment() {
    local model_name=$2
    local seed=$3
    screen -dmS "${experiment_type}_${model_name}_${seed}" bash -c "source $CONDA_PATH && conda activate actmed && python /home/sr933/BayesianReasoning/src/osceexperiment.py $model_name $seed; exec bash"
}

# Start all experiments in parallel
for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Starting $experiment_type with $model_name and seed $seed..."
            run_experiment $experiment_type $model_name $seed
        done
    done
done

# Wait for all screens to finish
echo "Waiting for all experiments to complete..."
while screen -ls | grep -q Detached; do
    sleep 600  # Check every 600 seconds if screens are still running
done

# Close all screens
echo "Closing all screens..."
screen -ls | grep Detached | awk '{print $1}' | xargs -I {} screen -S {} -X quit

echo "All experiments completed and screens closed."
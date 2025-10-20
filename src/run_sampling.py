import sys
import os
import pandas as pd
import random

# (If needed, adjust import paths according to your folder structure)
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_folder = os.path.join(parent_folder, 'lib')
sys.path.append(lib_folder)

from datasets import (
    KidneyDataset, DiabetesDataset, HepatitisDataset,
    
)
from bed import (
    KidneyBEDModel, DiabetesBEDModel, HepatitisBEDModel
)

results_folder = os.path.join(parent_folder, 'results', 'sampling')
os.makedirs(results_folder, exist_ok=True)

def get_sampling_config(model_name, seed):
    """
    Returns configuration for the sampling experiment, specifying:
      - The Dataset class
      - The BED model class
      - The data file
      - Output file for storing samples
      - Features to sample
    """
    return {
        "kidney": {
            "dataset_class": KidneyDataset,
            "model_class": KidneyBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'kidney', 'ckd.csv'),
            "output_file": os.path.join(results_folder, f'kidney_sampling_results_{model_name}_{seed}.csv'),
            "features_to_sample": ['bp','bgr','bu','sc','sod','pot','hemo','rc']  # example subset
        },
        "diabetes": {
            "dataset_class": DiabetesDataset,
            "model_class": DiabetesBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'diabetes', 'diabetes.csv'),
            "output_file": os.path.join(results_folder, f'diabetes_sampling_results_{model_name}_{seed}.csv'),
            "features_to_sample": ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']
        },
        "hepatitis": {
            "dataset_class": HepatitisDataset,
            "model_class": HepatitisBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'hepatitis', 'Hepatitis_subset.csv'),
            "output_file": os.path.join(results_folder, f'hepatitis_sampling_results_{model_name}_{seed}.csv'),
            "features_to_sample": ["ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]
        }
    }

def run_sampling_experiment(experiment_type, model_name, seed, N=10):
    """
    run_sampling_experiment orchestrates data loading, sampling,
    and result storage.

    Parameters:
    -----------
    experiment_type : str
        One of the dataset keys (kidney, diabetes, hepatitis, cirrhosis, stroke, heart)
    model_name : str
        Name of the model, e.g. 'gpt-4o' or 'gpt-4.1-mini'
    seed : int
        Random seed for reproducibility
    N : int
        Number of samples to draw for each feature per patient
    """
    config_dict = get_sampling_config(model_name, seed)
    if experiment_type not in config_dict:
        print(f"Invalid experiment type: {experiment_type}.")
        print("Choose from 'kidney', 'diabetes', 'hepatitis', 'cirrhosis', 'stroke', or 'heart'.")
        sys.exit(1)

    random.seed(seed)
    config = config_dict[experiment_type]
    dataset_class = config["dataset_class"]
    model_class = config["model_class"]
    data_file = config["data_file"]
    output_file = config["output_file"]
    features_to_sample = config.get("features_to_sample", [])

    print(f"=== Running sampling experiment ===")
    print(f"Dataset: {experiment_type} | Model: {model_name} | Seed: {seed} | N={N}")
    print(f"Loading data from {data_file}...")

    # Load and preprocess the dataset
    dataset = dataset_class(data_file)
    dataset.preprocess_data()
    DATASET_ITEMS = dataset.return_length()

    # Prepare a DataFrame to store all sampling results
    all_samples = []

    for patient_id in range(DATASET_ITEMS):
        row_data = dataset.get_item(patient_id)
        if not row_data:
            continue
        # row_data might differ depending on how your get_item() method structures the return
        # Typically: row_data = (feature_table, label)
        # We only need the feature_table for sampling

        # Convert feature_table to a DataFrame if it's not already
        if isinstance(row_data, tuple):
            feature_table, _ = row_data
        else:
            feature_table = row_data

        # For each feature in features_to_sample, treat that as the "unknown" to sample
        for feature in features_to_sample:
            # Create a known-features table by dropping the column we want to sample
            known_features = feature_table.drop(columns=[feature], errors="ignore")

            # We call sample_random_variable N times and store results
            # Initialize a row dictionary to store the results
            sample_row = {
                "PatientID": patient_id,
                "Dataset": experiment_type,
                "Model": model_name,
                "FeatureToSample": feature
            }

            # Initialize the BED model
            # If your BED model needs a dataframe in constructor, pass feature_table or dataset data
            bed_model = model_class(model_name, feature_table)

            # For i in [1..N], we sample
            for i in range(1, N+1):
                try:
                    sampled_val = bed_model.sample_random_variable(known_features, feature)
                except Exception as exc:
                    sampled_val = f"Error: {exc}"
                sample_row[f"Sample{i}"] = sampled_val

            # Append the result row to our all_samples list
            all_samples.append(sample_row)

        # Create a DataFrame from all collected samples
        df_results = pd.DataFrame(all_samples)
        df_results.to_csv(output_file, index=False)
    print(f"Sampling results saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python TestSampling.py <experiment_type> <model_name> <seed>")
        print("experiment_type: 'kidney', 'diabetes', 'hepatitis', 'cirrhosis', 'stroke', or 'heart'")
        print("model_name: Name of the model, e.g., 'gpt-4o'")
        print("seed: Random seed for reproducibility")
        sys.exit(1)

    experiment_type_arg = sys.argv[1].lower()
    model_name_arg = sys.argv[2]
    seed_arg = int(sys.argv[3])

    run_sampling_experiment(experiment_type_arg, model_name_arg, seed_arg, N=10)
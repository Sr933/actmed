import sys
import os
import pandas as pd
import random

# Add the parent folder to the system path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_folder = os.path.join(parent_folder, 'lib')
sys.path.append(lib_folder)
results_folder = os.path.join(parent_folder, 'results', 'main')
os.makedirs(results_folder, exist_ok=True)

from datasets import  DiabetesDataset, HepatitisDataset, KidneyDataset
from bed import DiabetesBEDModel, HepatitisBEDModel, KidneyBEDModel

def get_experiment_config(model_name, seed):
    """
    Generate the experiment configuration with filenames including the model name and seed.
    """
    return {
        "kidney": {
            "dataset_class": KidneyDataset,
            "model_class": KidneyBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'kidney', 'ckd.csv'),
            "output_file": os.path.join(results_folder, f'kidney_experiment_results_{model_name}_{seed}.csv'),
            "global_best_features": ['bp', 'sc', 'hemo'],
            "n_tries" : 3
        },
        "diabetes": {
            "dataset_class": DiabetesDataset,
            "model_class": DiabetesBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'diabetes', 'diabetes.csv'),
            "output_file": os.path.join(results_folder, f'diabetes_experiment_results_{model_name}_{seed}.csv'),
            "global_best_features": ['Glucose', 'BMI', 'Insulin'],
            "n_tries" : 3
        },
        "hepatitis": {
            "dataset_class": HepatitisDataset,
            "model_class": HepatitisBEDModel,
            "data_file": os.path.join(parent_folder, 'data', 'hepatitis', 'Hepatitis_subset.csv'),
            "output_file": os.path.join(results_folder, f'hepatitis_experiment_results_{model_name}_{seed}.csv'),
            "global_best_features": ['ALT', 'AST', 'BIL'],
            "n_tries" : 3
        }
     
        
    }

def run_experiment(experiment_type, model_name, seed):
    if experiment_type not in get_experiment_config(model_name, seed):
        print(f"Invalid experiment type: {experiment_type}. Choose from 'kidney', 'diabetes', 'hepatitis', 'heart', or 'cirrhosis'.")
        sys.exit(1)

    # Set the random seed
    random.seed(seed)
    print(f"Using random seed: {seed}")

    config = get_experiment_config(model_name, seed)[experiment_type]
    dataset_class = config["dataset_class"]
    model_class = config["model_class"]
    data_file = config["data_file"]
    output_file = config["output_file"]

    print(f"Running {experiment_type} experiment with model {model_name}...")
    print(f"Loading data from {data_file}...")

    # Load and preprocess the dataset
    dataset = dataset_class(data_file)
    dataset.preprocess_data()

    DATASET_ITEMS = dataset.return_length()
    MAX_TRIES = config.get("n_tries", 3)  # Default to 3 tries if not specified

    # DataFrame to store results
    results = None
    for i in range(DATASET_ITEMS):
        try:
            print(f"Processing patient {i+1}...")
            # Get feature table and label for the current patient
            feature_table, label = dataset.get_item(i)
            base_known_features, base_unknown_features = dataset.return_feature_names()
            bayesian_known_features, bayesian_unknown_features = base_known_features.copy(), base_unknown_features.copy()
            implicit_feature_table = feature_table[bayesian_known_features].copy()
            bayesian_feature_table = feature_table[bayesian_known_features].copy()
            random_known_features, random_unknown_features = base_known_features.copy(), base_unknown_features.copy()
            implicit_known_features, implicit_unknown_features = base_known_features.copy(), base_unknown_features.copy()

            # Initialize the model
            model = model_class(model_name, feature_table)

            # Predict risk using all features (baseline)
            risk_all = model.predict_risk(feature_table)
            
            

            for trie in range(1, MAX_TRIES + 1):
                # Bayesian feature selection
                bayesian_feature, KL_div = model.select_feature(bayesian_feature_table, bayesian_unknown_features)
                if not bayesian_feature:
                    print(f"Error: Bayesian feature selection failed for patient {i+1}")
                    continue
                if bayesian_feature not in bayesian_known_features:
                    bayesian_known_features.append(bayesian_feature)
                if bayesian_feature in bayesian_unknown_features:
                    bayesian_unknown_features.remove(bayesian_feature)


                # Update the sequential feature table without overriding previously known values
                bayesian_feature_table = feature_table[bayesian_known_features]

                
                bayesian_selection_risk = model.predict_risk(bayesian_feature_table)

                # Random feature selection
                random_feature = random.choice(random_unknown_features)
                if random_feature not in random_known_features:
                    random_known_features.append(random_feature)
                if random_feature in random_unknown_features:
                    random_unknown_features.remove(random_feature)

                random_feature_table = feature_table[random_known_features]
                risk_random = model.predict_risk(random_feature_table)

                # Global best feature selection
                best_features = config.get("global_best_features", None)[:trie]
                if best_features is None:
                    best_features = model.get_best_global_features(base_unknown_features, trie)
                if not best_features or any(f not in feature_table.columns for f in best_features):
                    print(f"Error: Invalid best features for patient {i+1}: {best_features}")
                    continue
                best_feature_table = feature_table[best_features + base_known_features]
                risk_global_best = model.predict_risk(best_feature_table)

                # Implicit feature selection
                implicit_feature = model.select_feature_implicit(implicit_feature_table, implicit_unknown_features)
                if implicit_feature not in implicit_known_features:
                    implicit_known_features.append(implicit_feature)
                if implicit_feature in implicit_unknown_features:
                    implicit_unknown_features.remove(implicit_feature)

                implicit_feature_table = feature_table[implicit_known_features]
                implicit_selection_risk = model.predict_risk(implicit_feature_table)

                # Append results for the current try
                result_row = {
                    "PatientID": i,
                    "Label": label,
                    "Try": trie,
                    "FullRisk": risk_all,
                    "BayesianRisk": bayesian_selection_risk,
                    "RiskRandom": risk_random,
                    "RiskGlobalBest": risk_global_best,
                    "ImplicitSelectionRisk": implicit_selection_risk,
                    "ImplicitFeature": implicit_feature,
                    "RandomFeature": random_feature,
                    "BayesianFeature": bayesian_feature,
                    "KL_div": KL_div,
                    "GlobalBestFeature": best_features,
                }
                if results is None:
                    results = pd.DataFrame([result_row])  # Initialize results
                else:
                    results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)

            # Save results to CSV
            results.to_csv(output_file, index=False)
        except Exception as e:
            print(f"Error processing patient {i+1}: {e}")
            # Optionally, you can log the error or save it to a separate file
            with open(os.path.join(results_folder, 'error_log.txt'), 'a') as error_log:
                error_log.write(f"Error processing patient {i+1}: {e}\n")
                error_log.write(f"Traceback: {sys.exc_info()}\n")
            continue


    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_experiment.py <experiment_type> <model_name> <seed>")
        print("experiment_type: 'kidney', 'diabetes', 'hepatitis'")
        print("model_name: Name of the model to use (e.g., 'gpt-4o-mini')")
        print("seed: Random seed for reproducibility")
        sys.exit(1)

    experiment_type = sys.argv[1].lower()
    model_name = sys.argv[2]
    seed = int(sys.argv[3])
    run_experiment(experiment_type, model_name, seed)
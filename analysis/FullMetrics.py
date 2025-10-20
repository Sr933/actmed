import pandas as pd
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)

# Config
datasets = ["kidney", "hepatitis", "diabetes", "osce"]
models = ["gpt-4o-mini", "gpt-4o"]
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')
results_dir = os.path.join(parent_folder, 'results', 'main')
os.makedirs(plot_folder, exist_ok=True)

methods = ["Bayesian", "Random", "Global Best", "Implicit", "All Features"]
pred_cols = {
    "Bayesian": "BayesianRisk",
    "Random": "RiskRandom",
    "Global Best": "RiskGlobalBest",
    "Implicit": "ImplicitSelectionRisk",
    "All Features": "FullRisk"
}

def calculate_metrics(y_true, y_prob):
    y_pred = np.round(y_prob)
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {key: (np.mean([m[key] for m in metrics_list]), np.std([m[key] for m in metrics_list])) for key in keys}

# Loop through each model
for model in models:
    rows = []
    for dataset in datasets:
        for method in methods:
            metric_vals = []
            for seed in [0, 42, 100, 123, 456]:
                file_path = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    num_tries = 5 if dataset == "cirrhosis" else 3
                    df = df[df["Try"] == num_tries]
                    if pred_cols[method] in df.columns and not df.empty:
                        try:
                            y_true = df["Label"]
                            y_prob = df[pred_cols[method]]
                            metrics = calculate_metrics(y_true, y_prob)
                            metric_vals.append(metrics)
                        except ValueError:
                            continue  # skip if metrics fail due to only one class etc.

            if metric_vals:
                agg = aggregate_metrics(metric_vals)
                row = {
                    "Model": model,
                    "Dataset": dataset,
                    "Method": method
                }
                for metric in agg:
                    mean, std = agg[metric]
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = std
                rows.append(row)

    # Save per model
    output_path = os.path.join(plot_folder, f"{model}_metrics_summary.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)

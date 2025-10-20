import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon, cdist
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import matplotlib.pyplot as plt


def kl_divergence(p, q, eps=1e-8):
    """KL divergence with smoothing to avoid division by zero."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= np.sum(p)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))

def compute_distribution_metrics(original_values, sampled_values, bins=30):
    """
    Compute Wasserstein, Energy Distance, and Maximum Mean Discrepancy (MMD)
    between the actual (ground truth) and sampled distributions.
    """
    # Wasserstein distance
    wasserstein = stats.wasserstein_distance(original_values, sampled_values)

    # Energy Distance
    energy_distance = stats.energy_distance(original_values, sampled_values)

    # Maximum Mean Discrepancy (MMD) using Gaussian kernel
    gamma = 1.0 / (2 * np.var(np.concatenate([original_values, sampled_values])))
    kernel_xx = rbf_kernel(original_values[:, None], original_values[:, None], gamma=gamma)
    kernel_yy = rbf_kernel(sampled_values[:, None], sampled_values[:, None], gamma=gamma)
    kernel_xy = rbf_kernel(original_values[:, None], sampled_values[:, None], gamma=gamma)
    mmd = np.mean(kernel_xx) + np.mean(kernel_yy) - 2 * np.mean(kernel_xy)

    return wasserstein, energy_distance, mmd

# Load ground truth distributions from datasets
def load_ground_truth_distributions(dataset_name):
    """
    Load the ground truth distributions directly from the dataset files.
    """
    base_data_path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    dataset_paths = {
        "kidney": os.path.join(base_data_path, "kidney/ckd.csv"),
        "diabetes": os.path.join(base_data_path, "diabetes/diabetes.csv"),
        "hepatitis": os.path.join(base_data_path, "hepatitis/Hepatitis_subset.csv"),
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_file = dataset_paths[dataset_name]
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    # Read the dataset into a DataFrame
    df = pd.read_csv(data_file)

    # Extract ground truth distributions for all features
    ground_truth = {}
    for column in df.columns:
        ground_truth[column] = pd.to_numeric(df[column], errors="coerce").dropna().tolist()

    return ground_truth

def min_max_scale(values, min_val, max_val):
    """
    Scale a distribution using min-max scaling.
    """
    return (values - min_val) / (max_val - min_val)

def analyze_distribution_matching():
    datasets = ["kidney", "diabetes", "hepatitis"]
    models = ["gpt-4o", "gpt-4o-mini"]
    seeds = [0, 42, 100, 123, 456]
    results_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'results', 'sampling')
    output_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'plots')
    os.makedirs(output_folder, exist_ok=True)

    metrics = []

    for ds in datasets:
        # Load ground truth distributions
        ground_truth = load_ground_truth_distributions(ds)

        for model in models:
            for seed in seeds:
                filename = f"{ds}_sampling_results_{model}_{seed}.csv"
                filepath = os.path.join(results_folder, filename)
                if not os.path.isfile(filepath):
                    print(f"Skipping missing file: {filepath}")
                    continue

                df = pd.read_csv(filepath)
                if df.empty:
                    continue

                sample_cols = [c for c in df.columns if c.startswith("Sample")]

                # Compute metrics for the entire dataset distribution
                for feature in df["FeatureToSample"].unique():
                    feature_df = df[df["FeatureToSample"] == feature]

                    # Combine all sampled values for this feature
                    sampled_values = []
                    for col in sample_cols:
                        vals = pd.to_numeric(feature_df[col], errors="coerce").dropna().tolist()
                        sampled_values.extend(vals)

                    # Use ground truth distributions for comparison
                    actual_values = ground_truth.get(feature, [])

                    if not sampled_values or not actual_values:
                        continue

                    # Ensure numeric conversion for histogram computation
                    sampled_values = np.array(sampled_values, dtype=np.float64)
                    actual_values = np.array(actual_values, dtype=np.float64)

                    # Min-max scale both distributions
                    min_val, max_val = np.min(actual_values), np.max(actual_values)
                    if min_val == max_val:
                        print(f"Skipping feature {feature} for dataset {ds} due to zero range in ground truth.")
                        continue

                    sampled_values = min_max_scale(sampled_values, min_val, max_val)
                    actual_values = min_max_scale(actual_values, min_val, max_val)

                    w_dist, energy_dist, mmd = compute_distribution_metrics(actual_values, sampled_values)

                    metrics.append({
                        "Dataset": ds,
                        "Model": model,
                        "Seed": seed,
                        "PatientID": "ALL",
                        "Feature": feature,
                        "Wasserstein": w_dist,
                        "Energy_Distance": energy_dist,
                        "MMD": mmd
                    })

                # Group by PatientID to compute metrics per patient
                for patient_id, patient_df in df.groupby("PatientID"):
                    for feature, feat_df in patient_df.groupby("FeatureToSample"):
                        # Combine all sampled values for this feature
                        sampled_values = []
                        for col in sample_cols:
                            vals = pd.to_numeric(feat_df[col], errors="coerce").dropna().tolist()
                            sampled_values.extend(vals)

                        # Use ground truth distributions for comparison
                        actual_values = ground_truth.get(feature, [])

                        if not sampled_values or not actual_values:
                            continue

                        # Ensure numeric conversion for histogram computation
                        sampled_values = np.array(sampled_values, dtype=np.float64)
                        actual_values = np.array(actual_values, dtype=np.float64)

                        # Min-max scale both distributions
                        min_val, max_val = np.min(actual_values), np.max(actual_values)
                        if min_val == max_val:
                            print(f"Skipping feature {feature} for PatientID {patient_id} due to zero range in ground truth.")
                            continue

                        sampled_values = min_max_scale(sampled_values, min_val, max_val)
                        actual_values = min_max_scale(actual_values, min_val, max_val)

                        w_dist, energy_dist, mmd = compute_distribution_metrics(actual_values, sampled_values)

                        metrics.append({
                            "Dataset": ds,
                            "Model": model,
                            "Seed": seed,
                            "PatientID": patient_id,
                            "Feature": feature,
                            "Wasserstein": w_dist,
                            "Energy_Distance": energy_dist,
                            "MMD": mmd
                        })

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Calculate average and standard deviation by feature
    summary_df = metrics_df.groupby(["Dataset", "Model", "Feature"]).agg(
        Average_Wasserstein=("Wasserstein", "mean"),
        Std_Wasserstein=("Wasserstein", "std"),
        Average_Energy_Distance=("Energy_Distance", "mean"),
        Std_Energy_Distance=("Energy_Distance", "std"),
        Average_MMD=("MMD", "mean"),
        Std_MMD=("MMD", "std"),
    ).reset_index()

    # Save detailed metrics and summary
    metrics_path = os.path.join(output_folder, "distribution_metrics.csv")
    summary_path = os.path.join(output_folder, "distribution_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved detailed metrics to {metrics_path}")
    print(f"Saved summary metrics to {summary_path}")

    # Calculate and print average and standard deviation metrics per dataset for each model
    dataset_model_avg = metrics_df.groupby(["Dataset", "Model"]).agg(
        Average_Wasserstein=("Wasserstein", "mean"),
        Std_Wasserstein=("Wasserstein", "std"),
        Average_Energy_Distance=("Energy_Distance", "mean"),
        Std_Energy_Distance=("Energy_Distance", "std"),
        Average_MMD=("MMD", "mean"),
        Std_MMD=("MMD", "std")
    ).reset_index()

    print("\nAverage and standard deviation metrics per dataset for each model:")
    print(dataset_model_avg)

    avg_path = os.path.join(output_folder, "dataset_model_avg_metrics.csv")
    dataset_model_avg.to_csv(avg_path, index=False)
    print(f"Saved average and standard deviation metrics per dataset to {avg_path}")

    # Aggregate by feature/model for heatmaps (average over seeds)
    agg_df = summary_df.groupby(["Dataset", "Model", "Feature"], as_index=False).mean()

    for metric in ["Average_Wasserstein", "Average_Energy_Distance", "Average_MMD"]:
        pivot = agg_df.pivot_table(index="Feature", columns="Model", values=metric)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot,
            annot=True,
            cmap="mako",
            fmt=".3f",
            cbar_kws={"label": metric}
        )
        plt.title(f"{metric} Distance between Actual vs Sampled (averaged across seeds)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric.lower()}_heatmap.pdf"))
        plt.close()

if __name__ == "__main__":
    analyze_distribution_matching()

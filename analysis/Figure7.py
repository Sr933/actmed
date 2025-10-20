import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches

# Add the parent folder to the system path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')

def calculate_kl_divergence(prior, posterior):
    epsilon = 1e-10
    prior = np.clip(prior, epsilon, 1 - epsilon)
    posterior = np.clip(posterior, epsilon, 1 - epsilon)
    kl_values = posterior * np.log(posterior / prior) + (1 - posterior) * np.log((1 - posterior) / (1 - prior))
    return np.mean(kl_values)

def should_accept_test(p_prior, kl_observed, decision_boundary=0.5, gamma=0.5):
    delta = abs(p_prior - decision_boundary)

    if delta == 0:
        return True  # Maximum uncertainty

    if p_prior > decision_boundary:
        q_target = decision_boundary + (1 - gamma) * delta
    else:
        q_target = decision_boundary - (1 - gamma) * delta

    kl_threshold = calculate_kl_divergence(p_prior, q_target)
    return kl_observed >= kl_threshold

def analyze_kl_termination(data, decision_boundary=0.5, gamma=0.5):
    results = []
    id_col = "PatientID" if "PatientID" in data.columns else "CaseIndex"
    for patient_id, group in data.groupby(id_col):
        try1 = group[group["Try"] == 1]
        p_prior = try1["BayesianRisk"].values[0]
        true_label = group.iloc[-1]["Label"]
        selected_tries = [1]

        for _, row in group[group["Try"] > 1].iterrows():
            kl_div = row["KL_div"]
            if should_accept_test(p_prior, kl_div, decision_boundary=decision_boundary, gamma=gamma):
                p_posterior = row["BayesianRisk"]
                selected_tries.append(row["Try"])
                p_prior = p_posterior
            else:
                break

        final_prediction_try3 = group.iloc[-1]["BayesianRisk"]
        max_try_needed = max(selected_tries) if selected_tries else 1
        max_try_prediction = group.loc[group["Try"] == max_try_needed, "BayesianRisk"].values[0]

        results.append({
            # Always store the actual identifier value under a common key
            "PatientID": patient_id,
            # Keep which column was used (useful for joins later)
            "IdColumn": id_col,
            "TrueLabel": true_label,
            "TotalTries": len(group),
            "TriesNeeded": len(selected_tries),
            "FinalPredictionTry3": final_prediction_try3,
            "MaxTryPrediction": max_try_prediction,
            "SelectedTries": selected_tries
        })
    return pd.DataFrame(results)

def process_all_seeds(datasets, seeds, model, results_dir, gamma_values, decision_boundary=0.5):
    all_results = {gamma: [] for gamma in gamma_values}
    for dataset in datasets:
        for seed in seeds:
            results_file = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
            if os.path.exists(results_file):
                data = pd.read_csv(results_file)
                for gamma in gamma_values:
                    results = analyze_kl_termination(data, decision_boundary, gamma)
                    results["Dataset"] = dataset.capitalize()
                    results["Seed"] = seed
                    results["Gamma"] = gamma
                    all_results[gamma].append(results)
    return {gamma: pd.concat(lst, ignore_index=True) for gamma, lst in all_results.items() if lst}

def calculate_random_baseline_accuracy(results, model, results_dir):
    """
    Compute baseline accuracies for Implicit and Global Best selections using the
    per-patient predictions from the saved results files. Handles both PatientID
    and CaseIndex identifier schemes.
    """
    accuracy_data = []
    for dataset in results["Dataset"].unique():
        ds_lower = dataset.lower()
        for seed in results["Seed"].unique():
            subset = results[(results["Dataset"] == dataset) & (results["Seed"] == seed)]
            data_path = os.path.join(results_dir, f"{ds_lower}_experiment_results_{model}_{seed}.csv")
            if not os.path.exists(data_path):
                continue
            full_df = pd.read_csv(data_path)

            # Identify key column in the full results
            if "PatientID" in full_df.columns:
                id_full = "PatientID"
            elif "CaseIndex" in full_df.columns:
                id_full = "CaseIndex"
            else:
                continue

            preds_implicit = []
            preds_global = []
            for _, row in subset.iterrows():
                # We stored the actual identifier value in 'PatientID' and recorded the column name in 'IdColumn'
                patient_id = row["PatientID"]
                if patient_id is None:
                    continue
                patient_rows = full_df[full_df[id_full] == patient_id]
                if patient_rows.empty:
                    continue
                random_try = 3  # fixed try for comparability
                try_row = patient_rows[patient_rows["Try"] == random_try]
                if try_row.empty:
                    continue

                if "ImplicitSelectionRisk" in try_row.columns:
                    pred_imp = try_row["ImplicitSelectionRisk"].values[0]
                    preds_implicit.append((pred_imp >= 0.5) == row["TrueLabel"])

                if "RiskGlobalBest" in try_row.columns:
                    pred_gb = try_row["RiskGlobalBest"].values[0]
                    preds_global.append((pred_gb >= 0.5) == row["TrueLabel"])

            if preds_implicit:
                accuracy_data.append({
                    "Dataset": dataset,
                    "Seed": seed,
                    "Method": "Implicit",
                    "Accuracy": float(np.mean(preds_implicit)),
                })
            if preds_global:
                accuracy_data.append({
                    "Dataset": dataset,
                    "Seed": seed,
                    "Method": "Global Best",
                    "Accuracy": float(np.mean(preds_global)),
                })

    return pd.DataFrame(accuracy_data)

def plot_kl_accuracy_comparison_all_gammas(results_dict):
    sns.set(style="whitegrid", font_scale=1.7, rc={"axes.titlesize": 18})

    # Create a figure with two subplots side-by-side, sharing the y-axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    # Base palette keyed by Method (no Model)
    set2_colors = sns.color_palette("Set2")
    #Want gamma t influence strength of the blue
    method_palette = {
    "Implicit": set2_colors[0],       # Light green
    "Global Best": set2_colors[1],    # Orange
    "ACTMED γ=0.3": set2_colors[5],   # Light teal (less intense)
    "ACTMED γ=0.5": set2_colors[4],   # Mid blue
    "ACTMED γ=0.7": set2_colors[6],   # Deep teal / green-blue (most intense)
}

    # Build method-based legend patches
    method_handles = []
    for method_name, color_hex in method_palette.items():
        method_handles.append(mpatches.Patch(color=color_hex, label=method_name))

    # Plot each model on its own subplot
    for i, (model, gamma_results_dict) in enumerate(results_dict.items()):
        combined_accuracy_data = []

        # Gather data across all gamma values
        for gamma, results in gamma_results_dict.items():
            for dataset in results["Dataset"].unique():
                for seed in results["Seed"].unique():
                    subset = results[(results["Dataset"] == dataset) & (results["Seed"] == seed)]
                    if not subset.empty:
                        kl_accuracy = subset.apply(
                            lambda row: (row["MaxTryPrediction"] >= 0.5) == row["TrueLabel"],
                            axis=1
                        ).mean()
                        combined_accuracy_data.append({
                            "Dataset": dataset,
                            "Seed": seed,
                            "Method": f"ACTMED γ={gamma}",
                            "Accuracy": kl_accuracy
                        })

        # Random baseline from first gamma’s results
        if gamma_results_dict:
            first_gamma_df = next(iter(gamma_results_dict.values()))
            # Load baseline accuracies from the repo's results directory for this model
            repo_results_dir = os.path.join(parent_folder, 'results', 'main')
            random_baseline_acc = calculate_random_baseline_accuracy(first_gamma_df, model, repo_results_dir)
            for row in random_baseline_acc.to_dict("records"):
                combined_accuracy_data.append(row)

        # Convert to DataFrame
        df_plot = pd.DataFrame(combined_accuracy_data)

        # Plot with "Method" as the hue (no model name in the legend)
        sns.barplot(
    ax=axes[i],
    data=df_plot,
    x="Dataset",
    y="Accuracy",
    hue="Method",
    palette=method_palette,
    errorbar='sd',       # Keep standard deviation as the error
    errwidth=1.5,        # Thickness of the error bar line
    capsize=0.1,         # Add horizontal caps (side handles)
    legend=False,
    edgecolor='none',
)

        #axes[i].set_ylim(0, 1.1)
        axes[i].set_xlabel("Dataset")
        axes[i].set_ylabel("Accuracy" if i == 0 else "")
        axes[i].set_title(model.upper())
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)
        axes[i].set_ylim(0.3, 1.05)
    # Add one main legend for the methods at the top
    fig.legend(
        handles=method_handles,
        title="Method",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),  # Move it a bit up
        ncol=5,
        frameon=False,
    )

    # Adjust layout so the legend fits well
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save the figure
    os.makedirs(plot_folder, exist_ok=True)
    plt.savefig(os.path.join(plot_folder, "Figure7.pdf"), dpi=300, bbox_inches="tight")
    plt.show()

# === CONFIGURATION ===
datasets = [ "hepatitis", "diabetes", "kidney"]
seeds = [0, 42, 100, 123, 456]
decision_boundary = 0.5
gamma_values = [0.3, 0.5, 0.7]

# === MAIN EXECUTION ===
results_by_model = {}
for model in ["gpt-4o-mini", "gpt-4o"]:
    results_dir = os.path.join(parent_folder, 'results', 'main')
    gamma_results = process_all_seeds(datasets, seeds, model, results_dir, gamma_values, decision_boundary)

    print(f"\n{model.upper()} - Tests Needed per Dataset by Gamma:")
    for gamma, results in gamma_results.items():
        print(f"  γ = {gamma}")
        for dataset in results["Dataset"].unique():
            subset = results[results["Dataset"] == dataset]
            mean_tests = subset["TriesNeeded"].mean()
            std_tests = subset["TriesNeeded"].std()
            acc = subset.apply(lambda row: (row["MaxTryPrediction"] >= 0.5) == row["TrueLabel"], axis=1).mean()
            #Calculate acc stf
            acc_std = subset.apply(lambda row: (row["MaxTryPrediction"] >= 0.5) == row["TrueLabel"], axis=1).std()
            print(f"    {dataset}: {mean_tests:.2f} ± {std_tests:.2f} tests, Accuracy = {acc:.3f}, Std = {acc_std:.3f}")

    results_by_model[model] = gamma_results

plot_kl_accuracy_comparison_all_gammas(results_by_model)
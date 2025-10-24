import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Add the parent folder to the system path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')
os.makedirs(plot_folder, exist_ok=True)
# Aesthetics for NeurIPS-style plots
sns.set(style="whitegrid", font_scale=1.7, rc={"axes.titlesize": 18})

# Config
datasets = [ "diabetes", "kidney", "hepatitis", "osce"]
models = ["gpt-4o-mini", "gpt-4o"]
results_dir = os.path.join(parent_folder, 'results', 'main')


methods = ["ACTMED", "Random", "Global Best", "Implicit", "All Features"]
pred_cols = {
    "ACTMED": "BayesianRisk",
    "Random": "RiskRandom",
    "Global Best": "RiskGlobalBest",
    "Implicit": "ImplicitSelectionRisk",
    "All Features": "FullRisk"
}

def calculate_accuracy(results, methods, label_col="Label", pred_cols=None):
    accuracies = {method: [] for method in methods}
    for method, pred_col in pred_cols.items():
        if pred_col in results.columns:
            preds = results[pred_col].round()
            true_labels = results[label_col]
            accuracy = accuracy_score(true_labels, preds)
            #accuracy= roc_auc_score(true_labels, preds)
            accuracies[method].append(accuracy)
    return accuracies

def aggregate_accuracy(accuracy_list):
    aggregated = {}
    for method in accuracy_list[0].keys():
        values = [acc[method] for acc in accuracy_list if method in acc]
        aggregated[method] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }
    return aggregated

# Collect data per model
aggregated_all = {}
for model in models:
    aggregated_metrics = {}
    for dataset in datasets:
        accuracy_list = []
        seeds= [0, 42, 100, 123, 456]
        for seed in seeds:
            results_file = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
            if os.path.exists(results_file):
                results = pd.read_csv(results_file)
                num_tries =3
                results_try_3 = results[results["Try"] == num_tries]
                if not results_try_3.empty:
                    accuracies = calculate_accuracy(results_try_3, methods, pred_cols=pred_cols)
                    accuracy_list.append(accuracies)
        if accuracy_list:
            aggregated_metrics[dataset] = aggregate_accuracy(accuracy_list)
    # aggregated_all populated
    aggregated_all[model] = aggregated_metrics
# Print final accuracies
print("\nFinal accuracy summary per model and dataset:")
for mdl, metrics in aggregated_all.items():
    print(f"\nModel: {mdl}")
    for ds, vals in metrics.items():
        print(f" Dataset: {ds}")
        tbl = pd.DataFrame(vals).T.rename_axis("Method")
        tbl[['mean','std']] = tbl[['mean','std']].round(3)
        print(tbl.to_string())

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16,6), dpi=300, sharey=True)

bar_width = 0.15
x_positions = np.arange(len(datasets))
set2_colors = sns.color_palette("Set2")
colors = {
    "ACTMED": set2_colors[2],
    "Random": set2_colors[0],
    "Global Best": set2_colors[1],
    "Implicit": set2_colors[3],
    "All Features": set2_colors[4],
}


# Plot each model's results in its subplot
for ax, model in zip(axes, models):
    aggregated_metrics = aggregated_all[model]
    for i, method in enumerate(methods):
        
        means = [aggregated_metrics[dataset][method]["mean"] for dataset in datasets]
        stds = [aggregated_metrics[dataset][method]["std"] for dataset in datasets]
        
        ax.bar(
            x_positions + i * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            label=method,
            color=colors[method],
            edgecolor='none',  # Fully removes the black border
            capsize=3
        )
    ax.set_xticks(x_positions + (len(methods) - 1) * bar_width / 2)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_title(f"{model.upper()}")
    ax.set_xlabel("Dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0.3, 1.0)

axes[0].set_ylabel("Accuracy")

# Flat legend above both plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Methods", ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(plot_folder, "Figure5.pdf"), dpi=300, bbox_inches="tight")
plt.show()

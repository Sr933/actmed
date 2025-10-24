import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import sem
from sklearn.metrics import accuracy_score
# Style for NeurIPS
sns.set(style="whitegrid", font_scale=2, rc={"axes.titlesize": 16})
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Config
datasets = ["hepatitis", "diabetes", "osce", "kidney"]
models = ["gpt-4o-mini", "gpt-4o"]
results_dir = os.path.join(parent_dir, "results", "main")
output_dir = os.path.join(parent_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Load and collect data
all_data = []

for model in models:
    for dataset in datasets:
        for seed in [0, 42, 100, 123, 456]:
            results_file = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
            if os.path.exists(results_file):
                df = pd.read_csv(results_file)
                df["Model"] = model
                df["Dataset"] = dataset
                df["Seed"] = seed
                all_data.append(df)

# Combine
df_all = pd.concat(all_data, ignore_index=True)

# Compute accuracy for Bayesian predictions
df_all["correct"] = (df_all["Label"] == df_all["BayesianRisk"].round()).astype(int)

# Group by Model and Try to compute accuracy & 95% CI
summary_accuracy = (
    df_all
    .groupby(["Model", "Try"])
    .agg(mean_accuracy=("correct", "mean"),  # Compute mean accuracy
         sem_accuracy=("correct", sem))     # Compute standard error of the mean
    .reset_index()
)

# Plot accuracy
plt.figure(figsize=(12, 7), dpi=150)
set2_colors = sns.color_palette("Set2")
colors = {"gpt-4o": set2_colors[2], "gpt-4o-mini": set2_colors[1]}

for model in models:
    model_data = summary_accuracy[summary_accuracy["Model"] == model]
    plt.plot(model_data["Try"], model_data["mean_accuracy"], label=model.upper(), color=colors[model], marker='o')
    plt.fill_between(
        model_data["Try"],
        model_data["mean_accuracy"] - 1.96 * model_data["sem_accuracy"],
        model_data["mean_accuracy"] + 1.96 * model_data["sem_accuracy"],
        color=colors[model],
        alpha=0.3
    )

# Update x-axis to show only integers
plt.xticks(ticks=summary_accuracy["Try"].unique(), labels=summary_accuracy["Try"].unique())

# Update axis labels and title
plt.xlabel("Number of diagnostic tests")
plt.ylabel("Accuracy")
#plt.title("Bayesian Accuracy Across Diagnostic Tests", fontsize=22)

# Grid, legend, and save
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Model", loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bayesian_accuracy_over_tests.pdf"), dpi=300, bbox_inches="tight")
plt.show()

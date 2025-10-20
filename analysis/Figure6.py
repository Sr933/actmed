import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Add the parent folder to the system path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')

# ----- Load Data for Both Models -----
datasets = [ "hepatitis", "diabetes", "kidney", "osce"]
seeds = [0, 100, 42, 123, 456]
models = ["gpt-4o-mini", "gpt-4o"]
results_dir = os.path.join(parent_folder, 'results', 'main')

dataframes = {}
for model in models:
    model_data = {}
    for dataset in datasets:
        dataset_results = []
        for seed in seeds:
            file_path = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Seed"] = seed
                dataset_results.append(df)
            else:
                print(f"Warning: File not found for {dataset}, seed {seed}, model {model}")
        if dataset_results:
            model_data[dataset] = pd.concat(dataset_results, ignore_index=True)
    dataframes[model] = model_data

# ----- Compute Match Rate (Per Seed) -----
plot_data = []

for model_name, model_dfs in dataframes.items():
    for dataset, df in model_dfs.items():
        seed_grouped = df.groupby("Seed")
        method_to_seed_scores = {"ACTMED": [], "Implicit": [], "Random": []}

        for seed, seed_df in seed_grouped:
            id="PatientID" if dataset!="osce" else "CaseIndex"
            patient_grouped = seed_df.groupby(id)
            patient_scores = {"ACTMED": [], "Implicit": [], "Random": []}

            for _, patient_df in patient_grouped:
                global_features = set(eval(patient_df["GlobalBestFeature"].iloc[-1]))
                for method in patient_scores:
                    if method == "ACTMED":
                        selected_features = set(patient_df["BayesianFeature"].dropna().values)
                    else:
                        selected_features = set(patient_df[f"{method}Feature"].dropna().values)
                    matches = len(selected_features.intersection(global_features))
                    patient_scores[method].append(matches / len(global_features))

            for method in method_to_seed_scores:
                seed_avg = sum(patient_scores[method]) / len(patient_scores[method]) if patient_scores[method] else 0
                method_to_seed_scores[method].append(seed_avg)

        for method in method_to_seed_scores:
            scores = pd.Series(method_to_seed_scores[method])
            plot_data.append({
                "Model": model_name.upper(),
                "Dataset": dataset.capitalize(),
                "Method": method,
                "Mean": scores.mean(),
                "Std": scores.std()
            })

plot_df = pd.DataFrame(plot_data)

# ----- Plot -----
sns.set(style="whitegrid", font_scale=1.7, rc={"axes.titlesize": 18})
fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150, sharey=True)
set2_colors = sns.color_palette("Set2")
method_palette = {
    "ACTMED":  set2_colors[2],
    "Implicit": set2_colors[3],
    "Random": set2_colors[0],
}

model_order = ["GPT-4O-MINI", "GPT-4O"]

for ax, model in zip(axes, model_order):
    model_data = plot_df[plot_df["Model"] == model]
    sns.barplot(
        data=model_data,
        x="Dataset",
        y="Mean",
        hue="Method",
        palette=method_palette,
        errorbar=None,
        edgecolor='none',
        ax=ax
    )

    for i, row in model_data.iterrows():
        x_pos = list(model_data["Dataset"].unique()).index(row["Dataset"])
        offset = {"ACTMED": -0.25, "Implicit": 0.0, "Random": 0.25}[row["Method"]]
        ax.errorbar(
            x=x_pos + offset,
            y=row["Mean"],
            yerr=row["Std"],
            fmt="none",
            c="black",
            capsize=3
        )

    ax.set_title(model)
    ax.set_ylabel("Feature Match Rate" if ax == axes[0] else "")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.get_legend().remove()

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
#Replace Ours with
fig.legend(handles, labels, title="Method", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = os.path.join(plot_folder, "Figure6.pdf")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
plt.show()
